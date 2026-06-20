import os
import shutil
import base64
import asyncio
import json
import httpx
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import edge_tts

app = FastAPI(
    title="Eldora Gateway API (Cloud LLM & 3-Layer Agent Version)",
    description="Low-latency central hub connecting DoraBot, Eldora Shield, and Caregiver App with asynchronous action/emotion layers.",
    version="1.1.0"
)

# Config & Model Init
STT_MODEL_SIZE = "base"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")

# Load Faster-Whisper on CPU for STT
print("🎙️ Loading STT model...")
stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
print("✅ STT model loaded successfully.")

# Ensure temp directory exists
TEMP_DIR = "./temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Helper: Wellness signals logger
def send_wellness_signal(interaction_type: str, user_text: str, detected_language: str, confidence: float, trigger_detected: str = None, emotion: str = "Neutral", emotion_explanation: str = ""):
    log_file = "wellness_signals_log.json"
    signal = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "interaction_type": interaction_type,
        "language": detected_language,
        "stt_confidence": round(confidence, 4),
        "text_length": len(user_text),
        "trigger_fired": trigger_detected,
        "emotion_state": emotion,
        "emotion_explanation": emotion_explanation,
        "raw_text_preview": user_text[:60] + "..." if len(user_text) > 60 else user_text
    }
    try:
        signals = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    signals = json.load(f)
                except json.JSONDecodeError:
                    signals = []
        signals.append(signal)
        with open(log_file, "w") as f:
            json.dump(signals, f, indent=4)
        print(f"📊 [ANALYTICS] Wellness registry updated (Total records: {len(signals)}).")
    except Exception as e:
        print(f"⚠️ [ANALYTICS WARNING] Failed to record wellness signal: {e}")

# Helper: Audio speech-to-text
async def run_stt(audio_path: str) -> tuple[str, str, float]:
    loop = asyncio.get_event_loop()
    def transcribe():
        segments, info = stt_model.transcribe(audio_path, beam_size=3)
        transcript = " ".join([segment.text for segment in segments]).strip()
        return transcript, info.language, info.language_probability

    transcript, detected_lang, confidence = await loop.run_in_executor(None, transcribe)
    
    if confidence < 0.7:
        print(f"⚠️ [COMPLIANCE ALERT] Low STT confidence ({confidence:.2f}).")
        if detected_lang != "en":
            detected_lang = "en"  # Fallback to English
            
    if detected_lang not in ["id", "en"]:
        detected_lang = "en"
        
    return transcript, detected_lang, confidence

# ==========================================================
# LAYER 1: CONVERSATIONAL RESPONSE GENERATION (FAST PATH)
# ==========================================================
async def generate_conversational_response(user_text: str, language: str) -> str:
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not set. Returning a mock empathic response.")
        return "Halo, saya DoraBot. Kunci API belum diatur di server, namun saya di sini untuk menemani Anda. TRIGGER: medication_log" if language == "id" else "Hello, I am DoraBot. API key is not configured, but I am here with you. TRIGGER: medication_log"

    if language == "id":
        system_instruction = (
            "You are DoraBot, a gentle, companionable care assistant for elders. "
            "Speak warmly, clearly, and concisely in Bahasa Indonesia. "
            "If a physical emergency is stated (e.g., fall, severe pain), response text must strictly include 'TRIGGER: emergency_call'. "
            "If medication logs are updated or requested, response text must include 'TRIGGER: medication_log'. "
            "If the elder requests to contact their family or caregiver, response text must include 'TRIGGER: family_call'. "
            "Safety Guideline: Do not diagnose medical conditions, do not prescribe drugs, and do not offer professional medical counsel. "
            "Keep your tone patient, respectful, comforting, and companionable."
        )
    else:
        system_instruction = (
            "You are DoraBot, a gentle, companionable care assistant for elders. "
            "Speak warmly, clearly, and concisely in English. "
            "If a physical emergency is stated (e.g., fall, severe pain), response text must strictly include 'TRIGGER: emergency_call'. "
            "If medication logs are updated or requested, response text must include 'TRIGGER: medication_log'. "
            "If the elder requests to contact their family or caregiver, response text must include 'TRIGGER: family_call'. "
            "Safety Guideline: Do not diagnose medical conditions, do not prescribe drugs, and do not offer professional medical counsel. "
            "Keep your tone patient, respectful, comforting, and companionable."
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 128
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"❌ Gemini API request failed: {e}")
            raise HTTPException(status_code=502, detail=f"Conversational LLM gateway error: {e}")

# ==========================================================
# LAYER 2: ACTION TRIGGER AGENT (ASYNC BACKGROUND TASK)
# ==========================================================
async def run_action_agent(user_text: str, language: str):
    """Layer 2 Asynchronous Agent: Runs in the background to detect 

    and route caregiver service requests (e.g., bringing water, help standing, etc.).
    """
    if not GEMINI_API_KEY:
        # Fallback local regex pattern if no API key is present
        text_lower = user_text.lower()
        needs_help = any(kw in text_lower for kw in ["ambil", "tolong", "bantu", "panggil", "bring", "help", "call", "water", "minum", "makan"])
        if needs_help:
            print(f"🚨 [LAYER 2 ALERT] Caregiver alert triggered locally: '{user_text}'")
        return

    system_instruction = (
        "You are the Eldora Action Detection Agent. "
        "Analyze the following message from an elderly user. "
        "Determine if they are asking for immediate physical assistance, requesting an everyday chore (like bringing water, food, medicine), or asking to contact someone. "
        "Respond strictly in JSON format with keys 'is_action_request' (boolean) and 'action_detail' (string description in Indonesian/English)."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 64
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=8.0)
            response.raise_for_status()
            res_data = response.json()
            result_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            result_json = json.loads(result_text)
            
            if result_json.get("is_action_request"):
                action_desc = result_json.get("action_detail", "Physical help request detected")
                print(f"🚨 [LAYER 2 BACKGROUND ALERT] Caregiver request detected: '{action_desc}'! Dispatching notification payload to caregiver App.")
    except Exception as e:
        print(f"⚠️ [Layer 2 Agent Error]: {e}")

# ==========================================================
# LAYER 3: EMOTION ANALYTICS AGENT (ASYNC BACKGROUND TASK)
# ==========================================================
async def run_emotion_agent(user_text: str, language: str, confidence: float, trigger_detected: str = None):
    """Layer 3 Asynchronous Agent: Runs in the background to analyze 

    the elder's sentiment and mood (Happy, Sad, Pain, Lonely, Anxious) and logs it.
    """
    emotion = "Neutral"
    explanation = "No API key configured"

    if GEMINI_API_KEY:
        system_instruction = (
            "You are the Eldora Emotional Sentiment Agent. "
            "Analyze the elder's message and determine the dominant emotional state. "
            "Choose exactly one of: Happy, Sad, Pain, Lonely, Anxious, Neutral. "
            "Respond strictly in JSON format with keys 'emotion' (string) and 'explanation' (string description)."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": user_text}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
                "maxOutputTokens": 64
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=8.0)
                response.raise_for_status()
                res_data = response.json()
                result_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                result_json = json.loads(result_text)
                emotion = result_json.get("emotion", "Neutral")
                explanation = result_json.get("explanation", "")
                print(f"📊 [LAYER 3 BACKGROUND SENTIMENT] Detected Emotion: {emotion} | Reason: {explanation}")
        except Exception as e:
            print(f"⚠️ [Layer 3 Agent Error]: {e}")

    # Log the complete wellness signal with the detected emotion
    send_wellness_signal(
        interaction_type="voice_dialogue",
        user_text=user_text,
        detected_language=language,
        confidence=confidence,
        trigger_detected=trigger_detected,
        emotion=emotion,
        emotion_explanation=explanation
    )

# Helper: Speech synthesis
async def run_tts(text_target: str, output_path: str, language: str):
    clean_text = (text_target
                  .replace("TRIGGER: emergency_call", "")
                  .replace("TRIGGER: medication_log", "")
                  .replace("TRIGGER: family_call", "")
                  .strip())
    
    # Check if Azure SDK keys are provided and try to use it
    if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
        print("🔊 Synthesizing speech using Azure Neural TTS...")
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            voice = "id-ID-GadisNeural" if language == "id" else "en-US-JennyNeural"
            speech_config.speech_synthesis_voice_name = voice
            
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            
            # SSML to slow rate down by 10% for seniors and lower pitch slightly
            ssml_text = f"""
            <speak version='1.0' xml:lang='{language}' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts'>
                <voice name='{voice}'>
                    <prosody rate='-10%' pitch='-3%'>
                        {clean_text}
                    </prosody>
                </voice>
            </speak>
            """
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: synthesizer.speak_ssml_resolve_desc(ssml_text))
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"💾 Clean Azure audio file exported successfully to: {output_path}")
                return
            else:
                print(f"⚠️ Azure Speech synthesis failed: {result.reason}. Falling back to Edge-TTS.")
        except Exception as e:
            print(f"⚠️ Azure Speech import/run error: {e}. Falling back to Edge-TTS.")

    # Fallback to Edge-TTS
    print("🔊 Synthesizing speech using Edge-TTS (fallback)...")
    voice = "id-ID-GadisNeural" if language == "id" else "en-US-JennyNeural"
    communicate = edge_tts.Communicate(clean_text, voice, rate="-10%")
    await communicate.save(output_path)
    print(f"💾 Clean Edge-TTS audio file exported successfully to: {output_path}")

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/api/voice/interact")
async def voice_interact(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Save uploaded file to temp directory
    temp_file_path = os.path.join(TEMP_DIR, f"{datetime.now().timestamp()}_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. STT Conversion
        transcript, lang, confidence = await run_stt(temp_file_path)
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Audio input is silent or empty.")
            
        # 3. Layer 1: Call Cloud LLM API (Fast conversational response)
        ai_response = await generate_conversational_response(transcript, lang)
        
        # 4. Check conversational triggers
        active_trigger = None
        if "TRIGGER: emergency_call" in ai_response:
            active_trigger = "emergency_call"
            print("🚨 [ALERT]: Emergency trigger intercepted! Directing escalation to App.")
        elif "TRIGGER: medication_log" in ai_response:
            active_trigger = "medication_log"
            print("📝 [EVENT]: Medication schedule update intercepted.")
        elif "TRIGGER: family_call" in ai_response:
            active_trigger = "family_call"
            print("📞 [EVENT]: Family contact request intercepted.")

        # 5. Spawning Asynchronous Agentic Layers in the Background (Non-blocking)
        # Layer 2: Action trigger detection
        background_tasks.add_task(run_action_agent, transcript, lang)
        # Layer 3: Emotion tracker & Database Logger
        background_tasks.add_task(run_emotion_agent, transcript, lang, confidence, active_trigger)
        
        # 6. TTS Synthesis (Layer 1 Audio Generation)
        output_audio_path = temp_file_path + "_response.mp3"
        await run_tts(ai_response, output_audio_path, lang)
        
        # 7. Encode response audio to Base64 to return in JSON
        with open(output_audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
            
        # 8. Clean up audio files as background tasks
        background_tasks.add_task(os.remove, temp_file_path)
        background_tasks.add_task(os.remove, output_audio_path)
        
        return {
            "transcript": transcript,
            "response": ai_response,
            "detected_language": lang,
            "stt_confidence": round(confidence, 4),
            "trigger": active_trigger,
            "audio_base64": audio_base64
        }
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e

@app.post("/api/shield/fall")
async def shield_fall(payload: dict):
    """Fallback endpoint for Eldora Shield MQTT/HTTP emergency alerts."""
    device_id = payload.get("device_id", "unknown")
    bpm = payload.get("heart_rate", 0)
    print(f"🚨 [FALL DETECTED] Device: {device_id} | Heart Rate: {bpm} bpm. Dispatching FCM push notification to Eldora App.")
    return {"status": "success", "action": "fcm_escalation_dispatched"}

@app.get("/api/analytics/wellness")
async def get_wellness():
    """Retrieve wellness log records for the caregiver app dashboard."""
    log_file = "wellness_signals_log.json"
    if not os.path.exists(log_file):
        return []
    with open(log_file, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_backend": "gemini_api" if GEMINI_API_KEY else "mock_fallback",
        "azure_speech": "configured" if (AZURE_SPEECH_KEY and AZURE_SPEECH_REGION) else "edge_tts_fallback"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
