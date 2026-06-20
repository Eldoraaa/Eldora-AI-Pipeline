import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_DIR = "../eldercare_adapter"  # Path to your adapter folder
OUTPUT_DIR = "../merged_eldercare_qwen2"

def merge_and_export():
    print("🚀 Starting model merging process...")
    
    if not os.path.exists(ADAPTER_DIR):
        print(f"❌ Error: LoRA adapter directory not found at: {ADAPTER_DIR}")
        print("Please place your 'eldercare_adapter' folder in the project directory before merging.")
        return
        
    print(f"📦 Loading base model '{BASE_MODEL}' in float16...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load base model in float16 (required for clean merging)
    # Removing device_map avoids accelerate's CPU offload hooks, preventing KeyError on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    )
    
    print(f"💡 Loading LoRA adapter weights from '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("🔄 Merging adapter weights permanently into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {OUTPUT_DIR}")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Merging complete!")
    print("\nNext Steps for Ollama conversion:")
    print("1. Convert the merged model to GGUF format using llama.cpp's convert_hf_to_gguf.py script.")
    print("2. Create a Modelfile pointing to the GGUF file and run: ollama create eldora-bot -f Modelfile")

if __name__ == "__main__":
    merge_and_export()
