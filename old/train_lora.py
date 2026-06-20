import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./eldercare_adapter"

# ==========================================
# 2. BILINGUAL ELDER-CARE DIALOGUE DATASET
# ==========================================
DIALOGUE_DATA = [
    {
        "instruction": "Aduh Budi, kaki simbah linu sekali buat berdiri.",
        "response": "Simbah istirahat dulu ya, jangan dipaksa berdiri. Budi ambilkan minyak urut dan bantu bersandar nyaman."
    },
    {
        "instruction": "Dora, kepala saya pusing sekali setelah bangun tidur.",
        "response": "Harap tetap berbaring tenang ya. Saya akan segera menghubungi pengasuh Budi untuk membawakan air hangat."
    },
    {
        "instruction": "I feel so lonely today, nobody came to see me.",
        "response": "I am right here with you. You are not alone. Tell me, would you like to hear a nice story or some music?"
    },
    {
        "instruction": "Dora, apa sudah waktunya saya minum obat jantung?",
        "response": "Mari kita cek jadwalnya. Iya betul, obat merah untuk jantung diminum setelah makan pagi ini. TRIGGER: medication_log"
    },
    {
        "instruction": "Saya terpleset di lantai kamar mandi dan tidak bisa bangun!",
        "response": "Tetap tenang dan jangan bergerak. Saya segera menghubungi ambulans dan pengasuh Anda sekarang! TRIGGER: emergency_call"
    },
    {
        "instruction": "Bisa panggilkan anak saya sekarang? Saya kangen sekali.",
        "response": "Tentu, saya akan segera menghubungkan panggilan telepon ke anak Anda. Harap tunggu sebentar. TRIGGER: family_call"
    }
]

def format_prompt(sample):
    messages = [
        {"role": "system", "content": "You are Eldora, a gentle, companionable care assistant for elders."},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["response"]}
    ]
    return messages

def train_lora_adapter():
    print("🚀 Starting Eldora LoRA Fine-Tuning Setup...")

    # Configure hardware settings dynamically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"🔥 Found GPU: {torch.cuda.get_device_name(0)}")
        print("📦 Loading base model in 4-bit QLoRA configuration...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        optim_type = "paged_adamw_32bit"
        fp16_active = True
    else:
        print("⚠️ CUDA GPU not found or PyTorch CUDA is inactive.")
        print("🔄 Falling back to CPU Training Mode (slower, but functional for 10-step demo).")
        print("📦 Loading base model in float32 on CPU...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Load standard model in FP32 on CPU (BitsAndBytes 4-bit is not supported on CPU)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}
        )
        optim_type = "adamw_torch"
        fp16_active = False

    # 2. Configure PEFT LoRA
    print("💡 Designing LoRA adapter layers...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Format Dataset
    print("📊 Formatting dialogue dataset...")
    formatted_dataset = []
    for item in DIALOGUE_DATA:
        chat = format_prompt(item)
        tokenized = tokenizer.apply_chat_template(chat, tokenize=False)
        formatted_dataset.append({"text": tokenized})

    dataset = Dataset.from_list(formatted_dataset)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Set Training Arguments (10 steps)
    print("📝 Configuring hyperparameters...")
    training_args = TrainingArguments(
        output_dir="./lora_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=10,  # Keeping training fast
        fp16=fp16_active,
        optim=optim_type,
        save_strategy="no"
    )

    # 5. Initialize Trainer
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 6. Execute Training
    print("🏋️ Running fine-tuning loop...")
    trainer.train()

    # 7. Save Adapter Weights to parent root folder
    parent_output_dir = "../eldercare_adapter"
    print(f"💾 Saving fine-tuned LoRA weights to parent directory '{parent_output_dir}'...")
    
    # Save model and tokenizer
    model.save_pretrained(parent_output_dir)
    tokenizer.save_pretrained(parent_output_dir)
    
    print("✅ Fine-tuning completed successfully! Your adapter is now ready for use.")
    print("Start the main.py server to load these weights.")

if __name__ == "__main__":
    train_lora_adapter()
