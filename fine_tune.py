import torch
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

def main():
    # 1. Configuration
    model_name = "distilbert/distilgpt2"
    max_seq_length = 512
    device = "cpu"

    print(f"Loading model {model_name} on CPU...")
    
    # 2. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Standard loading on CPU (no quantization required for 16GB RAM)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device} # Force CPU
    )

    # 3. Add LoRA Adapters
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"], # Target modules for GPT2 attention
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. Data Preparation
    print("Loading dataset...")
    if not os.path.exists("train.txt"):
        print("Error: train.txt not found. Run data_prep.py first.")
        return

    dataset = load_dataset("text", data_files={"train": "train.txt"})

    # 5. Training
    print("Starting training on CPU (estimated 15-30 minutes)...")
    
    from trl import SFTConfig
    
    args = SFTConfig(
        output_dir = "outputs",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 500, # Increased for better pattern recognition
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = False,
        logging_steps = 10,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", # Smoother decay
        seed = 3407,
        report_to = "none",
        # SFT Specific parameters
        dataset_text_field = "text",
        max_length = max_seq_length,
    )

    # Define a formatting function for one example at a time
    def formatting_prompts_func(example):
        text = example['text']
        if "Output:" in text:
            parts = text.split("Output:")
            # Using a clearer structure: Data -> Prediction
            return f"Data: {parts[0].strip()}\nPrediction: {parts[1].strip()}<|endoftext|>"
        return text + "<|endoftext|>"

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset["train"],
        formatting_func = formatting_prompts_func,
        args = args,
    )

    trainer.train()
    print("Training completed.")

    # 6. Save the Model
    print("Saving model...")
    model.save_pretrained("nifty50_model")
    tokenizer.save_pretrained("nifty50_model")
    print("Model saved to 'nifty50_model' directory.")

if __name__ == "__main__":
    main()
