import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def predict_movement(date, open_val, high_val, low_val, sma20, sma50, rsi14, return_val, model_path="nifty50_model"):
    # 1. Load Base Model and Tokenizer
    base_model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        device_map={"": "cpu"},
        torch_dtype=torch.float32
    )
    
    # 2. Load LoRA Adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # 3. Prepare prompt to match the new training format
    # Format: Data: Date: ..., SMA20: ..., RSI14: ..., Return: ...%\nPrediction:
    prompt = (f"Data: Date: {date}, Open: {open_val}, High: {high_val}, Low: {low_val}, "
              f"SMA20: {sma20}, SMA50: {sma50}, RSI14: {rsi14}, Return: {return_val}%\nPrediction:")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            do_sample=False, # Keeping greedy decoding for stability
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction_raw = response.split("Prediction:")[-1].strip()
    
    if "Up" in prediction_raw:
        return "Up"
    elif "Down" in prediction_raw:
        return "Down"
    
    return prediction_raw
