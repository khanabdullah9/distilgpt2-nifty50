import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def predict_movement(date, open_val, high_val, low_val, model_path="nifty50_model"):
    print(f"Loading model on CPU from {model_path}...")
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
    print(f"Applying LoRA adapters from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval() # Set to evaluation mode

    # Round inputs to match training data noise reduction
    open_val = round(float(open_val), 2)
    high_val = round(float(high_val), 2)
    low_val = round(float(low_val), 2)

    # Prepare prompt to match the new training format
    prompt = f"Data: Date: {date}, Open: {open_val}, High: {high_val}, Low: {low_val}\nPrediction:"
    
    # Tokenize for CPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    print("Generating prediction on CPU with sampling...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            do_sample=True, # Enable sampling to avoid "only Down" bias
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 5. Extract the word after "Prediction:"
    # We look for "Up" or "Down" to ensure we don't return partial noise
    prediction_raw = response.split("Prediction:")[-1].strip()
    
    if "Up" in prediction_raw:
        return "Up"
    elif "Down" in prediction_raw:
        return "Down"
    
    return prediction_raw # Fallback for debugging if it still misses


if __name__ == "__main__":
    # Example input from recent data
    test_date = "2026-03-27"
    test_open = 23_173.55
    test_high = 23_186.10
    test_low = 22_819.60
    
    try:
        result = predict_movement(test_date, test_open, test_high, test_low)
        print(f"\nInput: Date: {test_date}, Open: {test_open}, High: {test_high}, Low: {test_low}")
        print(f"Predicted Movement: {result}")
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Note: Make sure you have trained the model and saved it to 'nifty50_model' first.")
