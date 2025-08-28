# local_models.py (Memory-Efficient Version)
import torch
import gc
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import config

# --- We will no longer keep models in global variables to save memory ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory():
    """Clears GPU memory and runs garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# In local_models.py, replace the generate_with_mllm function with this:

def generate_with_mllm(image_path: str, prompt: str) -> str:
    """
    Loads, uses, and then unloads the Target MLLM to generate a description.
    """
    clear_memory()
    print("Loading Target MLLM...")
    
    processor = AutoProcessor.from_pretrained(config.TARGET_MLLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.TARGET_MLLM_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    print("Target MLLM loaded. Generating...")
    raw_image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                # The processor expects the prompt text first for Qwen2-VL
                {"type": "text", "text": prompt},
                {"type": "image"} 
            ]
        }
    ]
    
    # === CORRECTED BLOCK ===
    # Do NOT use apply_chat_template. Pass the messages and image directly to the processor.
    inputs = processor(text=messages, images=raw_image, return_tensors="pt").to(device)
    # === END CORRECTION ===
    
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
    # The decoding logic also needs to be simplified
    decoded_output = processor.decode(output[0], skip_special_tokens=True).strip()

    # The decoded output for Qwen2 includes the prompt, so we need to remove it.
    # This is a bit complex, we find where the user part ends and take the rest.
    try:
        # A common separator is '<|im_end|>\n<|im_start|>assistant\n'
        # We will split by the assistant's starting tag
        response_part = decoded_output.split("assistant\n")[-1]
    except:
        response_part = decoded_output # Fallback if split fails

    # --- Crucial Step: Release memory ---
    del model
    del processor
    clear_memory()
    print("Target MLLM unloaded.")
    
    return response_part

def generate_with_llm(prompt: str) -> str:
    """
    Loads, uses, and then unloads the Judge LLM for text-only tasks.
    """
    clear_memory()
    print("Loading Judge LLM...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.JUDGE_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.JUDGE_LLM_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    print("Judge LLM loaded. Generating...")
    messages = [
        {"role": "system", "content": "You are a helpful and precise assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False
    )
    response = outputs[0][input_ids.shape[-1]:]
    decoded_response = tokenizer.decode(response, skip_special_tokens=True)

    # --- Crucial Step: Release memory ---
    del model
    del tokenizer
    clear_memory()
    print("Judge LLM unloaded.")
    
    return decoded_response