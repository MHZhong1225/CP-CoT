# local_models.py (Final, Architecturally Correct Version)
import torch
import gc
# Import the correct class for Vision-Language models
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import config
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory():
    """Clears GPU memory and runs garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_with_mllm(image_path: str, prompt: str) -> str:
    """
    Loads, uses, and then unloads the Qwen-VL model using the correct chat template and processor.
    """
    clear_memory()
    print("Loading Target MLLM (Qwen-VL)...")

    processor = AutoProcessor.from_pretrained(config.TARGET_MLLM_NAME, trust_remote_code=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.TARGET_MLLM_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    print("Target MLLM loaded. Generating...")

    # Correct Fix: Create a structured message list for the chat template.
    # Note that the prompt text comes BEFORE the image placeholder.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"}, 
            ]
        }
    ]

    # 1. Apply the chat template to get the formatted text prompt string.
    text_prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return "Error: Image file not found."
    
    # 2. Pass the formatted text and the image to the processor.
    inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(device)

    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
    # Decode and clean up the output
    input_token_len = inputs['input_ids'].shape[1]
    response_ids = output_ids[0][input_token_len:]
    decoded_output = processor.decode(response_ids, skip_special_tokens=True).strip()

    # --- Release memory ---
    del model
    del processor
    del inputs
    del output_ids
    del image
    clear_memory()
    print("Target MLLM unloaded.")
    
    return decoded_output
def generate_with_llm(prompt: str) -> str:
    """
    Loads, uses, and then unloads the Judge LLM (a text-only model).
    This part remains correct.
    """
    clear_memory()
    print("Loading Judge LLM...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.JUDGE_LLM_NAME)
    # AutoModelForCausalLM is correct here because the judge is a text-only model.
    model = AutoModelForCausalLM.from_pretrained(
        config.JUDGE_LLM_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        cache_dir="/home/icdm/.cache/huggingface/hub/",
        low_cpu_mem_usage=True
    ).to(device)
    
    print("Judge LLM loaded. Generating...")
    messages = [
        {"role": "system", "content": "You are a helpful and precise assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=False
    )
    response = outputs[0][input_ids.shape[-1]:]
    decoded_response = tokenizer.decode(response, skip_special_tokens=True)

    # --- Release memory ---
    del model
    del tokenizer
    clear_memory()
    print("Judge LLM unloaded.")
    
    return decoded_response