# local_models.py
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import config

# --- 全局模型变量，确保只加载一次 ---
target_mllm_model = None
target_mllm_processor = None
judge_llm_model = None
judge_llm_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """加载所有需要的本地模型到内存中"""
    global target_mllm_model, target_mllm_processor, judge_llm_model, judge_llm_tokenizer
    
    print("Loading Target MLLM...")
    target_mllm_processor = AutoProcessor.from_pretrained(config.TARGET_MLLM_NAME)
    target_mllm_model = LlavaForConditionalGeneration.from_pretrained(
        config.TARGET_MLLM_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    print("Target MLLM loaded.")

    print("Loading Judge LLM...")
    judge_llm_tokenizer = AutoTokenizer.from_pretrained(config.JUDGE_LLM_NAME)
    judge_llm_model = AutoModelForCausalLM.from_pretrained(
        config.JUDGE_LLM_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    print("Judge LLM loaded.")

def generate_with_mllm(image_path: str, prompt: str) -> str:
    """使用本地MLLM生成CoT描述"""
    if not target_mllm_model:
        load_models()
        
    raw_image = Image.open(image_path)
    # LLaVA的prompt格式通常是 "USER: <image>\n{prompt}\nASSISTANT:"
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    
    inputs = target_mllm_processor(full_prompt, images=raw_image, return_tensors="pt").to(device, torch.float16)
    
    output = target_mllm_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
    decoded_output = target_mllm_processor.decode(output[0], skip_special_tokens=True)
    assistant_response = decoded_output.split("ASSISTANT:")[1].strip()
    return assistant_response

def generate_with_llm(prompt: str) -> str:
    """使用本地LLM（裁判）进行文本生成或判断"""
    if not judge_llm_model:
        load_models()
        
    messages = [
        {"role": "system", "content": "You are a helpful and precise assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = judge_llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(judge_llm_model.device)

    terminators = [
        judge_llm_tokenizer.eos_token_id,
        judge_llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = judge_llm_model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False
    )
    response = outputs[0][input_ids.shape[-1]:]
    decoded_response = judge_llm_tokenizer.decode(response, skip_special_tokens=True)
    return decoded_response