# local_models.py (优化后版本，已消除警告)
import torch
import gc
# 导入 GenerationConfig 来精确控制生成参数
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from PIL import Image
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory():
    """Clears GPU memory and runs garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- MLLM (视觉语言模型) 相关函数 ---

def load_mllm():
    """加载并返回 MLLM 模型和处理器"""
    print("Loading Target MLLM (Qwen-VL)...")
    processor = AutoProcessor.from_pretrained(config.TARGET_MLLM_NAME, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.TARGET_MLLM_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    print("Target MLLM loaded.")
    return model, processor

def generate_with_mllm(model, processor, image_path: str, prompt: str) -> str:
    """使用已加载的 MLLM 模型进行推理"""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}
    ]
    text_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return "Error: Image file not found."
    
    inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(device)
    
    # --- FIX START: 创建一个干净的生成配置 ---
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False
    )
    output_ids = model.generate(**inputs, generation_config=generation_config)
    # --- FIX END ---
    
    input_token_len = inputs['input_ids'].shape[1]
    response_ids = output_ids[0][input_token_len:]
    return processor.decode(response_ids, skip_special_tokens=True).strip()

# --- LLM (语言模型) 相关函数 ---
def load_llm():
    """加载并返回 LLM 模型和分词器"""
    print("Loading Judge LLM...")
    tokenizer = AutoTokenizer.from_pretrained(config.JUDGE_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.JUDGE_LLM_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("Judge LLM loaded.")
    return model, tokenizer

def generate_with_llm(model, tokenizer, prompt: str) -> str:
    """使用已加载的 LLM 模型进行推理"""
    messages = [
        {"role": "system", "content": "You are a helpful and precise assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, enable_thinking=False,add_generation_prompt=True, return_tensors="pt").to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") 
    ]
    eos_token_ids = [token for token in terminators if token is not None]

    # --- FIX START: 为 LLM 也创建一个干净的生成配置 ---
    generation_config = GenerationConfig(
        max_new_tokens=512,
        eos_token_id=eos_token_ids,
        do_sample=False
    )
    outputs = model.generate(input_ids, generation_config=generation_config)
    # --- FIX END ---

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)