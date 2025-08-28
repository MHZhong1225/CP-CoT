# data_builder.py
import os
import json
import requests  # Or openai library
from PIL import Image
from tqdm import tqdm
import config
# import utils

# data_builder.py (部分修改)
import local_models # 导入新模块

def generate_cot_description(image_path: str) -> str:
    """调用本地目标MLLM生成CoT描述"""
    prompt = """请详细描述这张图片。请使用分步推理的方式，首先列出你观察到的关键视觉证据（至少3点），然后根据这些证据给出一个总结性描述。"""
    return local_models.generate_with_mllm(image_path, prompt)

def parse_cot_to_claims(cot_text: str) -> dict:
    """调用本地裁判模型(Llama-3)将CoT文本解析为原子断言"""
    prompt = f"请将以下文本解析为'视觉证据'和'总结描述'的断言列表，并以JSON格式返回：\n\n{cot_text}"
    response_text = local_models.generate_with_llm(prompt)
    # ... 此处需要添加解析JSON的代码，例如 utils.parse_json(response_text) ...
    return {"visual_evidence": ["..."], "summary_description": ["..."]} # 示例

def label_claim_validity(image_path: str, claim: str) -> bool:
    """
    调用本地模型判断断言的真伪。
    注意：由于裁判LLM无法直接看图，我们采用一种两步策略：
    1. 让MLLM生成一个简短的、高置信度的图像摘要。
    2. 让LLM判断断言是否与该摘要相符。
    """
    context_prompt = "请用一句话简短描述这张图片的核心内容。"
    context_description = local_models.generate_with_mllm(image_path, context_prompt)
    
    # 然后让LLM基于这个文字描述来判断
    prompt = f"背景描述：'{context_description}'\n\n请判断以下断言是否与背景描述相符且合理？\n断言：'{claim}'\n请只回答‘是’或‘否’。"
    response_text = local_models.generate_with_llm(prompt)
    return "是" in response_text


def build_calibration_set():
    """主函数，遍历图片，生成、解析、标注，最后保存数据集"""
    calibration_data = []
    image_files = [f for f in os.listdir(config.IMAGE_DIR) if f.endswith(('png', 'jpg'))]

    for image_file in tqdm(image_files, desc="Building Calibration Set"):
        image_path = os.path.join(config.IMAGE_DIR, image_file)

        # 1. 生成
        cot_text = generate_cot_description(image_path)
        
        # 2. 解析
        claims_dict = parse_cot_to_claims(cot_text)
        all_claims = claims_dict["visual_evidence"] + claims_dict["summary_description"]
        
        # 3. 标注
        is_valid_list = [label_claim_validity(image_path, claim) for claim in all_claims]
        final_is_valid = 1 if all(is_valid_list) else 0
        
        calibration_data.append({
            "image_path": image_path,
            "generated_cot": cot_text,
            "parsed_claims": claims_dict,
            "is_valid": final_is_valid
        })

    with open(config.CALIBRATION_SET_PATH, 'w', encoding='utf-8') as f:
        json.dump(calibration_data, f, ensure_ascii=False, indent=4)
    print(f"Calibration set built and saved to {config.CALIBRATION_SET_PATH}")

if __name__ == "__main__":
    build_calibration_set()