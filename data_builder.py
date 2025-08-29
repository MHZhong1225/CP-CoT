# data_builder.py (最终完美版 - 强化解析器)
import os
import json
from tqdm import tqdm
import config
import local_models
import re # 导入正则表达式库

def python_parse_cot_to_claims(cot_text: str) -> dict:
    """
    一个稳定、高效且更强大的Python函数，用于直接从CoT文本中解析出断言。
    这个版本可以处理多种列表格式（-, *, 1., 2. 等）。
    """
    claims = {
        "visual_evidence": [],
        "summary_description": []
    }
    
    try:
        # 使用正则表达式来寻找两个关键部分
        visual_evidence_match = re.search(r"Key Visual Evidence:(.*?)Summary Description:", cot_text, re.DOTALL | re.IGNORECASE)
        summary_description_match = re.search(r"Summary Description:(.*)", cot_text, re.DOTALL | re.IGNORECASE)

        # --- FIX: 使用更强大的正则表达式来分割列表 ---
        # 这个表达式可以匹配 -, *, 或 数字后跟一个点 (e.g., 1., 2.)
        split_pattern = r'\n\s*(?:[-\*]|\d+\.)\s*'

        if visual_evidence_match:
            visual_text = visual_evidence_match.group(1).strip()
            visual_claims = re.split(split_pattern, visual_text)
            # 过滤掉可能产生的空字符串或无意义的短字符串
            claims["visual_evidence"] = [claim.strip() for claim in visual_claims if len(claim.strip()) > 5]

        if summary_description_match:
            summary_text = summary_description_match.group(1).strip()
            summary_claims = re.split(split_pattern, summary_text)
            claims["summary_description"] = [claim.strip() for claim in summary_claims if len(claim.strip()) > 5]
            
    except Exception as e:
        print(f"Error during Python parsing: {e}")

    return claims


def build_calibration_set():
    """Builds the calibration set using the final robust Python parser."""
    if os.path.exists(config.LLM_RAW_RESPONSE_LOG_PATH):
        os.remove(config.LLM_RAW_RESPONSE_LOG_PATH)

    all_image_files = [f for f in os.listdir(config.IMAGE_DIR) if f.endswith(('png', 'jpg'))]
    image_files = all_image_files
    # image_files = all_image_files[:5] # DEBUG: Process only 5 images first
    print(f"--- Running with all {len(image_files)} images ---")
    
    # --- STAGE 1: MLLM Generation ---
    print("--- STAGE 1: Generating all descriptions with MLLM ---")
    mllm, mllm_processor = local_models.load_mllm()
    generated_data = []
    cot_prompt = "Please describe this image in detail. Use a step-by-step reasoning approach: first, list the key visual evidence you observe (at least 3 points), then provide a summary description based on this evidence."
    context_prompt = "Please describe the core content of this image in one short sentence."
    for image_file in tqdm(image_files, desc="1/2 - Generating CoTs & Contexts"):
        image_path = os.path.join(config.IMAGE_DIR, image_file)
        cot_text = local_models.generate_with_mllm(mllm, mllm_processor, image_path, cot_prompt)
        context_description = local_models.generate_with_mllm(mllm, mllm_processor, image_path, context_prompt)
        generated_data.append({"image_path": image_path, "generated_cot": cot_text, "context_description": context_description})
    del mllm, mllm_processor
    local_models.clear_memory()
    print("--- MLLM Unloaded ---")
    with open(config.GENERATED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
    print(f"\n--- Intermediate generated data saved to {config.GENERATED_DATA_PATH} ---")

    # --- STAGE 2: Python Parsing and LLM Labeling ---
    print("\n--- STAGE 2: Python Parsing and LLM Labeling ---")
    llm, llm_tokenizer = local_models.load_llm()
    calibration_data = []
    with open(config.LLM_RAW_RESPONSE_LOG_PATH, 'a', encoding='utf-8') as log_file:
        for data in tqdm(generated_data, desc="2/2 - Parsing and Labeling"):
            data["parsed_claims"] = python_parse_cot_to_claims(data['generated_cot'])
            all_claims = data["parsed_claims"].get("visual_evidence", []) + data["parsed_claims"].get("summary_description", [])
            is_valid_list = []
            if not all_claims:
                is_valid_list.append(False)
            else:
                for claim in all_claims:
                    label_prompt = f"""Evaluate the following claim based *only* on the provided background description. Respond with a single word: "Yes" if the claim is consistent and reasonable, or "No" if it is not. Do not provide any explanation or other words. Background description: '{data['context_description']}'\nClaim: '{claim}'"""
                    response_text = local_models.generate_with_llm(llm, llm_tokenizer, label_prompt).strip()
                    log_entry = {"image_path": data["image_path"], "task": "label_claim", "claim": claim, "prompt": label_prompt, "response": response_text}
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    
                    lower_response = response_text.lower()
                    if "no" in lower_response or "否" in lower_response:
                        is_valid_list.append(False)
                    elif "yes" in lower_response or "是" in lower_response:
                        is_valid_list.append(True)
                    else:
                        is_valid_list.append(False)
                        print(f"Warning: Ambiguous response for claim validity: '{response_text}'. Defaulting to invalid.")
            
            final_is_valid = 1 if all(is_valid_list) else 0
            calibration_data.append({"image_path": data["image_path"], "generated_cot": data["generated_cot"], "parsed_claims": data["parsed_claims"], "is_valid": final_is_valid})
    del llm, llm_tokenizer
    local_models.clear_memory()
    print("--- LLM Unloaded ---")

    with open(config.CALIBRATION_SET_PATH, 'w', encoding='utf-8') as f:
        json.dump(calibration_data, f, ensure_ascii=False, indent=4)
    print(f"\nCalibration set built and saved to {config.CALIBRATION_SET_PATH}")
    print(f"LLM raw responses logged to {config.LLM_RAW_RESPONSE_LOG_PATH}")
    print("\n--- DATA BUILDER FINISHED SUCCESSFULLY ---")

if __name__ == "__main__":
    build_calibration_set()