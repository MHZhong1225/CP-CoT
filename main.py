# main.py (科学严谨最终版)
import json
import os
import numpy as np
from tqdm import tqdm
import config
from scoring import CombinedScorer
import local_models 
from data_builder import python_parse_cot_to_claims

# ... (calculate_s_budget 和 run_calibration 函数保持不变) ...
def calculate_s_budget(claims_dict, scores_dict):
    total_risk = 0
    visual_evidence_claims = claims_dict.get("visual_evidence", [])
    summary_description_claims = claims_dict.get("summary_description", [])
    for _, score in zip(visual_evidence_claims, scores_dict["visual_evidence"]):
        total_risk += config.WEIGHTS["visual_evidence"] * score
    for _, score in zip(summary_description_claims, scores_dict["summary_description"]):
        total_risk += config.WEIGHTS["summary_description"] * score
    return total_risk

def run_calibration():
    print("--- Starting Calibration Phase ---")
    with open(config.CALIBRATION_SET_PATH, 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    scorer = CombinedScorer()
    valid_samples = [s for s in calibration_data if s['is_valid'] == 1]
    if not valid_samples:
        raise ValueError("No valid samples found in the calibration set.")
    llm, llm_tokenizer = local_models.load_llm()
    budget_scores = []
    for sample in tqdm(valid_samples, desc="Calculating S_budget for valid samples"):
        image_path = sample['image_path']
        claims_dict = sample['parsed_claims']
        visual_evidence_claims = claims_dict.get("visual_evidence", [])
        summary_description_claims = claims_dict.get("summary_description", [])
        scores_dict = {
            "visual_evidence": [scorer.score(image_path, c, llm, llm_tokenizer) for c in visual_evidence_claims],
            "summary_description": [scorer.score(image_path, c, llm, llm_tokenizer) for c in summary_description_claims]
        }
        s_budget = calculate_s_budget(claims_dict, scores_dict)
        budget_scores.append(s_budget)
    del llm, llm_tokenizer
    local_models.clear_memory()
    quantile = (1 - config.ALPHA) * 100
    q_budget = np.percentile(budget_scores, quantile) if budget_scores else 0
    print(f"Calibration finished. Number of valid samples: {len(valid_samples)}")
    print(f"Confidence Level: {1 - config.ALPHA}, Risk Budget Threshold (q_budget): {q_budget:.4f}")
    with open(config.CALIBRATION_THRESHOLD_PATH, 'w') as f:
        json.dump({"q_budget": q_budget}, f)
    print(f"Threshold saved to {config.CALIBRATION_THRESHOLD_PATH}")
    return q_budget

def test_single_description(image_path: str, cot_text: str, q_budget: float, scorer, model, tokenizer):
    """测试单个CoT描述，计算风险并作出决策"""
    print("\n" + "="*80)
    print(f"Testing Description:\n{cot_text}")
    print("="*80)
    claims_dict = python_parse_cot_to_claims(cot_text)
    if not claims_dict.get("visual_evidence") and not claims_dict.get("summary_description"):
        print("\n[ERROR]: Failed to parse the description into claims. Skipping.")
        return
    visual_claims = claims_dict.get("visual_evidence", [])
    summary_claims = claims_dict.get("summary_description", [])
    scores_dict = {
        "visual_evidence": [scorer.score(image_path, c, model, tokenizer) for c in visual_claims],
        "summary_description": [scorer.score(image_path, c, model, tokenizer) for c in summary_claims]
    }
    s_new_budget = calculate_s_budget(claims_dict, scores_dict)
    print(f"\nCalculated Risk Score (S_budget): {s_new_budget:.4f}")
    print(f"Calibrated Threshold (Q_budget): {q_budget:.4f}")
    if s_new_budget <= q_budget:
        print("\n[DECISION]: ACCEPT ✔️")
        print(f"This description's risk is within the calibrated budget for >={1 - config.ALPHA} confidence.")
    else:
        print("\n[DECISION]: REJECT ❌")
        print("This description's risk EXCEEDS the budget and is likely to contain hallucinations.")
    print("="*80)

def run_demonstration(image_path: str, q_budget: float):
    """运行一个科学严谨的演示，对比三种与图片强相关的描述。"""
    print("\n\n--- Starting Rigorous CP Effect Demonstration ---")
    print(f"Processing image: {image_path}")

    # --- Step 1: Generation Phase (MLLM only) ---
    print("\n--- Phase 1: Generating original description with Target MLLM ---")
    mllm, mllm_processor = local_models.load_mllm()
    cot_prompt = "Please describe this image in detail. Use a step-by-step reasoning approach: first, list the key visual evidence you observe (at least 3 points), then provide a summary description based on this evidence."
    model_generated_cot = local_models.generate_with_mllm(mllm, mllm_processor, image_path, cot_prompt)
    del mllm, mllm_processor
    local_models.clear_memory()
    print("--- Target MLLM Unloaded ---")
    
    # --- Step 2: Create meaningful "Good" and "Bad" descriptions based on the generated one ---
    
    # The "Good" description is the model's own output, but with the specific hallucination manually removed.
    # This represents the ideal, factually correct version.
    good_cot = model_generated_cot.replace("The foal is nursing from the adult horse, indicating a nurturing moment.", "The foal is standing near the adult horse.").strip()

    # The "Bad" description is a handcrafted, but RELEVANT, hallucination.
    bad_cot = """
        ### Step-by-Step Reasoning Approach:
        1. **Key Visual Evidence:**
        - **Animals:** There are two horses in what appears to be a river.
        - **Color:** The larger horse is brown with a white mane, and the smaller horse is a lighter brown.
        - **Environment:** The horses are surrounded by water and splashing happily.
        2. **Summary Description:**
        - The image shows a majestic adult horse and its foal taking a refreshing bath in a river on a sunny day. The water is sparkling around them.
    """

    # --- Step 3: Scoring Phase (Scorer and Judge LLM) ---
    print("\n--- Phase 2: Scoring all three relevant descriptions ---")
    scorer = CombinedScorer()
    llm, ll_tokenizer = local_models.load_llm()

    # Case 1: The model's original output, which contains a subtle hallucination.
    print("\n--- Testing Case 1: Model's Original Output (with subtle hallucination) ---")
    test_single_description(image_path, model_generated_cot, q_budget, scorer, llm, ll_tokenizer)

    # Case 2: The "Corrected" version of the model's output, factually accurate.
    print("\n--- Testing Case 2: Manually Corrected 'Good' Description ---")
    test_single_description(image_path, good_cot, q_budget, scorer, llm, ll_tokenizer)
    
    # Case 3: The handcrafted, but relevant, "Bad" description.
    print("\n--- Testing Case 3: Manually Crafted 'Bad' Description (relevant hallucination) ---")
    test_single_description(image_path, bad_cot, q_budget, scorer, llm, ll_tokenizer)
    
    del llm, ll_tokenizer
    local_models.clear_memory()
    print("--- Scorer and Judge LLM Unloaded ---")


if __name__ == "__main__":
    # 运行校准，并使用新图片进行科学严谨的演示
    q_budget_threshold = run_calibration()
    demo_image = './caldata/VQAv2/raw_images/vqa_image_305.jpg' # 直接使用你上传的图片进行测试
    if os.path.exists(demo_image):
        run_demonstration(demo_image, q_budget_threshold)
    else:
        print(f"\nError: Demo image '{demo_image}' not found. Please make sure it's in the same directory.")