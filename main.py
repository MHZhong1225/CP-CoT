# main.py
import json
import os
import numpy as np
from tqdm import tqdm
import config
from scoring import CombinedScorer
from data_builder import generate_cot_description, parse_cot_to_claims # 复用

def calculate_s_budget(claims_dict, scores_dict):
    """计算单条CoT的累积风险分数"""
    total_risk = 0
    # 计算视觉证据的风险
    for claim, score in zip(claims_dict["visual_evidence"], scores_dict["visual_evidence"]):
        total_risk += config.WEIGHTS["visual_evidence"] * score
    # 计算总结描述的风险
    for claim, score in zip(claims_dict["summary_description"], scores_dict["summary_description"]):
        total_risk += config.WEIGHTS["summary_description"] * score
    return total_risk

def run_calibration():
    """执行离线校准，计算并保存阈值"""
    print("--- Starting Calibration Phase ---")
    with open(config.CALIBRATION_SET_PATH, 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)

    scorer = CombinedScorer()
    valid_samples = [s for s in calibration_data if s['is_valid'] == 1]
    
    if not valid_samples:
        raise ValueError("No valid samples found in the calibration set. Cannot perform calibration.")

    budget_scores = []
    for sample in tqdm(valid_samples, desc="Calculating S_budget for valid samples"):
        image_path = sample['image_path']
        claims_dict = sample['parsed_claims']
        
        scores_dict = {
            "visual_evidence": [scorer.score(image_path, c) for c in claims_dict["visual_evidence"]],
            "summary_description": [scorer.score(image_path, c) for c in claims_dict["summary_description"]]
        }
        
        s_budget = calculate_s_budget(claims_dict, scores_dict)
        budget_scores.append(s_budget)
        
    # 计算分位数来确定阈值
    quantile = (1 - config.ALPHA) * 100
    q_budget = np.percentile(budget_scores, quantile)
    
    print(f"Calibration finished. Number of valid samples: {len(valid_samples)}")
    print(f"Confidence Level: {1 - config.ALPHA}, Risk Budget Threshold (q_budget): {q_budget}")
    
    with open(config.CALIBRATION_THRESHOLD_PATH, 'w') as f:
        json.dump({"q_budget": q_budget}, f)
    print(f"Threshold saved to {config.CALIBRATION_THRESHOLD_PATH}")

def run_inference(image_path: str):
    """对新图片执行在线推理"""
    print("\n--- Starting Inference Phase ---")
    print(f"Processing new image: {image_path}")
    
    # 加载已校准的阈值
    with open(config.CALIBRATION_THRESHOLD_PATH, 'r') as f:
        q_budget = json.load(f)['q_budget']
    
    scorer = CombinedScorer()
    
    # 1. 生成
    cot_text = generate_cot_description(image_path)
    claims_dict = parse_cot_to_claims(cot_text)
    
    # 2. 计算风险
    scores_dict = {
        "visual_evidence": [scorer.score(image_path, c) for c in claims_dict["visual_evidence"]],
        "summary_description": [scorer.score(image_path, c) for c in claims_dict["summary_description"]]
    }
    s_new_budget = calculate_s_budget(claims_dict, scores_dict)
    
    print(f"Calculated S_budget for new CoT: {s_new_budget}")
    print(f"Calibrated threshold Q_budget: {q_budget}")
    
    # 3. 决策
    if s_new_budget <= q_budget:
        print("\n[DECISION]: ACCEPT")
        print("This description is considered reliable with >=", (1 - config.ALPHA), "confidence.")
        print("--- Output ---")
        print(cot_text)
    else:
        print("\n[DECISION]: REJECT")
        print("This description exceeds the risk budget and is likely to contain hallucinations.")
        print("--- Output ---")
        print("抱歉，我无法生成关于此图像的、确保事实准确的详细描述。")

if __name__ == "__main__":
    # 假设校准集已通过 data_builder.py 生成
    # 1. 运行一次校准流程
    run_calibration()
    
    # 2. 对一张新图片进行推理
    new_image_example = os.path.join(config.IMAGE_DIR, os.listdir(config.IMAGE_DIR)[0]) # 以第一张图为例
    run_inference(new_image_example)