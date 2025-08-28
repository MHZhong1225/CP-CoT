# config.py

# --- Model Names (Local Hugging Face Identifiers) ---
# 警告：请确保您有足够的VRAM来加载这些模型。
# 目标模型 (MLLM) - LLaVA是一个很好的选择
# TARGET_MLLM_NAME = "Qwen/Qwen2.5-7B-Instruct" 

TARGET_MLLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# 裁判模型 (LLM)
JUDGE_LLM_NAME = "openai/gpt-oss-20b"
DATANAME = 'VQAv2'


# --- Paths ---
IMAGE_DIR = f"./data/{DATANAME}/raw_images"
CALIBRATION_SET_PATH = f"./data/{DATANAME}/calibration_set.json"
CALIBRATION_THRESHOLD_PATH = f"./data/{DATANAME}/q_budget_threshold.json"

# --- Hyperparameters ---
ALPHA = 0.1  # 错误率
SCORE_LAMBDA = 0.7  # 视觉分数和合理性分数的组合权重
GENERATION_TEMPERATURE = 0.7 # 生成时的温度参数

# 步骤权重
WEIGHTS = {
    "visual_evidence": 1.0,
    "summary_description": 1.5
}