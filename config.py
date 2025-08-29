# config.py

# --- Model Names (Local Hugging Face Identifiers) ---
# 警告：请确保您有足够的VRAM来加载这些模型。
# 目标模型 (MLLM) - LLaVA是一个很好的选择
# TARGET_MLLM_NAME = "Qwen/Qwen2.5-7B-Instruct" 

TARGET_MLLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# 裁判模型 (LLM)
# JUDGE_LLM_NAME = "openai/gpt-oss-20b"
JUDGE_LLM_NAME = "Qwen/Qwen3-8B-FP8"
DATANAME = 'VQAv2'

LLM_RAW_RESPONSE_LOG_PATH = f"./caldata/{DATANAME}/llm_raw_responses.jsonl" # 使用 .jsonl 格式

# --- Paths ---
IMAGE_DIR = f"./caldata/{DATANAME}/raw_images"
CALIBRATION_SET_PATH = f"./caldata/{DATANAME}/calibration_set.json"
CALIBRATION_THRESHOLD_PATH = f"./caldata/{DATANAME}/q_budget_threshold.json"
GENERATED_DATA_PATH = f"./caldata/{DATANAME}/generated_data.json" 
COCO_CAPTIONS_PATH = f"./caldata/VQAv2/annotations/captions_val2014.json"


# --- Hyperparameters ---
# 1. 提高 ALPHA，让阈值更严格
# 将 ALPHA 从 0.1 改为 0.25。这意味着我们将基于校准集中分数最低的75%来设定阈值，
# 而不是之前的90%。这会大大降低阈值，提高系统的“警惕性”。
ALPHA = 0.1  # 原来是 0.1

SCORE_LAMBDA = 1.0  # 原来是 0.7
GENERATION_TEMPERATURE = 0.7 # 生成时的温度参数



# 步骤权重
WEIGHTS = {
    "visual_evidence": 1.0,
    "summary_description": 1.5
}