# scoring.py
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import config

class BaseScorer:
    def __init__(self):
        pass
    def score(self, image_path, claim_text):
        raise NotImplementedError

class CLIPScorer(BaseScorer):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP model loaded.")

    def score(self, image_path, claim_text):
        image = Image.open(image_path)
        inputs = self.processor(text=[claim_text], images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.item() / 100.0 # 归一化
        # 返回距离作为不一致性分数
        return 1.0 - similarity

# scoring.py (部分修改)
import local_models # 导入新模块

class PlausibilityScorer(BaseScorer):
    def score(self, image_path, claim_text):
        prompt = f"请仅根据常识，为以下陈述的不合理程度打分（1-10分，分数越高越不合理）：‘{claim_text}’。请只返回数字。"
        response_text = local_models.generate_with_llm(prompt)
        # ... 此处需要添加代码来安全地解析返回的数字 ...
        try:
            score_val = float(response_text.strip())
        except ValueError:
            score_val = 5.0 # 如果解析失败，给一个中间值
        return score_val / 10.0 # 归一化

class CombinedScorer(BaseScorer):
    def __init__(self):
        self.clip_scorer = CLIPScorer()
        self.plausibility_scorer = PlausibilityScorer()
        self.lambda_val = config.SCORE_LAMBDA

    def score(self, image_path, claim_text):
        s_visual = self.clip_scorer.score(image_path, claim_text)
        s_plausibility = self.plausibility_scorer.score(image_path, claim_text)
        
        combined_score = self.lambda_val * s_visual + (1 - self.lambda_val) * s_plausibility
        return combined_score