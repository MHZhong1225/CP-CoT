# scoring.py (Final Robust Version with Truncation)
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import config
import local_models

class BaseScorer:
    def __init__(self):
        pass
    def score(self, image_path, claim_text, model=None, tokenizer=None):
        raise NotImplementedError

class CLIPScorer(BaseScorer):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # --- FIX: Store the max sequence length for truncation ---
        self.max_length = self.model.config.text_config.max_position_embeddings
        print("CLIP model loaded.")

    def score(self, image_path, claim_text, model=None, tokenizer=None):
        image = Image.open(image_path)
        # --- FIX: Truncate the text input to prevent overflow ---
        inputs = self.processor(
            text=[claim_text], 
            images=image, 
            return_tensors="pt", 
            padding="max_length", # Use max_length padding
            truncation=True,    # Enable truncation
            max_length=self.max_length
        ).to(self.device)
        
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.item() / 100.0
        return 1.0 - similarity

class PlausibilityScorer(BaseScorer):
    def score(self, image_path, claim_text, model, tokenizer):
        prompt = f"Please rate the implausibility of the following statement based on common sense (1-10, where higher is more implausible): '{claim_text}'. Please return only the number."
        response_text = local_models.generate_with_llm(model, tokenizer, prompt)
        try:
            score_val = float(response_text.strip())
        except ValueError:
            score_val = 5.0
        return score_val / 10.0

class CombinedScorer(BaseScorer):
    def __init__(self):
        self.clip_scorer = CLIPScorer()
        self.plausibility_scorer = PlausibilityScorer()
        self.lambda_val = config.SCORE_LAMBDA

    def score(self, image_path, claim_text, model, tokenizer):
        s_visual = self.clip_scorer.score(image_path, claim_text)
        s_plausibility = self.plausibility_scorer.score(image_path, claim_text, model, tokenizer)
        
        combined_score = self.lambda_val * s_visual + (1 - self.lambda_val) * s_plausibility
        return combined_score
        