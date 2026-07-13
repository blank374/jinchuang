import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model.eval()

    def extract(self, images):
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=images)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features


if __name__ == "__main__":
    extractor = CLIPFeatureExtractor()
    dummy = torch.randn(2, 3, 224, 224)
    out = extractor.extract(dummy)
    print(f"Feature extraction OK: {out.shape}")
