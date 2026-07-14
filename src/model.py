import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_NAME = "google/siglip2-base-patch16-224"


def choose_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VisionLanguageFeatureExtractor:
    """Configurable image/text encoder for SigLIP2 or CLIP style models."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "auto"):
        self.model_name = model_name
        self.device = choose_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype="auto").to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def extract(self, images):
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=images)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            return F.normalize(features.float(), dim=-1)


class CLIPFeatureExtractor(VisionLanguageFeatureExtractor):
    """Backward-compatible name; the configured model can be SigLIP2 or CLIP."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "auto"):
        super().__init__(model_name=model_name, device=device)


if __name__ == "__main__":
    extractor = CLIPFeatureExtractor()
    dummy = torch.randn(2, 3, 224, 224)
    out = extractor.extract(dummy)
    print(f"Feature extraction OK: {out.shape}")
