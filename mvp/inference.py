from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from mvp.pipeline import DEFAULT_MODEL, LinearHead, SiglipEncoder, choose_device

CATEGORY_LABELS = {
    "bank_statement": "银行流水",
    "contract": "合同",
    "face_signing": "面签照片",
    "id_card_back": "身份证背面",
    "id_card_front": "身份证正面",
}

RISK_LABELS = {
    "high": "高风险",
    "medium": "中风险",
    "low": "低风险",
}

FIELD_LABELS = {
    "loan_id": "贷款编号",
    "query_loan_id": "查询贷款编号",
    "match_loan_id": "匹配贷款编号",
    "image_type": "原始影像类别",
    "predicted_type": "预测影像类别",
    "predicted_type_label": "预测影像类别（中文）",
    "confidence": "分类置信度",
    "cosine_similarity": "余弦相似度",
    "risk_level": "风险等级",
    "risk_level_label": "风险等级（中文）",
    "rank": "相似度排名",
    "query_path": "查询图片路径",
    "match_path": "匹配图片路径",
    "relative_path": "相对路径",
    "business_type": "业务类型",
}


class SimilarityRuntime:
    def __init__(self, output_dir: Path, device: str = "auto"):
        self.output_dir = output_dir
        with (output_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
            self.summary = json.load(handle)

        self.device = choose_device(device)
        self.encoder = SiglipEncoder(self.summary.get("model_name", DEFAULT_MODEL), self.device)

        checkpoint = torch.load(output_dir / "classifier.pt", map_location="cpu", weights_only=False)
        self.classes = list(checkpoint["classes"])
        self.classifier = LinearHead(int(checkpoint["input_dim"]), len(self.classes))
        self.classifier.load_state_dict(checkpoint["state_dict"])
        self.classifier.eval()

        self.face_embeddings = np.load(output_dir / "face_embeddings.npy").astype("float32")
        self.face_embeddings = normalize_rows(self.face_embeddings)
        self.face_manifest = pd.read_csv(output_dir / "face_manifest.csv")

    @property
    def high_threshold(self) -> float:
        return float(self.summary["high_risk_threshold"])

    @property
    def medium_threshold(self) -> float:
        return float(self.summary["medium_risk_threshold"])


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def risk_level(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


@lru_cache(maxsize=2)
def get_runtime(output_dir: str, device: str = "auto") -> SimilarityRuntime:
    return SimilarityRuntime(Path(output_dir), device)


def prepare_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def analyze_image(
    image: Image.Image,
    runtime: SimilarityRuntime,
    top_k: int = 5,
    force_search: bool = False,
) -> dict:
    image = prepare_image(image)
    embedding = runtime.encoder.encode([image], batch_size=1)
    embedding = normalize_rows(embedding)

    with torch.no_grad():
        logits = runtime.classifier(torch.from_numpy(embedding))
        probabilities = logits.softmax(dim=1).numpy()[0]

    best_index = int(np.argmax(probabilities))
    predicted_type = runtime.classes[best_index]
    is_face_signing = predicted_type == "face_signing"

    result = {
        "predicted_type": predicted_type,
        "predicted_type_label": CATEGORY_LABELS.get(predicted_type, predicted_type),
        "confidence": float(probabilities[best_index]),
        "class_scores": {
            class_name: {
                "label": CATEGORY_LABELS.get(class_name, class_name),
                "score": float(probabilities[index]),
            }
            for index, class_name in enumerate(runtime.classes)
        },
        "searched": bool(is_face_signing or force_search),
        "matches": [],
    }

    if not result["searched"]:
        result["message"] = "上传图片未被分类为面签照片，默认不进入相似度检索。"
        return result

    scores = runtime.face_embeddings @ embedding[0]
    indices = np.argsort(scores)[::-1][:top_k]
    matches = []
    for rank, index in enumerate(indices, start=1):
        score = float(scores[index])
        metadata = runtime.face_manifest.iloc[int(index)].to_dict()
        level = risk_level(score, runtime.high_threshold, runtime.medium_threshold)
        matches.append(
            {
                "rank": rank,
                "match_loan_id": metadata.get("loan_id"),
                "match_path": metadata.get("path"),
                "relative_path": metadata.get("relative_path"),
                "cosine_similarity": score,
                "risk_level": level,
                "risk_level_label": RISK_LABELS[level],
            }
        )
    result["matches"] = matches
    return result


def with_chinese_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: FIELD_LABELS.get(column, column) for column in frame.columns}
    return frame.rename(columns=rename_map)
