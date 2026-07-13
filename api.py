from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image


ROOT = Path(__file__).resolve().parent

extractor = None
searcher = None
classifier = None
preprocessor = None
config = None
loan_to_group: dict[str, str] = {}
risk_policy = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    lazy_init()
    yield


app = FastAPI(
    title="Financial Image Similarity API",
    description="SigLIP2-based financial image classification, retrieval, and stratified risk scoring.",
    version="1.1.0",
    lifespan=lifespan,
)


def load_config() -> dict:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def lazy_init() -> None:
    global extractor, searcher, classifier, preprocessor, config, risk_policy

    if extractor is not None:
        return

    from src.classifier import ImageClassifier
    from src.model import CLIPFeatureExtractor
    from src.preprocessing import PreprocessingPipeline
    from src.retrieval import SimilaritySearch
    from src.risk_policy import ThresholdPolicy

    config = load_config()
    model_name = config["model"]["name"]
    extractor = CLIPFeatureExtractor(model_name=model_name)
    searcher = SimilaritySearch(
        embedding_dim=config["model"]["embedding_dim"],
        index_path=config["retrieval"].get("index_path", "checkpoints/faiss_index.bin"),
        index_type=config["retrieval"].get("index_type", "flat"),
        nlist=config["retrieval"].get("nlist", 100),
    )
    classifier = ImageClassifier(
        model=extractor.model,
        processor=extractor.processor,
        categories=config["classifier"]["categories"],
        device=extractor.device,
    )
    preprocessor = PreprocessingPipeline(config.get("preprocessing", {}))
    risk_policy = ThresholdPolicy.from_config(config)

    searcher.load()
    loan_to_group.clear()
    for item in searcher.metadata:
        loan_id = str(item.get("loan_id", "") or item.get("biz_id", ""))
        similar_group = str(item.get("similar_group", "") or "")
        if loan_id and similar_group:
            loan_to_group[loan_id] = similar_group


def read_image(file: UploadFile, content: bytes) -> Image.Image:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse image: {exc}") from exc


def preprocess_image(image: Image.Image) -> torch.Tensor:
    processed = preprocessor(image).convert("RGB")
    return extractor.preprocess(processed)


def auto_detect_loan_id(results: list[dict]) -> str:
    for item in results:
        if float(item["score"]) >= 0.999:
            return str(item["metadata"].get("loan_id", "") or item["metadata"].get("biz_id", ""))
    return ""


def encode_image(image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        features = extractor.extract(image_tensor)
    values = features.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


@app.get("/health")
async def health() -> dict:
    lazy_init()
    return {
        "status": "ok",
        "model": config["model"]["name"],
        "baseline_model": config["model"].get("baseline_name"),
        "embedding_dim": config["model"]["embedding_dim"],
        "index_size": int(searcher.index.ntotal) if searcher and searcher.index else 0,
        "risk_policy": getattr(risk_policy, "__class__", type("", (), {})).__name__,
    }


@app.get("/stats")
async def stats() -> dict:
    lazy_init()
    from collections import Counter

    category_counts = Counter(item.get("cat_name", "unknown") for item in searcher.metadata)
    return {
        "total": int(searcher.index.ntotal) if searcher and searcher.index else 0,
        "index_type": searcher.index_type,
        "embedding_dim": searcher.embedding_dim,
        "category_distribution": dict(category_counts.most_common()),
        "known_business_groups": len(set(loan_to_group.values())),
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict:
    lazy_init()
    image = read_image(file, await file.read())
    image_tensor = preprocess_image(image)
    category_id, category_name, scores = classifier.classify(image_tensor)
    is_sign, sign_confidence = classifier.is_sign_photo(image_tensor)
    return {
        "filename": file.filename,
        "category_id": category_id,
        "category_name": category_name,
        "is_sign_photo": bool(is_sign),
        "sign_confidence": round(float(sign_confidence), 4),
        "all_scores": {
            name: round(float(score), 4)
            for name, score in sorted(scores.items(), key=lambda item: -item[1])
        },
    }


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    top_k: int | None = Query(default=None, ge=1, le=50),
    query_loan_id: str = Query(default=""),
    force_search: bool = Query(default=False),
) -> dict:
    lazy_init()
    from src.risk_policy import assess_match, summarize_risks

    image = read_image(file, await file.read())
    image_tensor = preprocess_image(image)
    category_id, category_name, scores = classifier.classify(image_tensor)
    is_sign, sign_confidence = classifier.is_sign_photo(image_tensor)

    if not is_sign and not force_search:
        return {
            "filename": file.filename,
            "category_id": category_id,
            "category_name": category_name,
            "is_sign_photo": False,
            "sign_confidence": round(float(sign_confidence), 4),
            "searched": False,
            "message": "The image is not classified as a sign-photo; similarity search was skipped.",
            "all_scores": {
                name: round(float(score), 4)
                for name, score in sorted(scores.items(), key=lambda item: -item[1])
            },
            "similar_results": [],
            "risk_summary": summarize_risks([]),
        }

    requested_top_k = top_k or int(config["app"].get("top_k", 5))
    raw_results = searcher.search(encode_image(image_tensor)[0], top_k=requested_top_k + 5)
    if not query_loan_id:
        query_loan_id = auto_detect_loan_id(raw_results)

    similar_results = []
    for rank, item in enumerate(raw_results, start=1):
        risk = assess_match(
            score=float(item["score"]),
            query_loan_id=query_loan_id,
            metadata=item["metadata"],
            loan_to_group=loan_to_group,
            policy=risk_policy,
        )
        if risk["relation"] == "self":
            continue

        metadata = item["metadata"]
        similar_results.append(
            {
                "rank": len(similar_results) + 1,
                "raw_rank": rank,
                "similarity": round(float(item["score"]), 4),
                "threshold_applied": round(float(risk["threshold_used"]), 4),
                "relationship": risk["relation"],
                "relationship_label": risk["relation_label"],
                "is_suspicious": risk["is_suspicious"],
                "risk_level": risk["risk_level"],
                "risk_type": risk["risk_type"],
                "risk_type_label": risk["risk_type_label"],
                "review_priority": risk["review_priority"],
                "recommended_action": risk["recommended_action"],
                "policy_version": risk["policy_version"],
                "loan_id": metadata.get("loan_id", metadata.get("biz_id", "")),
                "business_type": metadata.get("business_type", ""),
                "similar_group": metadata.get("similar_group", ""),
                "category": metadata.get("cat_name", ""),
                "path": metadata.get("path", ""),
            }
        )
        if len(similar_results) >= requested_top_k:
            break

    dynamic = config["retrieval"].get("dynamic_threshold", {})
    return {
        "filename": file.filename,
        "model": config["model"]["name"],
        "category_id": category_id,
        "category_name": category_name,
        "is_sign_photo": bool(is_sign),
        "sign_confidence": round(float(sign_confidence), 4),
        "searched": True,
        "query_loan_id": query_loan_id,
        "all_scores": {
            name: round(float(score), 4)
            for name, score in sorted(scores.items(), key=lambda item: -item[1])
        },
        "dynamic_threshold": {
            "enabled": bool(dynamic.get("enabled", False)),
            "cross_customer_threshold": risk_policy.cross_customer,
            "same_customer_threshold": risk_policy.same_customer,
            "default_threshold": risk_policy.default,
        },
        "risk_summary": summarize_risks(similar_results),
        "similar_results": similar_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the financial image similarity API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
