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
import pandas as pd
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
loan_to_customer: dict[str, str] = {}
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


def search_image(image: Image.Image, filename: str, top_k: int, query_loan_id: str, force_search: bool) -> dict:
    """Shared single-image inference used by the interactive and batch APIs."""
    image_tensor = preprocess_image(image)
    category_id, category_name, scores = classifier.classify(image_tensor)
    is_sign, sign_confidence = classifier.is_sign_photo(image_tensor)
    result = {
        "filename": filename,
        "category_id": category_id,
        "category_name": category_name,
        "is_sign_photo": bool(is_sign),
        "sign_confidence": round(float(sign_confidence), 4),
        "searched": bool(is_sign or force_search),
        "all_scores": {name: round(float(score), 4) for name, score in sorted(scores.items(), key=lambda item: -item[1])},
        "similar_results": [],
    }
    if not result["searched"]:
        result["message"] = "The image is not classified as a sign-photo; similarity search was skipped."
        result["risk_summary"] = {"cross_customer_fraud": 0, "same_customer_repeat": 0, "normal_low_risk": 0}
        return result

    from src.risk_policy import assess_match, summarize_risks
    raw_results = searcher.search(encode_image(image_tensor)[0], top_k=top_k + 5)
    effective_query_loan_id = query_loan_id or auto_detect_loan_id(raw_results)
    for raw_rank, item in enumerate(raw_results, start=1):
        risk = assess_match(float(item["score"]), effective_query_loan_id, item["metadata"], loan_to_customer, risk_policy)
        if risk["relation"] == "self":
            continue
        metadata = item["metadata"]
        result["similar_results"].append({
            "rank": len(result["similar_results"]) + 1,
            "raw_rank": raw_rank,
            "similarity": round(float(item["score"]), 4),
            "threshold_applied": round(float(risk["threshold_used"]), 4),
            "relationship": risk["relation"], "relationship_label": risk["relation_label"],
            "is_suspicious": risk["is_suspicious"], "fraud_type": risk["risk_type"],
            "risk_level": risk["risk_level"], "review_priority": risk["review_priority"],
            "recommended_action": risk["recommended_action"],
            "loan_id": metadata.get("loan_id", metadata.get("biz_id", "")),
            "business_type": metadata.get("business_type", ""), "similar_group": metadata.get("similar_group", ""),
            "path": metadata.get("path", ""),
        })
        if len(result["similar_results"]) >= top_k:
            break
    result["query_loan_id"] = effective_query_loan_id
    result["risk_summary"] = summarize_risks(result["similar_results"])
    return result


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
    loan_to_customer.clear()
    for item in searcher.metadata:
        loan_id = str(item.get("loan_id", "") or item.get("biz_id", ""))
        customer_id = str(item.get("customer_id", item.get("customer_no", "")) or "")
        if loan_id and customer_id:
            loan_to_customer[loan_id] = customer_id


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
        "known_customer_ids": len(set(loan_to_customer.values())),
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
    review_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict:
    lazy_init()
    from src.risk_policy import assess_match, summarize_risks

    image = read_image(file, await file.read())
    requested_top_k = top_k or int(config["app"].get("top_k", 5))
    effective_review_threshold = (
        float(review_threshold)
        if review_threshold is not None
        else float(config["retrieval"].get("high_risk_threshold", config["retrieval"]["similarity_threshold"]))
    )
    result = search_image(image, file.filename or "upload", requested_top_k, query_loan_id, force_search)
    for item in result["similar_results"]:
        item["review_threshold"] = round(effective_review_threshold, 4)
        item["selected_by_review_threshold"] = bool(item["similarity"] >= effective_review_threshold)

    dynamic = config["retrieval"].get("dynamic_threshold", {})
    return {
        **result,
        "model": config["model"]["name"],
        "dynamic_threshold": {
            "enabled": bool(dynamic.get("enabled", False)),
            "cross_customer_threshold": risk_policy.cross_customer,
            "same_customer_threshold": risk_policy.same_customer,
            "default_threshold": risk_policy.default,
            "high_risk_threshold": risk_policy.high_risk,
            "medium_risk_threshold": risk_policy.medium_risk,
        },
        "review_threshold": effective_review_threshold,
    }


@app.post("/batch-search")
async def batch_search(
    files: list[UploadFile] = File(...),
    top_k: int = Query(default=5, ge=1, le=50),
    force_search: bool = Query(default=False),
) -> dict:
    """Classify and search a batch; callers may persist the returned rows as CSV."""
    lazy_init()
    records = []
    for file in files:
        image = read_image(file, await file.read())
        records.append(search_image(image, file.filename or "upload", top_k, "", force_search))
    suspicious = sum(sum(1 for match in item["similar_results"] if match["is_suspicious"]) for item in records)
    return {"total_files": len(records), "searched_files": sum(item["searched"] for item in records), "suspicious_matches": suspicious, "results": records}


@app.get("/monitoring-report")
async def monitoring_report(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    """Expose the latest offline graph-monitoring result to a dashboard client."""
    report = ROOT / "outputs" / "mvp" / "fraud_monitoring.csv"
    summary = ROOT / "outputs" / "mvp" / "fraud_monitoring_summary.json"
    if not report.exists():
        raise HTTPException(status_code=404, detail="Run `python -m mvp.pipeline` to generate the monitoring report.")
    rows = pd.read_csv(report).head(limit).replace({np.nan: None}).to_dict(orient="records")
    import json
    return {"summary": json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the financial image similarity API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
