from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "mvp"

app = FastAPI(
    title="Financial Image Similarity MVP API",
    version="1.0.0",
    description="Read-only API for results produced by mvp.pipeline.",
)


def require_output_file(name: str) -> Path:
    path = OUTPUT / name
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{name} not found. Run: python -m mvp.pipeline",
        )
    return path


def read_json(name: str) -> dict:
    with require_output_file(name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(require_output_file(name))


def frame_records(frame: pd.DataFrame) -> list[dict]:
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    ready = (OUTPUT / "run_summary.json").exists()
    return {"status": "ok" if ready else "missing_outputs", "output_dir": str(OUTPUT)}


@app.get("/summary")
def summary() -> dict:
    return read_json("run_summary.json")


@app.get("/metrics")
def metrics() -> dict:
    return read_json("classification_metrics.json")


@app.get("/thresholds")
def thresholds() -> dict:
    return {
        "metadata": read_json("threshold_metadata.json"),
        "rows": frame_records(read_csv("threshold_experiment.csv")),
    }


@app.get("/calibration")
def calibration() -> dict:
    summary_data = read_json("run_summary.json")
    labels_path = OUTPUT / "review_labels.csv"
    labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
    positives = int(labels["is_similar"].sum()) if not labels.empty else 0
    reviewed = int(len(labels))
    return {
        "high_risk_threshold": summary_data["high_risk_threshold"],
        "medium_risk_threshold": summary_data["medium_risk_threshold"],
        "threshold_basis": "manual_review_calibration",
        "calibration_note": (
            ">=0.97 candidates were confirmed similar in manual review; "
            "0.95~0.97 started to show unstable candidates."
        ),
        "reviewed_pairs": reviewed,
        "confirmed_similar_pairs": positives,
        "rejected_pairs": reviewed - positives,
        "labels_file": str(labels_path),
    }


@app.get("/predictions")
def predictions(
    image_type: str | None = None,
    predicted_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    frame = read_csv("classification_predictions.csv")
    if image_type:
        frame = frame[frame["image_type"] == image_type]
    if predicted_type:
        frame = frame[frame["predicted_type"] == predicted_type]
    return {"count": int(len(frame)), "rows": frame_records(frame.head(limit))}


@app.get("/matches/{loan_id}")
def matches(loan_id: str, threshold: float | None = None) -> dict:
    frame = read_csv("topk_results.csv")
    rows = frame[frame["query_loan_id"] == loan_id].copy()
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No matches found for {loan_id}")
    if threshold is not None:
        rows["selected_by_threshold"] = rows["cosine_similarity"] >= threshold
    return {"loan_id": loan_id, "count": int(len(rows)), "rows": frame_records(rows)}


@app.get("/risks")
def risks(
    min_score: float | None = None,
    risk_level: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    frame = read_csv("topk_results.csv")
    if min_score is not None:
        frame = frame[frame["cosine_similarity"] >= min_score]
    if risk_level:
        frame = frame[frame["risk_level"] == risk_level]
    frame = frame.sort_values("cosine_similarity", ascending=False)
    return {"count": int(len(frame)), "rows": frame_records(frame.head(limit))}
