"""Build a two-stage similarity detection and fraud typing pipeline.

Stage 1 detects whether a pair of face-signing photos belongs to the same
similar_group using image-only pair evidence.

Stage 2 explains risk type only for pairs predicted similar by Stage 1, using
business identity evidence such as customer hash, name match, same_iddd, and
edit_type. similar_group is used only as an offline label.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IMAGE_FEATURE_COLUMNS = [
    "global_semantic_similarity",
    "subject_region_hist_similarity",
    "background_hist_similarity",
    "local_structure_orb_ratio",
    "dhash_similarity",
    "mirror_subject_region_hist_similarity",
    "mirror_background_hist_similarity",
    "mirror_dhash_similarity",
    "equalized_dhash_similarity",
    "edge_dhash_similarity",
    "edge_hist_similarity",
    "rotated_dhash_similarity",
    "rotated_edge_dhash_similarity",
    "brightness_delta",
    "contrast_delta",
    "blur_ratio",
]


RENEWAL_EDIT_TYPES = {"bg", "hair", "shirt", "shirt_bg", "background", "clothes", "background_change", "hair_change", "clothes_change"}
MANIPULATION_EDIT_TYPES = {"brightness", "contrast", "rotate", "rotation", "crop", "mirror", "flip"}


def normalize_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def dataset_loan_id_from_path(frame: pd.DataFrame) -> pd.Series:
    return frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


def choose_name_column(frame: pd.DataFrame) -> str:
    if "姓名" in frame.columns:
        return "姓名"
    if "base_from" in frame.columns:
        index = list(frame.columns).index("base_from")
        if index + 1 < len(frame.columns):
            return frame.columns[index + 1]
    raise ValueError("Could not infer name column")


def load_metadata(annotations_path: Path, output_dir: Path) -> pd.DataFrame:
    annotations = pd.read_csv(annotations_path, dtype=str, encoding="utf-8-sig").fillna("")
    name_column = choose_name_column(annotations)
    annotations = annotations.assign(dataset_loan_id=dataset_loan_id_from_path(annotations))
    columns = ["dataset_loan_id", "file_path", "loan_id", "similar_group", "is_similar_pair", "edit_type", "base_from", "same_iddd", name_column]
    metadata = (
        annotations[columns]
        .rename(columns={"loan_id": "business_loan_id", name_column: "name"})
        .drop_duplicates("dataset_loan_id")
    )
    metadata["name_norm"] = metadata["name"].map(normalize_name)
    metadata["loan_group_key"] = metadata["similar_group"].where(metadata["similar_group"].astype(str).ne(""), metadata["dataset_loan_id"])
    metadata["base_from_loan_id"] = (
        metadata["base_from"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    )
    identity_path = output_dir / "customer_identity_map_from_annotations.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path, dtype=str).fillna("")
        metadata = metadata.merge(
            identity[["dataset_loan_id", "customer_id_hash", "status"]],
            on="dataset_loan_id",
            how="left",
        )
    else:
        metadata["customer_id_hash"] = ""
        metadata["status"] = ""
    return metadata


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                IMAGE_FEATURE_COLUMNS,
            )
        ]
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        l2_regularization=0.05,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def best_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    y = np.asarray(y_true).astype(bool)
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = probabilities >= threshold
        tp = int((prediction & y).sum())
        fp = int((prediction & ~y).sum())
        fn = int((~prediction & y).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best


def metrics_at_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true).astype(int)
    prediction = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) > 1 else 0.0,
        "positive_predictions": int(prediction.sum()),
    }


def make_group_split(frame: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    all_groups = np.array(sorted(set(frame["query_loan_group_key"]) | set(frame["match_loan_group_key"])))
    rng.shuffle(all_groups)
    test_count = max(1, int(round(len(all_groups) * test_size)))
    test_groups = set(all_groups[:test_count])
    query_is_test = frame["query_loan_group_key"].isin(test_groups)
    match_is_test = frame["match_loan_group_key"].isin(test_groups)
    train = frame[~query_is_test & ~match_is_test].copy()
    test = frame[query_is_test & match_is_test].copy()
    dropped = frame[query_is_test ^ match_is_test].copy()
    return train, test, dropped


def explain_stage2(row: pd.Series) -> str:
    if not bool(row["stage1_predicted_similar"]):
        return "not_suspicious"
    if bool(row["renewal_base_pair"]):
        return "normal_renewal_similarity"
    if bool(row["id_conflict"]) and bool(row["name_match"]):
        return "same_name_cross_id_fraud"
    if bool(row["id_conflict"]):
        return "cross_customer_fraud"
    if bool(row["same_iddd_pair"]) or bool(row["id_match"]):
        return "same_customer_repeat_review"
    if bool(row["name_match"]):
        return "same_name_pending_identity"
    return "high_similarity_pending_identity"


def table_stage2_type(row: pd.Series) -> str:
    if bool(row["renewal_base_pair"]):
        return "normal_renewal_similarity"
    if bool(row["same_similar_group"]):
        if bool(row["same_iddd_pair"]):
            return "same_customer_repeat_review"
        return "cross_customer_fraud"
    return "not_labeled_similar"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-stage similarity and fraud type reports.")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--pair-report", default="outputs/mvp/pair_evidence_model_report.csv")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pairs = pd.read_csv(args.pair_report)
    metadata = load_metadata(Path(args.annotations), output_dir)
    left = metadata.add_prefix("query_")
    right = metadata.add_prefix("match_")
    pairs = pairs.merge(left, left_on="query_loan_id", right_on="query_dataset_loan_id", how="left")
    pairs = pairs.merge(right, left_on="match_loan_id", right_on="match_dataset_loan_id", how="left")
    pairs["query_edit_type_norm"] = pairs["query_edit_type"].fillna("").astype(str).str.lower()
    pairs["match_edit_type_norm"] = pairs["match_edit_type"].fillna("").astype(str).str.lower()
    pairs["renewal_base_pair"] = (
        pairs["query_edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
        & pairs["query_base_from_loan_id"].fillna("").astype(str).eq(pairs["match_loan_id"].astype(str))
    ) | (
        pairs["match_edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
        & pairs["match_base_from_loan_id"].fillna("").astype(str).eq(pairs["query_loan_id"].astype(str))
    )
    pairs["stage1_label"] = (pairs["same_similar_group"].astype(bool) | pairs["renewal_base_pair"]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        pairs[IMAGE_FEATURE_COLUMNS],
        pairs["stage1_label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=pairs["stage1_label"],
    )
    pair_model = build_model()
    pair_model.fit(X_train, y_train)
    pair_test_probabilities = pair_model.predict_proba(X_test)[:, 1]
    pair_best = best_threshold(y_test, pair_test_probabilities)
    pair_metrics = metrics_at_threshold(y_test, pair_test_probabilities, float(pair_best["threshold"]))

    group_train, group_test, group_dropped = make_group_split(pairs, args.test_size, args.random_state)
    group_model = build_model()
    group_model.fit(group_train[IMAGE_FEATURE_COLUMNS], group_train["stage1_label"])
    group_test_probabilities = group_model.predict_proba(group_test[IMAGE_FEATURE_COLUMNS])[:, 1]
    group_best = best_threshold(group_test["stage1_label"], group_test_probabilities)
    group_metrics = metrics_at_threshold(group_test["stage1_label"], group_test_probabilities, float(group_best["threshold"]))

    final_model = build_model()
    final_model.fit(pairs[IMAGE_FEATURE_COLUMNS], pairs["stage1_label"])
    pairs["stage1_similarity_probability"] = final_model.predict_proba(pairs[IMAGE_FEATURE_COLUMNS])[:, 1]
    final_threshold = float(pair_best["threshold"])
    pairs["stage1_predicted_similar"] = pairs["stage1_similarity_probability"] >= final_threshold
    pairs["stage2_predicted_type"] = pairs.apply(explain_stage2, axis=1)
    pairs["stage2_table_type"] = pairs.apply(table_stage2_type, axis=1)

    stage1_columns = [
        "query_loan_id",
        "match_loan_id",
        "rank",
        *IMAGE_FEATURE_COLUMNS,
        "stage1_similarity_probability",
        "stage1_predicted_similar",
        "stage1_label",
        "same_similar_group",
        "query_similar_group",
        "match_similar_group",
        "query_path",
        "match_path",
    ]
    stage2_columns = [
        "query_loan_id",
        "match_loan_id",
        "stage1_similarity_probability",
        "stage1_predicted_similar",
        "stage2_predicted_type",
        "stage2_table_type",
        "name_match",
        "id_match",
        "id_conflict",
        "same_iddd_pair",
        "renewal_base_pair",
        "query_edit_type",
        "match_edit_type",
        "query_base_from",
        "match_base_from",
        "query_customer_id_hash",
        "match_customer_id_hash",
        "query_path",
        "match_path",
    ]
    stage1_path = output_dir / "stage1_similarity_report.csv"
    stage2_path = output_dir / "stage2_fraud_type_report.csv"
    pairs[stage1_columns].to_csv(stage1_path, index=False, encoding="utf-8-sig")
    pairs.loc[pairs["stage1_predicted_similar"], stage2_columns].to_csv(stage2_path, index=False, encoding="utf-8-sig")

    summary = {
        "stage1": {
            "purpose": "image-only similar_group detection",
            "image_features": IMAGE_FEATURE_COLUMNS,
            "pair_level_split": {
                "rows_train": int(len(X_train)),
                "rows_test": int(len(X_test)),
                "best_threshold": pair_best,
                "metrics": pair_metrics,
            },
            "group_level_split": {
                "rows_train": int(len(group_train)),
                "rows_test": int(len(group_test)),
                "rows_dropped_cross_split": int(len(group_dropped)),
                "best_threshold": group_best,
                "metrics": group_metrics,
            },
            "final_threshold_for_reports": final_threshold,
            "final_predicted_similar": int(pairs["stage1_predicted_similar"].sum()),
        },
        "stage2": {
            "purpose": "fraud/renewal type explanation for Stage-1 similar pairs",
            "predicted_type_counts": dict(Counter(pairs.loc[pairs["stage1_predicted_similar"], "stage2_predicted_type"])),
            "table_type_counts_on_predicted_similar": dict(Counter(pairs.loc[pairs["stage1_predicted_similar"], "stage2_table_type"])),
        },
        "outputs": [str(stage1_path), str(stage2_path), str(output_dir / "two_stage_summary.json")],
    }
    summary_path = output_dir / "two_stage_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
