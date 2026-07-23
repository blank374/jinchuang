from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "mvp"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fraud_monitoring import (
    build_fraud_monitoring,
    load_annotations,
    summarize_monitoring,
)
from src.risk_policy import ThresholdPolicy

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
    "business_loan_id": "业务贷款号",
    "query_business_type": "查询贷款业务类型",
    "match_business_type": "匹配贷款业务类型",
    "official_similar": "官方标注相似",
    "split": "数据划分",
    "customer_relation": "客户关系",
    "customer_relation_label": "客户关系",
    "customer_relation_source": "客户关系判定来源",
    "identity_evidence_level": "身份依据强度",
    "query_customer_id": "查询身份证哈希",
    "match_customer_id": "匹配身份证哈希",
    "query_customer_id_status": "查询身份证识别状态",
    "match_customer_id_status": "匹配身份证识别状态",
    "monitor_threshold": "监测阈值",
    "is_suspicious": "是否可疑",
    "fraud_type": "监测类型",
    "fraud_type_label_zh": "监测类型",
    "monitor_risk_level": "监测风险等级",
    "review_priority": "复核优先级",
    "recommended_action_zh": "建议处置",
    "score_gap_to_threshold": "超过阈值",
    "query_business_loan_id": "查询业务贷款号",
    "match_business_loan_id": "匹配业务贷款号",
    "fraud_score": "综合欺诈分",
    "score_component_similarity": "评分：影像相似度",
    "score_component_threshold_margin": "评分：超阈值幅度",
    "score_component_customer_relation": "评分：跨客户关系",
    "score_component_cross_product": "评分：跨产品复用",
    "score_component_node_degree": "评分：节点连接度",
    "score_component_cluster_size": "评分：风险簇规模",
    "fraud_score_level_zh": "综合风险",
    "risk_cluster_id": "风险关系簇",
    "risk_cluster_size": "簇内业务数",
    "query_risk_degree": "查询节点连接数",
    "match_risk_degree": "匹配节点连接数",
    "cross_business_scene": "是否跨产品",
    "innovation_tags": "创新监测标签",
    "global_semantic_similarity": "全局语义相似度",
    "subject_region_hist_similarity": "主体区域相似度",
    "background_hist_similarity": "背景相似度",
    "local_structure_orb_ratio": "局部结构匹配",
    "dhash_similarity": "感知哈希相似度",
    "mirror_local_structure_orb_ratio": "镜像局部结构匹配",
    "mirror_subject_region_hist_similarity": "镜像主体相似度",
    "mirror_background_hist_similarity": "镜像背景相似度",
    "mirror_dhash_similarity": "镜像感知哈希相似度",
    "equalized_dhash_similarity": "均衡化感知哈希相似度",
    "edge_dhash_similarity": "边缘哈希相似度",
    "edge_hist_similarity": "边缘结构相似度",
    "rotated_dhash_similarity": "旋转感知哈希相似度",
    "rotated_dhash_gain": "旋转哈希增益",
    "rotated_edge_dhash_similarity": "旋转边缘哈希相似度",
    "rotated_edge_dhash_gain": "旋转边缘增益",
    "brightness_delta": "亮度差",
    "contrast_delta": "对比度差",
    "lab_delta_e2000": "CIEDE2000色差",
    "hsv_hist_similarity": "HSV色彩分布相似度",
    "blur_ratio": "清晰度比例",
    "stage1_similarity_probability": "阶段一相似概率",
    "probability_predicted_similar": "概率是否命中",
    "visual_override_predicted_similar": "视觉兜底是否命中",
    "visual_override_reason": "视觉兜底原因",
    "stage1_decision_source": "阶段一判定来源",
    "stage1_predicted_similar": "阶段一预测相似",
    "stage1_label": "CSV相似标签",
    "same_similar_group": "同similar_group",
    "stage2_predicted_type": "阶段二预测类型",
    "stage2_table_type": "CSV对应类型",
    "name_match": "姓名一致",
    "id_match": "身份证一致",
    "id_conflict": "身份证冲突",
    "same_iddd_pair": "same_iddd命中",
}

ERROR_TYPE_LABELS = {
    "TP": "命中正确",
    "FP": "误报：模型判相似，CSV不相似",
    "FN": "漏报：CSV相似，模型未命中",
    "TN": "排除正确",
}

STAGE2_TYPE_LABELS = {
    "cross_customer_fraud": "跨客户疑似欺诈",
    "same_customer_repeat_review": "同客户重复/异常复核",
    "normal_renewal_similarity": "正常续贷相似",
    "same_name_cross_id_fraud": "同名异证重点复核",
    "high_similarity_pending_identity": "高相似待身份核验",
    "same_name_pending_identity": "同名待身份核验",
    "not_suspicious": "非可疑",
    "not_labeled_similar": "CSV未标相似",
    "labeled_similar_unknown_identity": "CSV相似但身份未知",
    "same_customer_renewal_or_repeat": "同客户续贷/重复提交",
}

EVIDENCE_LABELS = {
    "global_semantic_similarity": "整体内容接近",
    "subject_region_hist_similarity": "人物主体区域接近",
    "background_hist_similarity": "背景环境接近",
    "local_structure_orb_ratio": "局部结构/同源痕迹",
    "dhash_similarity": "感知哈希/翻拍裁剪痕迹",
    "mirror_local_structure_orb_ratio": "镜像后局部结构强匹配",
    "mirror_subject_region_hist_similarity": "镜像后人物主体接近",
    "mirror_background_hist_similarity": "镜像后背景环境接近",
    "mirror_dhash_similarity": "镜像后感知哈希接近",
    "equalized_dhash_similarity": "亮度均衡后仍接近",
    "edge_dhash_similarity": "边缘哈希结构接近",
    "edge_hist_similarity": "边缘分布结构接近",
    "rotated_dhash_similarity": "小角度旋转后仍接近",
    "rotated_dhash_gain": "旋转补偿后哈希提升",
    "rotated_edge_dhash_similarity": "旋转后边缘结构接近",
    "rotated_edge_dhash_gain": "旋转补偿后边缘提升",
    "lab_delta_e2000": "感知色差较小",
    "hsv_hist_similarity": "色彩分布接近",
    "brightness_delta": "亮度差异",
    "contrast_delta": "对比度差异",
    "blur_ratio": "清晰度一致性",
}


def stage2_type_label(value: object) -> str:
    text = str(value or "")
    return STAGE2_TYPE_LABELS.get(text, text or "无")


def evidence_label(value: object) -> str:
    text = str(value or "")
    return EVIDENCE_LABELS.get(text, text)

FRAUD_MONITORING_REQUIRED_COLUMNS = [
    "fraud_score",
    "fraud_score_level_zh",
    "risk_cluster_id",
    "risk_cluster_size",
    "query_risk_degree",
    "match_risk_degree",
    "cross_business_scene",
    "innovation_tags",
    "customer_relation_source",
    "identity_evidence_level",
]


def with_chinese_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: FIELD_LABELS.get(column, column) for column in frame.columns}
    return frame.rename(columns=rename_map)


def load_inference_functions():
    from mvp.inference import analyze_image, get_runtime

    return analyze_image, get_runtime

st.set_page_config(page_title="金融影像风险检测", page_icon="search", layout="wide")


def read_json(name: str) -> dict:
    with (OUTPUT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_outputs() -> None:
    missing = [
        name
        for name in (
            "run_summary.json",
            "classification_metrics.json",
            "threshold_metadata.json",
            "classification_predictions.csv",
            "topk_results.csv",
            "threshold_experiment.csv",
        )
        if not (OUTPUT / name).exists()
    ]
    if missing:
        st.error("尚未生成完整 MVP 结果，请先运行：python -m mvp.pipeline")
        st.code("\n".join(missing), language="text")
        st.stop()


def find_annotations_path(summary: dict) -> Path | None:
    dataset_root = Path(summary["dataset_root"])
    candidate = dataset_root / "annotations.csv"
    return candidate if candidate.exists() else None


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    require_outputs()
    summary = read_json("run_summary.json")
    metrics = read_json("classification_metrics.json")
    threshold_metadata = read_json("threshold_metadata.json")
    predictions = pd.read_csv(OUTPUT / "classification_predictions.csv")
    topk = pd.read_csv(OUTPUT / "topk_results.csv")
    thresholds = pd.read_csv(OUTPUT / "threshold_experiment.csv")
    monitoring_path = OUTPUT / "fraud_monitoring.csv"
    monitoring_summary_path = OUTPUT / "fraud_monitoring_summary.json"
    monitoring = pd.read_csv(monitoring_path) if monitoring_path.exists() else pd.DataFrame()
    monitoring_summary = read_json("fraud_monitoring_summary.json") if monitoring_summary_path.exists() else {}
    two_stage_summary_path = OUTPUT / "two_stage_summary.json"
    two_stage_summary = read_json("two_stage_summary.json") if two_stage_summary_path.exists() else {}
    stage1_path = OUTPUT / "stage1_similarity_report.csv"
    stage2_path = OUTPUT / "stage2_fraud_type_report.csv"
    stage1 = pd.read_csv(stage1_path) if stage1_path.exists() else pd.DataFrame()
    stage2 = pd.read_csv(stage2_path) if stage2_path.exists() else pd.DataFrame()
    review_labels_path = OUTPUT / "review_labels.csv"
    review_labels = pd.read_csv(review_labels_path) if review_labels_path.exists() else pd.DataFrame()
    annotations_path = find_annotations_path(summary)
    annotations = load_annotations(annotations_path)
    return {
        "summary": summary,
        "metrics": metrics,
        "threshold_metadata": threshold_metadata,
        "predictions": predictions,
        "topk": topk,
        "monitoring": monitoring,
        "monitoring_summary": monitoring_summary,
        "two_stage_summary": two_stage_summary,
        "stage1": stage1,
        "stage2": stage2,
        "thresholds": thresholds,
        "review_labels": review_labels,
        "annotations": annotations,
        "annotations_path": str(annotations_path) if annotations_path else "",
    }


def pair_key(left: str, right: str) -> str:
    return "::".join(sorted([left, right]))


def loan_business_frame(annotations: pd.DataFrame) -> pd.DataFrame:
    if annotations.empty:
        return pd.DataFrame(columns=["loan_id", "business_loan_id", "business_type", "similar_group"])
    face = annotations[annotations["image_type"].eq("face_signing")].copy()
    if "dataset_loan_id" not in face.columns:
        face["dataset_loan_id"] = face["loan_id"]
    business_type = face["business_type"] if "business_type" in face.columns else pd.Series("", index=face.index)
    result = pd.DataFrame(
        {
            "loan_id": face["dataset_loan_id"].astype(str),
            "business_loan_id": face["loan_id"].astype(str),
            "business_type": business_type.fillna("").astype(str),
            "similar_group": face["similar_group"].fillna("").astype(str),
            "is_similar_pair": face["is_similar_pair"],
        }
    )
    return result.drop_duplicates("loan_id")


def enrich_topk(topk: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    loans = loan_business_frame(annotations)
    enriched = topk.copy()
    enriched["pair_key"] = [pair_key(a, b) for a, b in zip(enriched["query_loan_id"], enriched["match_loan_id"])]
    if not loans.empty:
        enriched = enriched.merge(
            loans.add_prefix("query_"),
            left_on="query_loan_id",
            right_on="query_loan_id",
            how="left",
        )
        enriched = enriched.merge(
            loans.add_prefix("match_"),
            left_on="match_loan_id",
            right_on="match_loan_id",
            how="left",
        )
    for column in ("query_similar_group", "match_similar_group", "query_business_type", "match_business_type"):
        if column not in enriched.columns:
            enriched[column] = ""
    enriched["official_similar"] = (
        enriched["query_similar_group"].fillna("").ne("")
        & enriched["query_similar_group"].eq(enriched["match_similar_group"])
    )
    return enriched


def unique_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (
        frame.sort_values(["cosine_similarity", "rank"], ascending=[False, True])
        .drop_duplicates("pair_key")
        .reset_index(drop=True)
    )


def official_positive_pair_count(annotations: pd.DataFrame) -> int:
    if annotations.empty:
        return 0
    face = annotations[
        annotations["image_type"].eq("face_signing")
        & annotations["similar_group"].fillna("").ne("")
    ].copy()
    if "dataset_loan_id" not in face.columns:
        face["dataset_loan_id"] = face["loan_id"]
    total = 0
    for _, group in face.groupby("similar_group"):
        total += len(list(combinations(group["dataset_loan_id"].tolist(), 2)))
    return total


def threshold_metrics(frame: pd.DataFrame, annotations: pd.DataFrame, threshold: float) -> dict:
    pairs = unique_pairs(frame)
    selected = pairs[pairs["cosine_similarity"] >= threshold]
    tp = int(selected["official_similar"].sum()) if "official_similar" in selected else 0
    fp = int(len(selected) - tp)
    total_positive = official_positive_pair_count(annotations)
    fn = max(total_positive - tp, 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_positive if total_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "selected": int(len(selected)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_positive": total_positive,
    }


def save_review_label(query_loan_id: str, match_loan_id: str, is_similar: int, note: str) -> None:
    path = OUTPUT / "review_labels.csv"
    columns = ["query_loan_id", "match_loan_id", "is_similar", "note"]
    labels = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    key = pair_key(query_loan_id, match_loan_id)
    labels["pair_key"] = [pair_key(a, b) for a, b in zip(labels["query_loan_id"], labels["match_loan_id"])]
    labels = labels[labels["pair_key"] != key].drop(columns=["pair_key"])
    labels = pd.concat(
        [
            labels,
            pd.DataFrame(
                [
                    {
                        "query_loan_id": query_loan_id,
                        "match_loan_id": match_loan_id,
                        "is_similar": is_similar,
                        "note": note,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    labels.to_csv(path, index=False, encoding="utf-8-sig")
    st.cache_data.clear()


def identity_card_path(face_path: str) -> str:
    return str(Path(str(face_path)).with_name("id_card_front.jpg"))


def mask_identity_hash(value: object) -> str:
    value = str(value or "")
    return f"{value[:10]}…{value[-6:]}" if len(value) > 18 else (value or "未识别")


def save_identity_review(query_loan_id: str, match_loan_id: str, decision: str, note: str) -> None:
    path = OUTPUT / "identity_review.csv"
    columns = ["query_loan_id", "match_loan_id", "decision", "note"]
    reviews = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    key = pair_key(query_loan_id, match_loan_id)
    if not reviews.empty:
        reviews["pair_key"] = [pair_key(a, b) for a, b in zip(reviews["query_loan_id"], reviews["match_loan_id"])]
        reviews = reviews[reviews["pair_key"] != key].drop(columns=["pair_key"])
    reviews = pd.concat([reviews, pd.DataFrame([{"query_loan_id": query_loan_id, "match_loan_id": match_loan_id, "decision": decision, "note": note}])], ignore_index=True)
    reviews.to_csv(path, index=False, encoding="utf-8-sig")
    st.cache_data.clear()


def save_stage1_review(query_loan_id: str, match_loan_id: str, decision: str, note: str) -> None:
    path = OUTPUT / "stage1_review.csv"
    columns = ["query_loan_id", "match_loan_id", "decision", "note"]
    reviews = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    key = pair_key(query_loan_id, match_loan_id)
    if not reviews.empty:
        reviews["pair_key"] = [pair_key(a, b) for a, b in zip(reviews["query_loan_id"], reviews["match_loan_id"])]
        reviews = reviews[reviews["pair_key"] != key].drop(columns=["pair_key"])
    reviews = pd.concat([reviews, pd.DataFrame([{"query_loan_id": query_loan_id, "match_loan_id": match_loan_id, "decision": decision, "note": note}])], ignore_index=True)
    reviews.to_csv(path, index=False, encoding="utf-8-sig")
    st.cache_data.clear()


def read_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame(columns=columns)


def review_decision_to_label(value: object) -> int | None:
    text = str(value or "").strip()
    if text in {"确认相似", "相似", "similar", "true", "1", "yes"}:
        return 1
    if text in {"确认不相似", "误报/不采用", "不相似", "not_similar", "false", "0", "no"}:
        return 0
    return None


def load_manual_review_pairs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stage1_reviews = read_optional_csv(
        OUTPUT / "stage1_review.csv",
        ["query_loan_id", "match_loan_id", "decision", "note"],
    )
    for item in stage1_reviews.itertuples(index=False):
        label = review_decision_to_label(getattr(item, "decision", ""))
        if label is None:
            continue
        rows.append(
            {
                "query_loan_id": str(item.query_loan_id),
                "match_loan_id": str(item.match_loan_id),
                "is_similar": label,
                "decision": str(item.decision),
                "note": str(getattr(item, "note", "")),
                "source": "stage1_review",
            }
        )

    review_labels = read_optional_csv(
        OUTPUT / "review_labels.csv",
        ["query_loan_id", "match_loan_id", "is_similar", "note"],
    )
    for item in review_labels.itertuples(index=False):
        label = review_decision_to_label(getattr(item, "is_similar", ""))
        if label is None:
            continue
        rows.append(
            {
                "query_loan_id": str(item.query_loan_id),
                "match_loan_id": str(item.match_loan_id),
                "is_similar": label,
                "decision": "确认相似" if label else "确认不相似",
                "note": str(getattr(item, "note", "")),
                "source": "review_labels",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["query_loan_id", "match_loan_id", "is_similar", "decision", "note", "source", "pair_key"])
    frame = pd.DataFrame(rows)
    frame["pair_key"] = [pair_key(a, b) for a, b in zip(frame["query_loan_id"], frame["match_loan_id"])]
    return frame.drop_duplicates("pair_key", keep="last").reset_index(drop=True)


RENEWAL_EDIT_TYPES = {"bg", "hair", "shirt", "shirt_bg", "background", "clothes", "background_change", "hair_change", "clothes_change"}


def collapse_duplicate_annotation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("file_path", sort=False):
        merged = {}
        for column in frame.columns:
            values = [str(value) for value in group[column].fillna("").tolist() if str(value)]
            if column == "similar_group":
                merged[column] = next((value for value in values if value), "")
            elif column == "is_similar_pair":
                merged[column] = "1" if "1" in values else (values[0] if values else "")
            else:
                merged[column] = max(values, key=len) if values else ""
        rows.append(merged)
    return pd.DataFrame(rows, columns=frame.columns)


def reviewed_annotations_outputs(annotation_path: str, manual_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not annotation_path:
        return pd.DataFrame(), pd.DataFrame(), {"error": "未找到 annotations.csv"}
    source_path = Path(annotation_path)
    if not source_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {"error": f"标注文件不存在：{source_path}"}

    annotations_raw = pd.read_csv(source_path, dtype=str).fillna("")
    if "file_path" not in annotations_raw.columns:
        return annotations_raw, pd.DataFrame(), {"error": "annotations.csv 缺少 file_path 字段"}

    corrected = annotations_raw.copy()
    normalized_paths = corrected["file_path"].astype(str).str.replace("\\", "/", regex=False)
    corrected["dataset_loan_id"] = normalized_paths.str.split("/").str[0]
    if "image_type" not in corrected.columns:
        corrected["image_type"] = normalized_paths.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    corrected = collapse_duplicate_annotation_rows(corrected)
    normalized_paths = corrected["file_path"].astype(str).str.replace("\\", "/", regex=False)
    corrected["dataset_loan_id"] = normalized_paths.str.split("/").str[0]
    corrected["image_type"] = normalized_paths.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]

    face_mask = corrected["image_type"].eq("face_signing")
    face_loans = sorted(corrected.loc[face_mask, "dataset_loan_id"].dropna().astype(str).unique().tolist())
    loan_set = set(face_loans)

    original_edges: set[tuple[str, str]] = set()
    original_groups = corrected.loc[
        face_mask & corrected["similar_group"].astype(str).ne(""),
        ["dataset_loan_id", "similar_group"],
    ]
    for _, group in original_groups.groupby("similar_group"):
        loans = sorted(group["dataset_loan_id"].astype(str).unique().tolist())
        for left, right in combinations(loans, 2):
            original_edges.add(tuple(sorted((left, right))))

    manual_positive: set[tuple[str, str]] = set()
    manual_negative: set[tuple[str, str]] = set()
    renewal_edges: set[tuple[str, str]] = set()
    if {"edit_type", "base_from"}.issubset(corrected.columns):
        renewal = corrected.loc[face_mask].copy()
        renewal["edit_type_norm"] = renewal["edit_type"].fillna("").astype(str).str.lower()
        renewal["base_from_loan_id"] = renewal["base_from"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
        renewal = renewal[
            renewal["edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
            & renewal["base_from_loan_id"].isin(loan_set)
            & renewal["dataset_loan_id"].ne(renewal["base_from_loan_id"])
        ]
        renewal_edges = {tuple(sorted((str(row.dataset_loan_id), str(row.base_from_loan_id)))) for row in renewal.itertuples(index=False)}
    override_rows = []
    for item in manual_pairs.itertuples(index=False):
        left = str(item.query_loan_id)
        right = str(item.match_loan_id)
        key = tuple(sorted((left, right)))
        if left not in loan_set or right not in loan_set:
            override_rows.append(
                {
                    "query_loan_id": left,
                    "match_loan_id": right,
                    "manual_is_similar": int(item.is_similar),
                    "source": item.source,
                    "note": item.note,
                    "status": "loan_not_found_in_annotations",
                }
            )
            continue
        if int(item.is_similar):
            manual_positive.add(key)
            status = "manual_positive"
        else:
            manual_negative.add(key)
            status = "manual_negative"
        override_rows.append(
            {
                "query_loan_id": left,
                "match_loan_id": right,
                "manual_is_similar": int(item.is_similar),
                "source": item.source,
                "note": item.note,
                "status": status,
            }
        )

    edges = (original_edges | renewal_edges | manual_positive) - manual_negative
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)

    group_by_loan: dict[str, str] = {}
    visited: set[str] = set()
    group_number = 1
    manual_positive_split = 0
    for loan in face_loans:
        if loan in visited:
            continue
        stack = [loan]
        component: list[str] = []
        visited.add(loan)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        constrained_components: list[list[str]] = []
        for component_loan in sorted(component, key=lambda value: (-len(graph[value]), value)):
            for candidate in constrained_components:
                if all(tuple(sorted((component_loan, existing))) not in manual_negative for existing in candidate):
                    candidate.append(component_loan)
                    break
            else:
                constrained_components.append([component_loan])

        for constrained_component in constrained_components:
            if len(constrained_component) < 2:
                continue
            group_id = f"MRG_{group_number:04d}"
            group_number += 1
            for component_loan in constrained_component:
                group_by_loan[component_loan] = group_id

    for left, right in manual_positive:
        if group_by_loan.get(left) != group_by_loan.get(right):
            manual_positive_split += 1

    corrected.loc[face_mask, "similar_group"] = corrected.loc[face_mask, "dataset_loan_id"].map(group_by_loan).fillna("")
    corrected.loc[face_mask, "is_similar_pair"] = corrected.loc[face_mask, "similar_group"].astype(str).ne("").astype(int).astype(str)
    corrected = corrected.drop(columns=["dataset_loan_id"], errors="ignore")

    overrides = pd.DataFrame(override_rows)
    conflicts = 0
    if not overrides.empty:
        positive_keys = {pair_key(a, b) for a, b in manual_positive}
        negative_keys = {pair_key(a, b) for a, b in manual_negative}
        conflicts = len(positive_keys & negative_keys)

    stats = {
        "source_annotations": str(source_path),
        "manual_pairs": int(len(manual_pairs)),
        "manual_positive": int(len(manual_positive)),
        "manual_negative": int(len(manual_negative)),
        "renewal_base_edges": int(len(renewal_edges)),
        "original_positive_edges": int(len(original_edges)),
        "corrected_groups": int(group_number - 1),
        "corrected_positive_loans": int(corrected.loc[face_mask, "similar_group"].astype(str).ne("").sum()),
        "conflicting_manual_pairs": int(conflicts),
        "manual_positive_pairs_split_by_negative_constraints": int(manual_positive_split),
    }
    return corrected, overrides, stats


def save_reviewed_annotations(annotation_path: str, manual_pairs: pd.DataFrame) -> dict:
    corrected, overrides, stats = reviewed_annotations_outputs(annotation_path, manual_pairs)
    if stats.get("error"):
        return stats
    corrected_path = OUTPUT / "annotations_review_corrected.csv"
    overrides_path = OUTPUT / "annotation_review_overrides.csv"
    summary_path = OUTPUT / "annotation_review_summary.json"
    corrected.to_csv(corrected_path, index=False, encoding="utf-8-sig")
    overrides.to_csv(overrides_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats.update(
        {
            "corrected_path": str(corrected_path),
            "overrides_path": str(overrides_path),
            "summary_path": str(summary_path),
        }
    )
    st.cache_data.clear()
    return stats


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


@st.cache_resource(show_spinner=False)
def load_runtime(output_dir: str):
    _, get_runtime = load_inference_functions()
    return get_runtime(output_dir)


data = load_data()
summary = data["summary"]
metrics = data["metrics"]
threshold_metadata = data["threshold_metadata"]
predictions = data["predictions"]
topk = enrich_topk(data["topk"], data["annotations"])
monitoring = data["monitoring"]
if monitoring.empty or any(column not in monitoring.columns for column in FRAUD_MONITORING_REQUIRED_COLUMNS):
    policy = ThresholdPolicy(
        enabled=True,
        same_customer=float(summary.get("same_customer_threshold", 0.92)),
        cross_customer=float(summary.get("cross_customer_threshold", 0.95)),
        default=float(summary["high_risk_threshold"]),
        high_risk=float(summary["high_risk_threshold"]),
        medium_risk=float(summary["medium_risk_threshold"]),
    )
    monitoring = build_fraud_monitoring(data["topk"], data["annotations"], policy)
    monitoring_summary = summarize_monitoring(monitoring)
else:
    monitoring_summary = data["monitoring_summary"] or summarize_monitoring(monitoring)
thresholds = data["thresholds"]
review_labels = data["review_labels"]
annotations = data["annotations"]
two_stage_summary = data["two_stage_summary"]
stage1 = data["stage1"]
stage2 = data["stage2"]
business = loan_business_frame(annotations)
unique_topk = unique_pairs(topk)
manual_review_pairs = load_manual_review_pairs()

st.title("金融影像智能相似度风险检测")

if st.button("刷新数据", help="重新读取 outputs/mvp 下的运行结果和人工审核标注"):
    st.cache_data.clear()
    st.rerun()

selected_threshold = st.slider(
    "当前可疑交易阈值",
    min_value=0.50,
    max_value=1.00,
    value=float(summary["high_risk_threshold"]),
    step=0.01,
)
official_eval = threshold_metrics(topk, annotations, selected_threshold)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("全部影像", summary["total_images"])
col2.metric("面签照片", summary["selected_face_signing"])
col3.metric("测试准确率", f'{metrics["test"]["accuracy"]:.1%}')
col4.metric("召回率", f'{official_eval["recall"]:.1%}')
col5.metric("当前可疑对", official_eval["selected"])

st.caption(
    f'模型：{summary["model_name"]} | 设备：{summary["device"]} | '
    f'上次流水线耗时：{summary["elapsed_seconds"]} 秒 | 标注文件：{data["annotations_path"] or "未找到"}'
)

tab_upload, tab_overview, tab_similarity_review, tab_fraud, tab_graph, tab_risks, tab_classification, tab_threshold, tab_method = st.tabs(
    ["上传检测", "检测汇总", "相似度复核", "风控命中（≥0.95）", "风险关系簇", "高风险候选（≥0.97）", "分类结果", "阈值实验", "后续实验建议"]
)

with tab_upload:
    st.subheader("上传影像自动检测")
    st.markdown("上传单张金融影像后，系统会先判断影像类别；若识别为面签照片，则继续进行相似度比对检测。")
    uploaded = st.file_uploader("上传影像", type=["jpg", "jpeg", "png", "webp", "bmp"])
    force_search = st.checkbox("即使不是面签照片也强制检索", value=False)
    top_k = st.number_input("返回相似结果数量", min_value=1, max_value=20, value=5, step=1)

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="上传影像", width=360)
        if st.button("开始检测", type="primary"):
            with st.spinner("正在加载模型并检测..."):
                analyze_image, _ = load_inference_functions()
                runtime = load_runtime(str(OUTPUT))
                result = analyze_image(image, runtime, top_k=int(top_k), force_search=force_search)

            c1, c2, c3 = st.columns(3)
            c1.metric("预测影像类别", result["predicted_type_label"])
            c2.metric("分类置信度", f'{result["confidence"]:.2%}')
            c3.metric("是否进入相似检索", "是" if result["searched"] else "否")

            score_rows = [
                {"影像类别": item["label"], "模型得分": item["score"]}
                for item in result["class_scores"].values()
            ]
            st.markdown("**分类得分**")
            st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True)

            if not result["searched"]:
                st.info(result.get("message", "未进入相似度检索。"))
            else:
                matches = pd.DataFrame(result["matches"])
                st.markdown("**Top-K 相似度比对结果**")
                st.dataframe(with_chinese_columns(matches), width="stretch", hide_index=True)
                for row in result["matches"]:
                    left, right = st.columns([1, 2])
                    with left:
                        st.image(row["match_path"], caption=f'匹配贷款：{row["match_loan_id"]}', width=280)
                    with right:
                        st.metric("余弦相似度", f'{row["cosine_similarity"]:.4f}')
                        st.write(f'风险等级：**{row["risk_level_label"]}**')
                        st.write(f'相似度排名：`{row["rank"]}`')

with tab_overview:
    st.subheader("数据基础概览")
    risk_counts = unique_topk["risk_level"].value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("高风险候选对", int(risk_counts.get("high", 0)))
    c2.metric("中风险候选对", int(risk_counts.get("medium", 0)))
    c3.metric("官方相似组数", int(business["similar_group"].fillna("").ne("").sum()))
    c4.metric("人工审核样本", int(len(review_labels)))

    left, right = st.columns(2)
    with left:
        st.markdown("**按影像类别统计**")
        st.bar_chart(predictions["image_type"].value_counts())
        st.dataframe(
            predictions.groupby(["image_type", "predicted_type"]).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.markdown("**按业务类型统计**")
        if not business.empty:
            st.bar_chart(business["business_type"].value_counts())
            st.dataframe(
                business.groupby("business_type").agg(
                    loans=("loan_id", "count"),
                    similar_marked=("similar_group", lambda values: int(values.fillna("").ne("").sum())),
                ),
                width="stretch",
            )
        else:
            st.info("未找到 annotations.csv，暂不能展示业务分类统计。")

    st.markdown("**当前阈值下的检测指标**")
    eval_table = pd.DataFrame(
        [
            {
                "threshold": selected_threshold,
                "precision": official_eval["precision"],
                "recall": official_eval["recall"],
                "f1": official_eval["f1"],
                "selected_pairs": official_eval["selected"],
                "true_positive": official_eval["tp"],
                "false_positive": official_eval["fp"],
                "known_positive_pairs": official_eval["total_positive"],
            }
        ]
    )
    st.dataframe(eval_table, width="stretch", hide_index=True)
    if two_stage_summary:
        st.markdown("**两阶段模型核心指标**")
        stage1_summary = two_stage_summary.get("stage1", {})
        pair_metrics = stage1_summary.get("pair_level_split", {}).get("metrics", {})
        group_metrics = stage1_summary.get("group_level_split", {}).get("metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Group Accuracy", f'{float(group_metrics.get("accuracy", 0)):.1%}')
        m2.metric("Group Precision", f'{float(group_metrics.get("precision", 0)):.1%}')
        m3.metric("Group Recall", f'{float(group_metrics.get("recall", 0)):.1%}')
        m4.metric("Group F1", f'{float(group_metrics.get("f1", 0)):.1%}')
        m5.metric("Group ROC-AUC", f'{float(group_metrics.get("roc_auc", 0)):.1%}')
        st.caption(
            f'Pair-level F1：{float(pair_metrics.get("f1", 0)):.1%}；'
            f'阶段一最终相似候选：{int(stage1_summary.get("final_predicted_similar", 0))} 对。'
        )

with tab_similarity_review:
    st.subheader("两阶段相似度复核与错误分析")
    st.caption("阶段一只使用图像多维特征判断是否同 similar_group；阶段二仅对相似候选结合姓名、身份证和业务标签解释风险类型。")
    if stage1.empty or not two_stage_summary:
        st.info("尚未生成两阶段结果，请先运行：python scripts/build_two_stage_pipeline.py")
    else:
        stage1_view = stage1.copy()
        stage1_view["pred_bool"] = bool_series(stage1_view["stage1_predicted_similar"])
        stage1_view["label_bool"] = bool_series(stage1_view["stage1_label"])
        stage1_view["error_type"] = "TN"
        stage1_view.loc[stage1_view["pred_bool"] & stage1_view["label_bool"], "error_type"] = "TP"
        stage1_view.loc[stage1_view["pred_bool"] & ~stage1_view["label_bool"], "error_type"] = "FP"
        stage1_view.loc[~stage1_view["pred_bool"] & stage1_view["label_bool"], "error_type"] = "FN"
        stage1_view["error_type_label"] = stage1_view["error_type"].map(ERROR_TYPE_LABELS).fillna(stage1_view["error_type"])
        stage1_view["pair_key"] = [pair_key(a, b) for a, b in zip(stage1_view["query_loan_id"], stage1_view["match_loan_id"])]

        if not stage2.empty:
            stage2_keys = stage2.copy()
            stage2_keys["pair_key"] = [pair_key(a, b) for a, b in zip(stage2_keys["query_loan_id"], stage2_keys["match_loan_id"])]
            stage1_view = stage1_view.merge(
                stage2_keys[["pair_key", "stage2_predicted_type", "stage2_table_type", "name_match", "id_match", "id_conflict", "same_iddd_pair"]],
                on="pair_key",
                how="left",
            )
        for column in ["stage2_predicted_type", "stage2_table_type", "name_match", "id_match", "id_conflict", "same_iddd_pair"]:
            if column not in stage1_view.columns:
                stage1_view[column] = ""
        stage1_view["stage2_predicted_type_label"] = stage1_view["stage2_predicted_type"].map(stage2_type_label)
        stage1_view["stage2_table_type_label"] = stage1_view["stage2_table_type"].map(stage2_type_label)

        pair_metrics = two_stage_summary.get("stage1", {}).get("pair_level_split", {}).get("metrics", {})
        group_metrics = two_stage_summary.get("stage1", {}).get("group_level_split", {}).get("metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Group F1", f'{float(group_metrics.get("f1", 0)):.1%}')
        c2.metric("Group Precision", f'{float(group_metrics.get("precision", 0)):.1%}')
        c3.metric("Group Recall", f'{float(group_metrics.get("recall", 0)):.1%}')
        c4.metric("Pair F1", f'{float(pair_metrics.get("f1", 0)):.1%}')
        c5.metric("相似候选", int(two_stage_summary.get("stage1", {}).get("final_predicted_similar", 0)))

        st.markdown("**人工复核闭环**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("已复核 pair", int(len(manual_review_pairs)))
        r2.metric("确认相似", int(manual_review_pairs["is_similar"].astype(str).eq("1").sum()) if not manual_review_pairs.empty else 0)
        r3.metric("确认不相似", int(manual_review_pairs["is_similar"].astype(str).eq("0").sum()) if not manual_review_pairs.empty else 0)
        r4.metric("原始标注文件", "已找到" if data["annotations_path"] else "未找到")
        if manual_review_pairs.empty:
            st.info("还没有可导出的人工复核结论。先在下方详情里保存“确认相似/确认不相似”。")
        else:
            st.dataframe(
                with_chinese_columns(manual_review_pairs[["query_loan_id", "match_loan_id", "is_similar", "decision", "source", "note"]]),
                width="stretch",
                hide_index=True,
            )
            if st.button("导出复核修正版 annotations", type="primary"):
                export_stats = save_reviewed_annotations(data["annotations_path"], manual_review_pairs)
                if export_stats.get("error"):
                    st.error(export_stats["error"])
                else:
                    st.success("已导出复核修正版标注文件。")
                    st.json(export_stats)

        counts = stage1_view["error_type_label"].value_counts().reindex([ERROR_TYPE_LABELS[key] for key in ["TP", "FP", "FN", "TN"]]).fillna(0).astype(int)
        st.markdown("**预测与 CSV similar_group 对比**")
        st.bar_chart(counts)

        f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
        with f1:
            error_options = ["全部"] + [ERROR_TYPE_LABELS[key] for key in ["FP", "FN", "TP", "TN"]]
            selected_error_label = st.selectbox("复核类型", error_options, index=1)
            reverse_error_labels = {value: key for key, value in ERROR_TYPE_LABELS.items()}
            selected_error = reverse_error_labels.get(selected_error_label, "全部")
        with f2:
            min_probability = st.slider("最低相似概率", 0.0, 1.0, 0.0, 0.01)
        with f3:
            max_rows = st.number_input("最多显示", min_value=20, max_value=1000, value=200, step=20)
        with f4:
            type_values = sorted([value for value in stage1_view["stage2_predicted_type"].dropna().astype(str).unique().tolist() if value])
            type_options = ["全部"] + [stage2_type_label(value) for value in type_values]
            selected_stage2_type_label = st.selectbox("阶段二类型", type_options)
            selected_stage2_type = {stage2_type_label(value): value for value in type_values}.get(selected_stage2_type_label, "全部")

        review_view = stage1_view[stage1_view["stage1_similarity_probability"] >= min_probability].copy()
        if selected_error != "全部":
            review_view = review_view[review_view["error_type"].eq(selected_error)]
        if selected_stage2_type != "全部":
            review_view = review_view[review_view["stage2_predicted_type"].astype(str).eq(selected_stage2_type)]
        review_view = review_view.sort_values(["error_type", "stage1_similarity_probability"], ascending=[True, False]).head(int(max_rows))

        table_columns = [
            "error_type_label",
            "query_loan_id",
            "match_loan_id",
            "stage1_similarity_probability",
            "stage1_decision_source",
            "visual_override_reason",
            "stage1_predicted_similar",
            "stage1_label",
            "stage2_predicted_type_label",
            "stage2_table_type_label",
            "global_semantic_similarity",
            "subject_region_hist_similarity",
            "background_hist_similarity",
            "local_structure_orb_ratio",
            "dhash_similarity",
            "mirror_local_structure_orb_ratio",
            "mirror_subject_region_hist_similarity",
            "mirror_background_hist_similarity",
            "mirror_dhash_similarity",
            "equalized_dhash_similarity",
            "edge_dhash_similarity",
            "edge_hist_similarity",
            "rotated_dhash_similarity",
            "rotated_dhash_gain",
            "rotated_edge_dhash_similarity",
            "rotated_edge_dhash_gain",
            "lab_delta_e2000",
            "hsv_hist_similarity",
            "brightness_delta",
            "contrast_delta",
            "blur_ratio",
        ]
        st.dataframe(with_chinese_columns(review_view[[column for column in table_columns if column in review_view.columns]]), width="stretch", hide_index=True)

        if not review_view.empty:
            labels = [
                f'{row.error_type_label} | {row.query_loan_id} ↔ {row.match_loan_id} | P={row.stage1_similarity_probability:.3f}'
                for row in review_view.itertuples()
            ]
            selected_label = st.selectbox("查看复核详情", labels)
            row = review_view.iloc[labels.index(selected_label)]
            left, right = st.columns(2)
            with left:
                st.image(row["query_path"], caption=f'查询：{row["query_loan_id"]}')
            with right:
                st.image(row["match_path"], caption=f'匹配：{row["match_loan_id"]}')

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("相似概率", f'{float(row["stage1_similarity_probability"]):.3f}')
            d2.metric("复核类型", row["error_type_label"])
            d3.metric("CSV标签", "相似" if bool(row["label_bool"]) else "不相似")
            d4.metric("模型预测", "相似" if bool(row["pred_bool"]) else "不相似")

            st.markdown("**多维图像依据**")
            evidence_columns = [
                "global_semantic_similarity",
                "subject_region_hist_similarity",
                "background_hist_similarity",
                "local_structure_orb_ratio",
                "dhash_similarity",
                "mirror_local_structure_orb_ratio",
                "mirror_subject_region_hist_similarity",
                "mirror_background_hist_similarity",
                "mirror_dhash_similarity",
                "equalized_dhash_similarity",
                "edge_dhash_similarity",
                "edge_hist_similarity",
                "rotated_dhash_similarity",
                "rotated_dhash_gain",
                "rotated_edge_dhash_similarity",
                "rotated_edge_dhash_gain",
                "lab_delta_e2000",
                "hsv_hist_similarity",
                "brightness_delta",
                "contrast_delta",
                "blur_ratio",
            ]
            evidence = pd.DataFrame(
                [{"依据": evidence_label(column), "字段": column, "数值": row.get(column, "")} for column in evidence_columns]
            )
            st.dataframe(evidence, width="stretch", hide_index=True)

            st.markdown("**阶段二业务解释**")
            st.dataframe(
                with_chinese_columns(
                    pd.DataFrame(
                        [
                            {
                                "stage2_predicted_type": row.get("stage2_predicted_type_label", ""),
                                "stage2_table_type": row.get("stage2_table_type_label", ""),
                                "name_match": row.get("name_match", ""),
                                "id_match": row.get("id_match", ""),
                                "id_conflict": row.get("id_conflict", ""),
                                "same_iddd_pair": row.get("same_iddd_pair", ""),
                                "query_similar_group": row.get("query_similar_group", ""),
                                "match_similar_group": row.get("match_similar_group", ""),
                            }
                        ]
                    )
                ),
                width="stretch",
                hide_index=True,
            )

            with st.form("stage1_review_form"):
                decision = st.radio("人工复核结论", ["确认相似", "确认不相似", "标签需检查", "暂不确定"], horizontal=True)
                note = st.text_input("复核备注", value=f"stage1_{row['error_type'].lower()}_review")
                submitted = st.form_submit_button("保存复核")
            if submitted:
                save_stage1_review(row["query_loan_id"], row["match_loan_id"], decision, note)
                st.success("已保存到 outputs/mvp/stage1_review.csv")

with tab_fraud:
    st.subheader("风控命中：按客户关系分层阈值（跨客户 ≥ 0.95）")
    suspicious = monitoring[monitoring["is_suspicious"].astype(bool)].copy() if not monitoring.empty else pd.DataFrame()
    fraud_counts = suspicious["fraud_type"].value_counts() if not suspicious.empty else pd.Series(dtype=int)
    priority_counts = suspicious["review_priority"].value_counts() if not suspicious.empty else pd.Series(dtype=int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("监测候选对", int(monitoring_summary.get("total_pairs", len(monitoring))))
    c2.metric("可疑命中", int(monitoring_summary.get("suspicious_pairs", len(suspicious))))
    c3.metric("确认跨客户欺诈（强依据）", int(fraud_counts.get("cross_customer_fraud", 0)))
    c4.metric("跨客户高风险候选", int(fraud_counts.get("cross_customer_candidate", 0)))

    st.caption("严格校验的 customer_id_hash 不同，才显示为确认跨客户欺诈；比赛模拟编号仅做格式级匹配时，显示为跨客户高风险候选。similar_group 只用于离线评估与阈值校准。")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("风险关系簇", int(monitoring_summary.get("risk_cluster_count", 0)))
    g2.metric("最大簇业务数", int(monitoring_summary.get("max_risk_cluster_size", 0)))
    g3.metric("跨产品可疑", int(monitoring_summary.get("cross_business_suspicious", 0)))
    g4.metric("极高风险候选", int(monitoring_summary.get("critical_alerts", 0)))

    left, right = st.columns(2)
    with left:
        st.markdown("**按监测类型**")
        if not suspicious.empty:
            st.bar_chart(suspicious["fraud_type_label_zh"].value_counts())
        else:
            st.info("当前阈值下没有命中可疑监测记录。")
    with right:
        st.markdown("**按复核优先级**")
        if not suspicious.empty:
            st.bar_chart(priority_counts)
        else:
            st.info("暂无复核优先级统计。")
    if not stage2.empty:
        st.markdown("**两阶段风险类型解释（Stage 2）**")
        stage2_display = stage2.copy()
        stage2_display["stage2_predicted_type_label"] = stage2_display["stage2_predicted_type"].map(stage2_type_label)
        stage2_display["stage2_table_type_label"] = stage2_display["stage2_table_type"].map(stage2_type_label)
        stage2_left, stage2_right = st.columns(2)
        with stage2_left:
            st.caption("模型预测类型")
            st.bar_chart(stage2_display["stage2_predicted_type_label"].fillna("未知").value_counts())
        with stage2_right:
            st.caption("CSV 对应类型")
            st.bar_chart(stage2_display["stage2_table_type_label"].fillna("未知").value_counts())

    type_options = ["全部", "陌生人跨客户疑似欺诈", "同客户重复提交", "低风险候选"]
    selected_type = st.selectbox("监测类型", type_options)
    relation_options = ["全部"] + sorted(monitoring["customer_relation_label"].dropna().unique().tolist())
    selected_relation = st.selectbox("客户关系", relation_options)
    only_suspicious = st.checkbox("只看可疑命中", value=True)

    fraud_view = monitoring.copy()
    if only_suspicious:
        fraud_view = fraud_view[fraud_view["is_suspicious"].astype(bool)]
    if selected_type != "全部":
        fraud_view = fraud_view[fraud_view["fraud_type_label_zh"].eq(selected_type)]
    if selected_relation != "全部":
        fraud_view = fraud_view[fraud_view["customer_relation_label"].eq(selected_relation)]
    fraud_view = fraud_view.sort_values(["monitor_risk_level", "cosine_similarity"], ascending=[True, False])

    columns = [
        "query_loan_id",
        "match_loan_id",
        "query_business_loan_id",
        "match_business_loan_id",
        "cosine_similarity",
        "monitor_threshold",
        "fraud_score",
        "fraud_score_level_zh",
        "score_gap_to_threshold",
        "customer_relation_label",
        "customer_relation_source",
        "identity_evidence_level",
        "query_customer_id_status",
        "match_customer_id_status",
        "fraud_type_label_zh",
        "risk_cluster_id",
        "risk_cluster_size",
        "query_risk_degree",
        "match_risk_degree",
        "cross_business_scene",
        "innovation_tags",
        "monitor_risk_level",
        "review_priority",
        "recommended_action_zh",
    ]
    available_columns = [column for column in columns if column in fraud_view.columns]
    st.dataframe(with_chinese_columns(fraud_view[available_columns]), width="stretch", hide_index=True)

    if not fraud_view.empty:
        labels = [
            f'{row.query_loan_id} ↔ {row.match_loan_id} | {row.cosine_similarity:.4f} | {row.fraud_type_label_zh}'
            for row in fraud_view.itertuples()
        ]
        selected_label = st.selectbox("查看监测详情", labels)
        row = fraud_view.iloc[labels.index(selected_label)]
        left, right = st.columns(2)
        with left:
            st.image(row["query_path"], caption=f'查询：{row["query_loan_id"]} / {row.get("query_business_loan_id", "")}')
        with right:
            st.image(row["match_path"], caption=f'命中：{row["match_loan_id"]} / {row.get("match_business_loan_id", "")}')
        st.markdown("**身份证主键核验**")
        id_left, id_right = st.columns(2)
        with id_left:
            query_id_path = identity_card_path(row["query_path"])
            if Path(query_id_path).exists():
                st.image(query_id_path, caption=f'查询身份证：{row["query_loan_id"]}', width=280)
            else:
                st.warning("未找到查询身份证正面")
        with id_right:
            match_id_path = identity_card_path(row["match_path"])
            if Path(match_id_path).exists():
                st.image(match_id_path, caption=f'匹配身份证：{row["match_loan_id"]}', width=280)
            else:
                st.warning("未找到匹配身份证正面")
        identity_fields = pd.DataFrame([
            {"业务": "查询", "贷款": row["query_loan_id"], "身份证哈希": mask_identity_hash(row.get("query_customer_id", "")), "OCR状态": row.get("query_customer_id_status", "未识别")},
            {"业务": "匹配", "贷款": row["match_loan_id"], "身份证哈希": mask_identity_hash(row.get("match_customer_id", "")), "OCR状态": row.get("match_customer_id_status", "未识别")},
        ])
        st.dataframe(identity_fields, width="stretch", hide_index=True)
        st.caption(f'客户关系来源：{row.get("customer_relation_source", "customer_id_unavailable")}。`matched_format_only` 为比赛模拟身份证编号的格式级依据，生产环境需使用严格校验或核心客户号。')
        with st.form("identity_review_form"):
            decision = st.radio("人工核验结论", ["确认跨客户", "确认同客户", "身份证无法辨识", "暂不确定"], horizontal=True)
            identity_note = st.text_input("身份证核验备注", value="manual_identity_review")
            identity_submitted = st.form_submit_button("保存身份证核验结论")
        if identity_submitted:
            save_identity_review(row["query_loan_id"], row["match_loan_id"], decision, identity_note)
            st.success("身份证核验结论已保存到 outputs/mvp/identity_review.csv")
        detail_cols = st.columns(4)
        fraud_score = float(row.get("fraud_score", 0.0) or 0.0)
        detail_cols[0].metric("综合欺诈分", f"{fraud_score:.4f}" if fraud_score else "-")
        detail_cols[1].metric("风险关系簇", row.get("risk_cluster_id", "") or "-")
        detail_cols[2].metric("簇内业务数", int(row.get("risk_cluster_size", 0)))
        is_cross_business = str(row.get("cross_business_scene", False)).lower() in {"true", "1", "yes"}
        detail_cols[3].metric("跨产品", "是" if is_cross_business else "否")
        st.write(f'创新监测标签：**{row.get("innovation_tags", "")}**')
        st.markdown("**处置建议**")
        st.info(row["recommended_action_zh"])

with tab_graph:
    st.subheader("风险关系图谱：业务、客户关系与相似影像")
    st.caption("每条边代表一组达到分层阈值的面签影像相似关系；节点连接度和簇规模用于提升复核优先级，而不替代相似度判定。")
    graph_nodes_path = OUTPUT / "risk_graph_nodes.csv"
    graph_edges_path = OUTPUT / "risk_graph_edges.csv"
    graph_nodes = pd.read_csv(graph_nodes_path) if graph_nodes_path.exists() else pd.DataFrame()
    graph_edges = pd.read_csv(graph_edges_path) if graph_edges_path.exists() else pd.DataFrame()
    if graph_nodes.empty:
        st.info("暂无风险关系簇。运行 MVP 流水线后会自动生成图谱节点和边表。")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("风险节点", len(graph_nodes))
        c2.metric("风险关系边", len(graph_edges))
        c3.metric("高连接节点（≥3）", int((graph_nodes["risk_degree"] >= 3).sum()))
        st.markdown("**风险簇汇总**")
        clusters = graph_nodes.groupby("risk_cluster_id", dropna=False).agg(
            业务数=("loan_id", "count"), 最大连接度=("risk_degree", "max"), 最高欺诈分=("max_fraud_score", "max"),
        ).sort_values(["业务数", "最高欺诈分"], ascending=False)
        st.dataframe(clusters, width="stretch")
        st.markdown("**图谱节点（贷款/业务）**")
        st.dataframe(with_chinese_columns(graph_nodes.sort_values(["risk_degree", "max_fraud_score"], ascending=False)), width="stretch", hide_index=True)
        st.markdown("**图谱关系（相似影像边）**")
        st.dataframe(with_chinese_columns(graph_edges.sort_values("fraud_score", ascending=False)), width="stretch", hide_index=True)

with tab_risks:
    st.subheader("高风险候选：统一高风险阈值 ≥ 0.97")
    st.caption("该视图用于紧急优先级队列；0.95–0.97 的风控命中仍会展示在“风控命中（≥0.95）”页。")
    risk_filter = st.multiselect("风险等级", ["high", "medium", "low"], default=["high"])
    business_types = sorted(set(topk["query_business_type"].dropna()) | set(topk["match_business_type"].dropna()))
    selected_business = st.selectbox("业务类型", ["全部"] + business_types)

    risk_view = unique_topk[unique_topk["risk_level"].isin(risk_filter)].copy()
    if selected_business != "全部":
        risk_view = risk_view[
            risk_view["query_business_type"].eq(selected_business)
            | risk_view["match_business_type"].eq(selected_business)
        ]
    risk_view = risk_view.sort_values("cosine_similarity", ascending=False)
    st.dataframe(
        with_chinese_columns(risk_view[
            [
                "query_loan_id",
                "match_loan_id",
                "cosine_similarity",
                "risk_level",
                "query_business_type",
                "match_business_type",
                "official_similar",
            ]
        ]),
        width="stretch",
        hide_index=True,
    )

    if not risk_view.empty:
        labels = [
            f'{row.query_loan_id} ↔ {row.match_loan_id} | {row.cosine_similarity:.4f} | {row.risk_level}'
            for row in risk_view.itertuples()
        ]
        selected_label = st.selectbox("查看可疑交易详情", labels)
        row = risk_view.iloc[labels.index(selected_label)]
        left, right = st.columns(2)
        with left:
            st.image(row["query_path"], caption=f'贷款 {row["query_loan_id"]} · {row.get("query_business_type", "")}')
        with right:
            st.image(row["match_path"], caption=f'贷款 {row["match_loan_id"]} · {row.get("match_business_type", "")}')

        st.markdown("**关联业务数据**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "role": "query",
                        "loan_id": row["query_loan_id"],
                        "business_type": row.get("query_business_type", ""),
                        "similar_group": row.get("query_similar_group", ""),
                    },
                    {
                        "role": "match",
                        "loan_id": row["match_loan_id"],
                        "business_type": row.get("match_business_type", ""),
                        "similar_group": row.get("match_similar_group", ""),
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**人工复核标注**")
        with st.form("review_form"):
            label = st.radio("这条可疑交易是否真实相似？", ["确认相似", "误报/不采用"], horizontal=True)
            note = st.text_input("备注", value="manual_review")
            submitted = st.form_submit_button("保存标注")
        if submitted:
            save_review_label(
                row["query_loan_id"],
                row["match_loan_id"],
                1 if label == "确认相似" else 0,
                note,
            )
            st.success("已保存标注，并会计入刷新后的人工审核统计。")
            st.rerun()

with tab_classification:
    st.subheader("分类与面签筛选")
    selected_type = st.selectbox("预测类别", ["全部"] + sorted(predictions["predicted_type"].unique().tolist()))
    view = predictions if selected_type == "全部" else predictions[predictions["predicted_type"] == selected_type]
    st.dataframe(
        with_chinese_columns(view[["loan_id", "image_type", "predicted_type", "confidence", "split", "relative_path"]]),
        width="stretch",
        hide_index=True,
    )

with tab_threshold:
    st.subheader("阈值、准确率、召回率与复核成本")
    st.markdown(
        "- 下表保留单一余弦相似度阈值实验，用于和多维模型做对照。\n"
        "- 当前主流程使用 Stage 1 多维图像证据概率做相似检测，similar_group 仅作比赛离线真值。\n"
        "- Stage 2 再使用客户主键、姓名、same_iddd、edit_type 解释续贷或欺诈类型。"
    )
    if two_stage_summary:
        stage1_summary = two_stage_summary.get("stage1", {})
        pair_metrics = stage1_summary.get("pair_level_split", {}).get("metrics", {})
        group_metrics = stage1_summary.get("group_level_split", {}).get("metrics", {})
        threshold_table = pd.DataFrame(
            [
                {"split": "group-level", **group_metrics},
                {"split": "pair-level", **pair_metrics},
            ]
        )
        st.markdown("**Stage 1 多维模型阈值结果**")
        st.dataframe(threshold_table, width="stretch", hide_index=True)
    threshold_rows = []
    for threshold in [0.85, 0.90, 0.93, 0.95, 0.97, 0.98]:
        item = threshold_metrics(topk, annotations, threshold)
        threshold_rows.append({"threshold": threshold, **item})
    st.dataframe(pd.DataFrame(threshold_rows), width="stretch", hide_index=True)
    st.line_chart(thresholds.set_index("threshold")[["precision", "recall", "f1"]])
    st.warning(threshold_metadata["note"])
    if not review_labels.empty:
        st.markdown("**人工审核标注**")
        st.dataframe(review_labels, width="stretch", hide_index=True)

with tab_method:
    st.subheader("方法说明与后续实验建议")
    st.markdown(
        "当前版本已经拆成两阶段：Stage 1 只用图像多维证据检测是否属于同一 `similar_group`；"
        "Stage 2 只对相似候选结合身份证哈希、姓名、`same_iddd` 和 `edit_type` 解释续贷或欺诈类型。"
    )
    st.markdown(
        "**Stage 1 图像证据：**\n"
        "1. 全局语义相似度：面签照片 embedding 余弦相似度。\n"
        "2. 人物主体代理：中心区域颜色直方图相似度。\n"
        "3. 背景环境：边缘背景区域直方图相似度。\n"
        "4. 局部结构：ORB 局部匹配和 dHash 感知哈希。\n"
        "5. 图像质量：亮度差、对比度差、清晰度比例。"
    )
    st.markdown(
        "**Stage 2 类型解释：**\n"
        "1. 高相似 + 身份证一致：同客户重复提交/异常复用复核，不直接定性为欺诈。\n"
        "2. 高相似 + 身份证冲突：跨客户疑似欺诈。\n"
        "3. 高相似 + 姓名一致但身份证不同：同名异证重点复核。\n"
        "4. 高相似 + 身份缺失：待身份核验高风险。"
    )
    st.markdown(
        "**建议优先补的实验：**\n"
        "1. 把中心区域代理升级为人脸/人体检测后的主体 embedding。\n"
        "2. 对 `edit_type` 分组评估：亮度、对比度、旋转、裁剪、镜像，以及背景、头发、衣服变化。\n"
        "3. 固定使用 group-level split 作为正式报告主指标，pair-level 只作为辅助对照。\n"
        "4. 增加人工复核闭环：把 FP/FN 的复核结果写回训练集，周期性重训 Stage 1。\n"
        "5. 增加分场景阈值：当产品、网点、拍摄规范差异明显时，比较统一阈值和分组阈值。"
    )
