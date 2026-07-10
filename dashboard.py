from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "mvp"

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


require_outputs()

summary = read_json("run_summary.json")
metrics = read_json("classification_metrics.json")
threshold_metadata = read_json("threshold_metadata.json")
predictions = pd.read_csv(OUTPUT / "classification_predictions.csv")
topk = pd.read_csv(OUTPUT / "topk_results.csv")
thresholds = pd.read_csv(OUTPUT / "threshold_experiment.csv")
review_labels_path = OUTPUT / "review_labels.csv"
review_labels = pd.read_csv(review_labels_path) if review_labels_path.exists() else pd.DataFrame()

st.title("金融影像智能相似度风险检测")

col1, col2, col3, col4 = st.columns(4)
col1.metric("全部影像", summary["total_images"])
col2.metric("面签照片", summary["selected_face_signing"])
col3.metric("测试集准确率", f'{metrics["test"]["accuracy"]:.1%}')
col4.metric("测试集 Macro-F1", f'{metrics["test"]["macro_f1"]:.1%}')

st.info(
    f'最终高风险阈值为 {summary["high_risk_threshold"]:.2f}，'
    f'中风险阈值为 {summary["medium_risk_threshold"]:.2f}。'
    "高风险阈值来自人工审核校准：>=0.97 的 33 组唯一候选已确认相似；0.95~0.97 区间开始出现误报。"
)

tab_overview, tab_classification, tab_search, tab_threshold = st.tabs(
    ["数据概览", "分类结果", "Top-K 检索", "阈值实验"]
)

with tab_overview:
    st.subheader("数据统计")
    counts = predictions.groupby(["image_type", "predicted_type"]).size().reset_index(name="count")
    st.bar_chart(predictions["predicted_type"].value_counts())
    st.dataframe(counts, use_container_width=True, hide_index=True)
    st.caption(f'模型：{summary["model_name"]} | 设备：{summary["device"]}')

with tab_classification:
    st.subheader("分类与面签筛选")
    selected_type = st.selectbox("预测类别", ["全部"] + sorted(predictions["predicted_type"].unique().tolist()))
    view = predictions if selected_type == "全部" else predictions[predictions["predicted_type"] == selected_type]
    st.dataframe(
        view[["loan_id", "image_type", "predicted_type", "confidence", "split", "relative_path"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_search:
    st.subheader("面签照片 Top-K 相似检索")
    threshold = st.slider(
        "风险阈值",
        min_value=0.0,
        max_value=1.0,
        value=float(summary["high_risk_threshold"]),
        step=0.01,
    )
    query_ids = sorted(topk["query_loan_id"].unique().tolist())
    query_id = st.selectbox("查询贷款", query_ids)
    matches = topk[topk["query_loan_id"] == query_id].copy()
    matches["当前判定"] = matches["cosine_similarity"].map(lambda score: "可疑" if score >= threshold else "正常")
    if not matches.empty:
        st.image(matches.iloc[0]["query_path"], caption=f"查询：{query_id}", width=360)
    for _, row in matches.iterrows():
        left, right = st.columns([1, 2])
        with left:
            st.image(row["match_path"], caption=f'Rank {int(row["rank"])} · {row["match_loan_id"]}', width=280)
        with right:
            st.metric("Cosine Similarity", f'{row["cosine_similarity"]:.4f}')
            st.write(f'当前判定：**{row["当前判定"]}**')
            st.write(f'预设风险等级：**{row["risk_level"]}**')
            st.write(f'关联贷款：`{row["match_loan_id"]}`')

with tab_threshold:
    st.subheader("阈值、召回率与人工复核成本")
    st.markdown(
        "- 最终 high 阈值：`0.97`，来自人工审核校准。\n"
        "- 最终 medium 阈值：`0.93`，作为抽检/二审候选池。\n"
        "- 代理阈值实验仅作为参考，正式风险分层优先采用人工审核结论。"
    )
    if not review_labels.empty:
        reviewed = len(review_labels)
        positives = int(review_labels["is_similar"].sum())
        negatives = reviewed - positives
        c1, c2, c3 = st.columns(3)
        c1.metric("已审核候选对", reviewed)
        c2.metric("确认相似", positives)
        c3.metric("确认不相似", negatives)
        st.dataframe(review_labels, use_container_width=True, hide_index=True)
    st.warning(threshold_metadata["note"])
    chart = thresholds.set_index("threshold")[["precision", "recall", "f1"]]
    st.line_chart(chart)
    st.dataframe(thresholds, use_container_width=True, hide_index=True)
    st.json(threshold_metadata["best_f1_threshold"])
