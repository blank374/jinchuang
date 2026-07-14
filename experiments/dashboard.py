from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "mvp"

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
    "query_business_type": "查询贷款业务类型",
    "match_business_type": "匹配贷款业务类型",
    "official_similar": "官方标注相似",
    "split": "数据划分",
}


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
    review_labels_path = OUTPUT / "review_labels.csv"
    review_labels = pd.read_csv(review_labels_path) if review_labels_path.exists() else pd.DataFrame()
    annotations_path = find_annotations_path(summary)
    annotations = pd.read_csv(annotations_path) if annotations_path else pd.DataFrame()
    return {
        "summary": summary,
        "metrics": metrics,
        "threshold_metadata": threshold_metadata,
        "predictions": predictions,
        "topk": topk,
        "thresholds": thresholds,
        "review_labels": review_labels,
        "annotations": annotations,
        "annotations_path": str(annotations_path) if annotations_path else "",
    }


def pair_key(left: str, right: str) -> str:
    return "::".join(sorted([left, right]))


def loan_business_frame(annotations: pd.DataFrame) -> pd.DataFrame:
    if annotations.empty:
        return pd.DataFrame(columns=["loan_id", "business_type", "similar_group"])
    face = annotations[annotations["image_type"].eq("face_signing")].copy()
    return face[["loan_id", "business_type", "similar_group", "is_similar_pair"]].drop_duplicates("loan_id")


def enrich_topk(topk: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    loans = loan_business_frame(annotations)
    enriched = topk.copy()
    enriched["pair_key"] = [pair_key(a, b) for a, b in zip(enriched["query_loan_id"], enriched["match_loan_id"])]
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
    ]
    total = 0
    for _, group in face.groupby("similar_group"):
        total += len(list(combinations(group["loan_id"].tolist(), 2)))
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
thresholds = data["thresholds"]
review_labels = data["review_labels"]
annotations = data["annotations"]
business = loan_business_frame(annotations)
unique_topk = unique_pairs(topk)

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

tab_upload, tab_overview, tab_risks, tab_classification, tab_threshold, tab_method = st.tabs(
    ["上传检测", "检测汇总", "高相似可疑交易", "分类结果", "阈值实验", "后续实验建议"]
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

with tab_risks:
    st.subheader("高相似可疑交易")
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
        "- 最终 high 阈值：`0.97`，来自人工审核校准。\n"
        "- 最终 medium 阈值：`0.93`，作为抽检/二审候选池。\n"
        "- 下表同时展示代理实验曲线和当前官方相似组口径下的动态指标。"
    )
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
    st.subheader("后续实验建议")
    st.markdown(
        "当前版本主要依赖通用视觉大模型的整图 embedding。为了回应“背景不固定、工牌不统一”的问题，"
        "下一步建议把面签照片拆成两路特征：一路保留整图场景，一路做人脸区域特征，最终按权重融合。"
    )
    st.markdown(
        "**建议优先补的实验：**\n"
        "1. 人像鲁棒性测试：对面签照片做亮度、裁剪、缩放、旋转、压缩扰动，再看相似度是否稳定。\n"
        "2. 跨业务类型泛化：用 A 类贷款训练/校准阈值，在 B 类贷款上测试准确率和召回率。\n"
        "3. 分业务阈值：如果不同业务类型的相似度分布差异明显，再考虑单独阈值；否则统一阈值更好解释。\n"
        "4. 样本量曲线：用 10%、30%、50%、70%、100% 训练样本分别训练分类头，观察准确率和召回率是否稳定。\n"
        "5. 周期性微调：每新增一批人工复核标注后，手动或定时重新训练分类头/阈值校准，不必每次都微调整个大模型。"
    )
