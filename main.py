import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
import torch
import numpy as np
from PIL import Image
from src.model import CLIPFeatureExtractor
from src.retrieval import SimilaritySearch
from src.classifier import ImageClassifier
from src.preprocessing import PreprocessingPipeline
import yaml
from collections import Counter

# 加载配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 初始化全局组件
extractor = CLIPFeatureExtractor()
searcher = SimilaritySearch(
    embedding_dim=config["model"]["embedding_dim"],
    index_type=config["retrieval"].get("index_type", "flat"),
)

# 初始化分类器（复用 CLIP 模型）
classifier = ImageClassifier(
    model=extractor.model,
    processor=extractor.processor,
    categories=config["classifier"]["categories"],
    device=extractor.device,
)

# 初始化预处理链
preprocessor = PreprocessingPipeline(config.get("preprocessing", {}))
print(f"预处理链: {preprocessor.describe()}")

# 加载索引
if searcher.load():
    print(f"索引已加载，共 {searcher.index.ntotal} 条记录 ({searcher.index_type})")
else:
    print("警告: 未找到索引，请先运行 python ingest.py --data_dir <图片目录>")


def get_index_stats():
    """获取索引统计信息"""
    if not searcher.index or searcher.index.ntotal == 0:
        return "索引为空。"

    total = searcher.index.ntotal
    cat_counter = Counter()
    for m in searcher.metadata:
        cat_counter[m.get("cat_name", "未知")] += 1
    category_dist = dict(cat_counter.most_common())

    fake_count = sum(1 for m in searcher.metadata if "fake_img" in m.get("path", ""))

    lines = [f"索引统计 ({searcher.index_type}, {total} 条记录)"]
    lines.append(f"\n  总记录数: {total}")
    if fake_count:
        lines.append(f"  模拟数据: {fake_count} 条")
    lines.append(f"\n  类别分布:")
    for cat, count in category_dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"    {cat}: {count:>4} 条 ({pct:5.1f}%) {bar}")

    lines.append(f"\n  索引类型: FAISS {searcher.index_type.upper()}")
    lines.append(f"  特征维度: {searcher.embedding_dim}")

    return "\n".join(lines)


def predict(image):
    try:
        if image is None:
            return "请上传图片", None, None, "等待检测..."

        # 1. 预处理（增强链 + 面签人脸裁剪）
        image = preprocessor(image)
        image = image.convert("RGB")
        img_tensor = torch.tensor(np.array(image.resize((224, 224))).transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # 2. 影像分类
        _, cat_name, scores = classifier.classify(img_tensor)
        is_sign, sign_confidence = classifier.is_sign_photo(img_tensor)

        # 格式化分类结果
        scores_str = "\n".join([f"  {k}: {v:.3f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])])
        classification_info = (
            f"【分类结果】\n"
            f"  预测类别: {cat_name}\n"
            f"  面签置信度: {sign_confidence:.3f}\n\n"
            f"【各类别得分】\n{scores_str}"
        )

        # 3. 如果不是面签照片，不进行相似度检测
        if not is_sign:
            return (
                classification_info + "\n\n图片不是面签照片，跳过相似度检测。",
                None,
                None,
                f"非面签照片: {cat_name}",
            )

        # 4. 提取特征
        with torch.no_grad():
            feat = extractor.extract(img_tensor)
        feat_np = feat.cpu().numpy().astype(np.float32)
        feat_norm = np.linalg.norm(feat_np)
        if feat_norm > 0:
            feat_np = feat_np / feat_norm

        # 5. 检索相似
        threshold = config["retrieval"]["similarity_threshold"]
        results = searcher.search(feat_np[0], top_k=config["app"]["top_k"])

        # 6. 格式化输出
        output_lines = [f"【可疑相似结果】（阈值: {threshold}）"]
        suspicious_count = 0
        for i, res in enumerate(results):
            score = res["score"]
            biz_id = res["metadata"]["biz_id"]
            cat = res["metadata"]["cat_name"]
            flag = "⚠️ 可疑" if score >= threshold else ""
            if flag:
                suspicious_count += 1
            output_lines.append(
                f"排名{i+1}: 相似度 {score:.4f} {'>= ' if score >= threshold else '< '}{threshold} {flag}\n"
                f"        业务ID: {biz_id} | 类型: {cat}"
            )

        if suspicious_count == 0:
            output_lines.append(
                f"\n未发现超过阈值（{threshold}）的相似图片，该面签照片通过初步审查。"
            )
        else:
            output_lines.append(
                f"\n发现 {suspicious_count} 条可疑相似记录，建议人工复核。"
            )

        top_img_path = results[0]["metadata"]["path"] if results else None

        return (
            classification_info,
            "\n".join(output_lines),
            top_img_path,
            f"面签照片 | 最高相似度: {results[0]['score']:.3f} | {'可疑' if results[0]['score'] >= threshold else '正常'}" if results else "无结果",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"处理出错: {str(e)}", None, None, "错误"


def batch_predict(images):
    """批量检测：一次上传多张图片，返回汇总报告"""
    if not images:
        return "请上传至少一张图片", None

    import csv
    from datetime import datetime

    threshold = config["retrieval"]["similarity_threshold"]
    rows = []

    for img in images:
        if img is None:
            continue

        try:
            # gr.File file_count="multiple" 返回文件路径列表（字符串）
            if isinstance(img, str):
                img = Image.open(img)
            elif hasattr(img, "read"):
                img = Image.open(img)
            elif not isinstance(img, Image.Image):
                img = Image.open(img)

            # 预处理 + 分类
            processed = preprocessor(img)
            processed = processed.convert("RGB")
            img_tensor = torch.tensor(np.array(processed.resize((224, 224))).transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            _, cat_name, _ = classifier.classify(img_tensor)
            is_sign, sign_conf = classifier.is_sign_photo(img_tensor)
        except Exception as e:
            rows.append({"类别": f"处理失败: {e}", "面签置信度": "-", "是否面签": "否", "最高相似度": "-", "判定结果": f"错误: {e}", "相似业务ID": ""})
            continue

        row = {
            "类别": cat_name,
            "面签置信度": f"{sign_conf:.3f}",
            "是否面签": "是" if is_sign else "否",
        }

        if is_sign:
            # 提取特征
            with torch.no_grad():
                feat = extractor.extract(img_tensor)
            feat_np = feat.cpu().numpy().astype(np.float32)
            feat_norm = np.linalg.norm(feat_np)
            if feat_norm > 0:
                feat_np = feat_np / feat_norm

            results = searcher.search(feat_np[0], top_k=config["app"]["top_k"])
            max_score = results[0]["score"] if results else 0
            suspicious = any(r["score"] >= threshold for r in results)

            row["最高相似度"] = f"{max_score:.4f}"
            row["判定结果"] = "可疑" if suspicious else "正常"
            suspicious_results = [r for r in results if r["score"] >= threshold]
            row["相似业务ID"] = ", ".join(set(r["metadata"]["biz_id"] for r in suspicious_results)) if suspicious_results else ""
            row["相似业务类型"] = ", ".join(set(r["metadata"].get("business_type", "") for r in suspicious_results if r["metadata"].get("business_type"))) if suspicious_results else ""
        else:
            row["最高相似度"] = "-"
            row["判定结果"] = "非面签无需检测"
            row["相似业务ID"] = ""
            row["相似业务类型"] = ""

        rows.append(row)

    # 生成 CSV 报告
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"批量检测报告_{timestamp}.csv")

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["类别", "面签置信度", "是否面签", "最高相似度", "判定结果", "相似业务类型", "相似业务ID"])
        writer.writeheader()
        writer.writerows(rows)

    # 汇总
    total = len(rows)
    suspicious = sum(1 for r in rows if r["判定结果"] == "可疑")
    normal = sum(1 for r in rows if r["判定结果"] == "正常")
    non_sign = sum(1 for r in rows if r["判定结果"] == "非面签无需检测")

    summary = (
        f"批量检测完成: 共 {total} 张\n"
        f"  可疑: {suspicious} 张\n"
        f"  正常: {normal} 张\n"
        f"  非面签: {non_sign} 张\n"
        f"报告已保存: {report_path}"
    )

    # 汇总表文本
    header = f"{'类别':<12} {'面签置信度':<12} {'最高相似度':<12} {'判定结果':<16} {'相似业务类型':<16}"
    sep = "-" * len(header)
    table_lines = [summary, "", header, sep]
    for r in rows:
        table_lines.append(
            f"{r['类别']:<12} {r['面签置信度']:<12} {r['最高相似度']:<12} {r['判定结果']:<16} {r.get('相似业务类型', '-'):<16}"
        )

    return summary + "\n\n" + "\n".join(table_lines), report_path


# 创建界面
with gr.Blocks(title="金融影像智能相似度检测", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 基于多模态大模型的金融影像智能相似度检测
        **功能**: 影像自动分类 → 面签照片识别 → 相似度比对检测
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                input_img = gr.Image(label="上传影像", type="pil", height=300)
                submit_btn = gr.Button("开始检测", variant="primary", size="lg")
            status_box = gr.Textbox(label="状态", value="等待检测...", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("分类结果"):
                    classify_output = gr.Textbox(label="影像分类详情", lines=10, interactive=False)

                with gr.TabItem("相似度检测"):
                    retrieve_output = gr.Textbox(label="检索结果", lines=10, interactive=False)
                    similar_img = gr.Image(label="最高相似历史影像", type="filepath", height=250)

                with gr.TabItem("批量检测"):
                    gr.Markdown("上传多张图片进行批量检测，自动生成 CSV 报告")
                    batch_input = gr.File(label="上传多张图片（可多选）", file_count="multiple", file_types=["image"])
                    batch_btn = gr.Button("开始批量检测", variant="primary", size="lg")
                    batch_output = gr.Textbox(label="检测结果", lines=15, interactive=False)
                    batch_report = gr.File(label="下载报告")

    # 统计面板
    with gr.Accordion("索引统计信息", open=False):
        stats_output = gr.Textbox(label="索引详情", lines=12, interactive=False)
        refresh_btn = gr.Button("刷新统计", size="sm", variant="secondary")
        refresh_btn.click(fn=get_index_stats, inputs=None, outputs=[stats_output])
        demo.load(fn=get_index_stats, inputs=None, outputs=[stats_output])

    # 提示信息
    gr.Markdown(
        """
    ---
    **工作流程**: 上传图片 → 预处理增强 → 自动分类（面签/身份证/权证/合同/其他）→
    如是面签照片则进行相似度检测 → 根据阈值判断是否存在重复提交风险

    **当前索引**: checkpoints/faiss_index.bin ({index_type})
    **相似度阈值**: {threshold}
    **预处理**: {preprocess}
    """.format(
            threshold=config["retrieval"]["similarity_threshold"],
            index_type=config["retrieval"].get("index_type", "flat"),
            preprocess=preprocessor.describe(),
        )
    )

    submit_btn.click(
        fn=predict,
        inputs=input_img,
        outputs=[classify_output, retrieve_output, similar_img, status_box],
    )

    batch_btn.click(
        fn=batch_predict,
        inputs=batch_input,
        outputs=[batch_output, batch_report],
    )

if __name__ == "__main__":
    demo.launch(share=True)
