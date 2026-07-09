# 金融影像智能相似度检测 MVP

本项目实现赛题主流程：

```text
金融影像 -> 数据清洗 -> SigLIP2 分类 -> 面签筛选 -> Embedding
-> FAISS Top-K -> Cosine Similarity -> 阈值判定 -> Dashboard
```

## 运行

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m pip install -r requirements-mvp.txt
.\run_mvp.ps1
F:\Environment\conda_envs\pytorch\python.exe -m streamlit run dashboard.py
```

首次运行会下载 `google/siglip2-base-patch16-224`。Windows CPU 环境建议
`BatchSize=4~8`；当前电脑的 `pytorch` Conda 环境会自动使用 GTX 1650，
Apple Silicon 会自动使用 MPS。

## 输出

全部实验产物保存在 `outputs/mvp/`：

- `data_manifest.csv`：清洗后的数据清单。
- `classification_predictions.csv`：分类与面签筛选结果。
- `classification_metrics.json`：训练、验证、测试集指标。
- `image_embeddings.npy`：SigLIP2 图像向量。
- `face_signing.faiss`：面签照片 FAISS 索引。
- `topk_results.csv`：Top-K、余弦相似度和风险等级。
- `threshold_experiment.csv`：阈值、Precision、Recall、F1 和复核量。

当前数据没有官方相似图片对标签，因此阈值实验使用“原图与确定性轻微增强”为正样本、
不同贷款面签照为负样本的代理评估。正式报告应使用人工复核或赛题提供的相似对标签重新标定阈值。
