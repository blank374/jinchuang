# 金融影像相似度检测 MVP

这个仓库保留一条清晰主线：扫描金融影像数据集，训练轻量分类头筛出面签照片，提取图像向量，使用 FAISS 做 Top-K 相似检索，并通过仪表盘或 API 查看风险结果。

## 项目结构

```text
.
+-- mvp/pipeline.py              # 核心流水线：清洗、分类、向量、检索、阈值实验
+-- dashboard.py                 # Streamlit 可视化仪表盘
+-- api.py                       # FastAPI 查询接口，读取 outputs/mvp 结果
+-- run_mvp.ps1                  # Windows 一键运行脚本
+-- requirements.txt             # 统一依赖
+-- docs/experiment_report.md    # 实验结论与阈值校准报告
+-- outputs/mvp/                 # 运行产物
+-- scripts/                     # 数据画像与复核报告脚本
```

## 安装

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m pip install -r requirements.txt
```

如果没有固定的 Conda 环境，也可以把上面的 Python 路径替换成你的 `python`。

## 运行流水线

```powershell
.\run_mvp.ps1
```

也可以直接运行：

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m mvp.pipeline --batch-size 8 --top-k 5 --device auto
```

默认会自动寻找仓库中以 `23-` 开头的数据集目录。需要指定数据集时：

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m mvp.pipeline --dataset-root "D:\path\to\dataset"
```

## 查看结果

启动仪表盘：

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m streamlit run dashboard.py
```

启动 API：

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

常用接口：

```text
GET /health
GET /summary
GET /predictions?limit=50
GET /matches/{loan_id}
GET /risks?min_score=0.9
GET /calibration
```

## 输出文件

流水线结果保存在 `outputs/mvp/`：

- `data_manifest.csv`：数据清单与坏图检查结果
- `classification_predictions.csv`：分类预测、置信度、数据划分
- `classification_metrics.json`：训练/验证/测试分类指标
- `image_embeddings.npy`：全部有效图片向量
- `face_manifest.csv`：筛出的面签照片清单
- `face_embeddings.npy`：面签照片向量
- `face_signing.faiss`：面签照片 FAISS 索引
- `topk_results.csv`：Top-K 相似检索与风险等级
- `threshold_experiment.csv`：阈值、Precision、Recall、F1、复核量
- `threshold_metadata.json`：阈值实验说明
- `review_labels.csv`：人工审核标注，用于校准阈值
- `run_summary.json`：本次运行摘要

## 当前方案

主模型默认使用 `google/siglip2-base-patch16-224`。分类部分在图像向量上训练一个线性头，相似度部分只对面签照片建索引。当前阈值实验使用“原图的轻微裁剪/亮度增强”作为正样本代理，不同贷款的面签照片作为负样本代理。正式交付前，建议用人工复核标签或赛题官方相似对标注重新校准阈值。

## 实验结论

本次最终高风险阈值采用 `0.97`，中风险阈值采用 `0.93`：

- `cosine_similarity >= 0.97`：high，必须人工复核
- `0.93 <= cosine_similarity < 0.97`：medium，建议抽检或二审
- `cosine_similarity < 0.93`：low，默认低风险

阈值 `0.97` 来自人工审核校准：已审核 44 组候选，其中 `>= 0.97` 的 33 组唯一候选全部确认相似；`0.95 ~ 0.97` 区间开始出现不稳定候选，因此不再下调 high 阈值。

完整实验结论见 [docs/experiment_report.md](docs/experiment_report.md)。
