# 金融影像智能相似度检测系统

**赛题**: 第五届中国研究生金融科技创新大赛——"揭榜挂帅"赛题  
**命题单位**: 无锡农村商业银行股份有限公司  
**模型**: CLIP ViT-B/32 (openai/clip-vit-base-patch32)

---

## 目录

- [功能概述](#功能概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [使用说明](#使用说明)
- [评估结果](#评估结果)
- [项目结构](#项目结构)

---

## 功能概述

本系统基于 CLIP 多模态预训练模型，构建"分类过滤 + 特征提取 + 向量检索"三段式架构，实现金融影像的智能分类与相似度检测：

- **影像分类**: 零样本识别 5 类影像（面签照片 / 身份证 / 合同 / 银行流水 / 其他）
- **面签筛选**: 自动从海量影像中筛选面签照片，进入相似度检测
- **相似度检测**: 对面签照片进行特征提取、FAISS 向量检索，标记高风险重复提交
- **批量检测**: 多张图片同时检测，导出 CSV 报告
- **REST API**: 提供 HTTP 接口，支持系统集成
- **对比学习微调**: 支持 TripletMarginLoss 对 CLIP 模型微调，提升金融影像域内区分能力

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9+（推荐 3.10） |
| 操作系统 | Windows / Linux / macOS |
| 内存 | ≥ 8 GB |
| 硬盘 | ≥ 1 GB 可用空间 |
| GPU | 可选（CPU 可运行，单张推理 ~0.3 秒） |

---

## 快速开始

### 1. 克隆项目

```bash
git clone <项目地址>
cd Finance_Image_Similarity
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖清单：

| 包名 | 版本 | 用途 |
|------|------|------|
| gradio | ≥4.44.1 | Web 演示界面 |
| transformers | ≥4.21.0 | CLIP 模型加载 |
| torch | ≥1.10.0 | 深度学习框架 |
| faiss-cpu | ≥1.7.0 | 向量检索 |
| opencv-python-headless | ≥4.8.0 | 图像预处理 |
| fastapi / uvicorn | ≥0.104.0 | REST API |

> **注意**: 如果使用中国大陆网络，代码中已配置镜像 `HF_ENDPOINT=https://hf-mirror.com`，自动从 HuggingFace 镜像站下载模型。若需切换回官方源，修改 `main.py`、`ingest.py`、`api.py` 中的 `HF_ENDPOINT` 环境变量即可。

### 4. 数据入库

将数据集图片（按 `data/` 目录结构组织，含 `annotations.csv` 标注文件）写入 FAISS 索引：

```bash
# 默认方式（自动检测 data/annotations.csv）
python ingest.py --data_dir ./data

# 强制重建索引
python ingest.py --data_dir ./data --force

# 使用 IVF 索引（大数据量推荐）
python ingest.py --data_dir ./data --index_type ivf

# 手动指定标注文件
python ingest.py --data_dir ./data --annotations ./data/annotations.csv
```

成功输出示例（首次入库）：
```
已加载标注信息: 370 条
预处理链: auto_rotate → enhance_contrast → correct_perspective
找到 370 张图片，开始处理...
入库完成！统计:
  成功入库: 370 张
  索引总记录: 370 条
```

> **注意**: 重复运行 `ingest.py` 会在现有索引上追加记录（变为 740 条等）。使用 `--force` 参数可重建索引。

### 5. 启动系统

提供两种运行方式：

#### 方式一：Gradio Web 界面（推荐）

```bash
python main.py
```

访问 `http://localhost:7860` 打开操作界面。

界面包含：
- **单图检测**: 上传图片 → 自动分类 → 面签筛选 → 相似度检索 → 结果展示
- **批量检测**: 多图上传 → 汇总报告 + CSV 导出
- **索引统计**: 查看 FAISS 索引状态

#### 方式二：FastAPI REST 服务

```bash
python api.py --host 0.0.0.0 --port 8000
```

接口文档：`http://localhost:8000/docs`

可用端点：

```
GET  /health         健康检查
GET  /stats          索引统计信息
POST /classify       图片分类
POST /search         图片分类 + 相似度检索
```

调用示例：

```bash
# 分类
curl -X POST -F "file=@photo.jpg" http://localhost:8000/classify

# 检索
curl -X POST -F "file=@photo.jpg" http://localhost:8000/search
```

### 6. 对比学习训练（可选）

```bash
# 使用三元组数据训练（验证训练流程）
python src/train.py --data_dir ./data_triplets --epochs 2

# 完整训练 20 个 epoch
python src/train.py --data_dir ./data_triplets --epochs 20
```

---

## 系统架构

```
输入影像 → 预处理增强 → CLIP 零样本分类 → 面签照片 → CLIP 特征提取 → L2归一化 → FAISS 向量检索 → 相似度分数
                                                    ↓ 非面签
                                                跳过检测
```

### 预处理链

| 步骤 | 功能 | 解决的实际问题 |
|------|------|--------------|
| auto_rotate | EXIF 自动旋转 | 手机/相机拍摄方向不一致 |
| enhance_contrast | CLAHE 光照增强 | 面签现场光照不足/过曝 |
| perspective_correct | 文档透视矫正 | 合同/证件翻拍角度不正 |

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| 特征提取 | [src/model.py](src/model.py) | CLIP 图像编码器封装 |
| 影像分类 | [src/classifier.py](src/classifier.py) | 零样本分类 + 面签判定 |
| 预处理 | [src/preprocessing.py](src/preprocessing.py) | 图像增强管道 |
| 向量检索 | [src/retrieval.py](src/retrieval.py) | FAISS 索引（Flat / IVF） |
| 数据入库 | [ingest.py](ingest.py) | 图片扫描 → 特征提取 → 索引入库 |
| Web 界面 | [main.py](main.py) | Gradio 交互界面 |
| REST API | [api.py](api.py) | FastAPI HTTP 接口 |
| 对比学习 | [src/train.py](src/train.py) | TripletMarginLoss 微调 |
| 评估工具 | [src/evaluate.py](src/evaluate.py) | 分类 + 检索评估 |

---

## 使用说明

### 单张检测

1. 打开 Gradio 界面
2. 上传一张图片（支持 jpg/png/bmp/tiff）
3. 点击"开始检测"
4. 查看分类结果（类别 + 各类别得分）
5. 如果是面签照片，自动进入相似度检测
6. 查看相似度分数、可疑标记、最高相似历史影像

### 批量检测

1. 在 Gradio 界面选择"批量检测"标签页
2. 上传多张图片（可多选）
3. 点击"开始批量检测"
4. 系统自动生成 CSV 报告（保存至 `reports/` 目录），包含：
   - 影像类别、面签置信度
   - 最高相似度分数、判定结果
   - 相似业务 ID、相似业务类型
5. 可直接下载 CSV 报告

### 配置调整

编辑 [config.yaml](config.yaml) 可调整：

```yaml
retrieval:
  similarity_threshold: 0.93    # 相似度判定阈值（基于真实数据评估确定 F1=0.9197 @ t=0.93）
  index_type: "flat"            # 索引类型 (flat / ivf)

preprocessing:
  auto_rotate: true             # 预处理链开关
  enhance_contrast: true
  perspective_correct: true

classifier:
  categories: [...]             # 分类类别 + prompt 定义

training:
  epochs: 20                    # 对比学习训练参数
```

---

## 评估结果

本系统在赛题数据集（74 笔贷款，370 张影像，15 个相似组）上评估的结果：

### 相似度检索评估

| 指标 | 数值 |
|------|------|
| 相似组 Top-5 命中率 | 100% |
| 最优阈值 | 0.93 |
| 最优 F1（阈值 0.93） | 0.9197 |
| 单张推理时间 | ~0.3 秒 |
| 同组相似度均值 | 0.9592 |
| 异组相似度均值 | 0.8732 |

### 分类评估

| 指标 | 数值 |
|------|------|
| 分类准确率 | 98.38% |

### 详细报告

- 检索评估报告：[eval_results/retrieval_evaluation.md](eval_results/retrieval_evaluation.md)
- 完整阈值分析：[eval_results/evaluation_report.md](eval_results/evaluation_report.md)
- PR 曲线图：[eval_results/pr_curve.png](eval_results/pr_curve.png)
- 技术报告（含消融实验）：[技术报告.md](技术报告.md)

### 运行评估

```bash
# 使用配置文件中的测试数据
python src/evaluate.py

# 自定义测试目录和阈值
python src/evaluate.py --test_dir ./test_eval --thresholds 0.5 0.6 0.7 0.8 0.9 0.95
```

---

## 数据说明

| 项目 | 数值 |
|------|------|
| 贷款笔数 | 74 笔 |
| 影像总数 | 370 张 |
| 每笔贷款影像数 | 5 张 |
| 影像类型 | 面签照片、身份证正面、身份证反面、合同、银行流水 |
| 业务类型 | 消费贷 / 商户易贷 / 锡微贷 |
| 相似组 | 15 个（39 张面签照片，模拟跨贷款重复提交人脸） |

---

## 项目结构

```
Finance_Image_Similarity/
├── main.py                  # Gradio Web 界面（主入口）
├── api.py                   # FastAPI REST 服务
├── ingest.py                # 数据入库脚本
├── config.yaml              # 系统配置文件
├── requirements.txt         # Python 依赖
├── README.md                # 本文件
├── 技术报告.md               # 技术报告（含消融实验与结果分析）
├── 部署文档.pdf              # 部署与运维说明
├── 23-多模态技术与数据治理赛道-无锡农商行-基于多模态大模型的金融影像智能相似度检测模型.docx
├── src/
│   ├── __init__.py
│   ├── model.py             # CLIP 特征提取器
│   ├── classifier.py        # 零样本分类器
│   ├── preprocessing.py     # 图像预处理链
│   ├── retrieval.py         # FAISS 向量检索
│   ├── dataset.py           # 对比学习数据集
│   ├── train.py             # 对比学习训练
│   └── evaluate.py          # 评估工具
├── data/                    # 赛题数据集（74 笔贷款，370 张影像）
│   ├── annotations.csv      # 标注信息（影像类型 + 相似组 + 业务类型）
│   ├── 数据集说明.pdf        # 赛题配套数据说明
│   ├── loan_001/            # 贷款 1 影像
│   ├── loan_002/            # 贷款 2 影像
│   └── ...
├── checkpoints/
│   ├── faiss_index.bin      # FAISS 索引文件
│   ├── faiss_index_meta.pkl # 索引元数据
│   └── finetuned/           # 对比学习训练输出
│       ├── final_model.pt
│       └── training_history.json
├── eval_results/            # 评估输出
│   ├── evaluation_report.md
│   ├── retrieval_evaluation.md
│   └── pr_curve.png
└── reports/                 # 批量检测 CSV 报告输出（运行时自动创建）
```

---

## 许可

本系统为第五届中国研究生金融科技创新大赛参赛作品。
