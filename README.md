# 金融影像智能相似度检测系统

本项目面向金融影像审核场景，构建“影像分类 -> 面签筛选 -> 向量检索 -> 分层风控”的检测流程。系统以 **SigLIP2** 作为主视觉语言表征模型，以 **CLIP** 作为 baseline 对照，通过 FAISS 完成相似影像召回，并结合客户/业务关系输出可解释的风险等级和复核建议。

## 当前方案

- 主模型：`google/siglip2-base-patch16-224`
- baseline：`openai/clip-vit-base-patch32`
- 向量维度：`768`
- 检索后端：FAISS `IndexFlatIP`，可切换 IVF
- 核心策略：跨客户与同客户采用不同风险阈值

整体流程：

```text
金融影像上传
-> 图像预处理增强
-> SigLIP2 影像类型识别
-> 筛出面签照片
-> SigLIP2 图像向量提取
-> FAISS Top-K 相似检索
-> 结合客户/业务关系做分层阈值判断
-> 输出风险等级、复核优先级和处置建议
```

## 核心亮点

1. **SigLIP2 主模型**  
   使用更新的视觉语言模型提取金融影像语义特征，CLIP 保留为 baseline 便于对比说明。

2. **面签照片定向检测**  
   系统先识别影像类型，只让面签照片进入相似度检索，避免身份证、合同、流水等非目标影像干扰风险判断。

3. **分层阈值风控**  
   同样的相似度在不同业务关系下含义不同：

   - 跨客户相似：疑似冒用/套用，阈值较低，优先提高召回；
   - 同客户相似：可能是续贷/复用，阈值较高，减少误报；
   - 自身匹配：自动跳过。

4. **可解释输出**  
   API 和页面不只返回 similarity score，还会返回：

   - `risk_level`
   - `risk_type`
   - `review_priority`
   - `recommended_action`
   - `risk_summary`

## 项目结构

```text
.
├── main.py                  # Gradio Web 界面
├── api.py                   # FastAPI REST 服务
├── ingest.py                # 数据入库与 FAISS 索引构建
├── config.yaml              # 模型、阈值、预处理与应用配置
├── src/
│   ├── model.py             # SigLIP2/CLIP 通用视觉语言编码器
│   ├── classifier.py        # 基于文本 prompt 的影像类型识别
│   ├── retrieval.py         # FAISS 向量检索
│   ├── preprocessing.py     # 图像预处理增强
│   ├── risk_policy.py       # 分层风控策略
│   ├── train.py             # 可选对比学习训练
│   ├── losses.py            # Triplet / dual margin loss
│   └── evaluate.py          # 评估工具
├── checkpoints/             # 索引与训练产物
└── reports/                 # 批量检测报告
```

## 环境准备

推荐使用项目现有 Python 环境：

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m pip install -r requirements.txt
```

如果首次运行时本地没有 SigLIP2 权重，会从 HuggingFace 镜像下载。代码已设置：

```text
HF_ENDPOINT=https://hf-mirror.com
```

## 数据入库

模型已从 CLIP 512 维切换为 SigLIP2 768 维，所以旧 FAISS 索引不能复用，必须重新建索引。

```powershell
cd D:\GitHub\jinchuang\remote-main-merge-worktree

F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<数据目录>" --annotations "<数据目录>\annotations.csv" --force
```

如果数据目录中已有 `annotations.csv`，也可以省略 `--annotations`：

```powershell
F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<数据目录>" --force
```

常用参数：

```powershell
# 追加入库
F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<数据目录>"

# 强制重建索引
F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<数据目录>" --force

# 使用 IVF 索引
F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<数据目录>" --index_type ivf --force
```

## 启动 Web 页面

```powershell
cd D:\GitHub\jinchuang\remote-main-merge-worktree
F:\Environment\conda_envs\pytorch\python.exe main.py
```

浏览器访问终端输出的地址，通常是：

```text
http://127.0.0.1:7860
```

Web 页面支持：

- 单图检测；
- 批量检测；
- 相似图片展示；
- FAISS 索引统计；
- 风险等级与复核建议展示。

## 启动 API

```powershell
cd D:\GitHub\jinchuang\remote-main-merge-worktree
F:\Environment\conda_envs\pytorch\python.exe api.py --host 127.0.0.1 --port 8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

主要接口：

```text
GET  /health
GET  /stats
POST /classify
POST /search
```

`POST /search` 返回示例字段：

```json
{
  "model": "google/siglip2-base-patch16-224",
  "category_name": "面签照片",
  "is_sign_photo": true,
  "dynamic_threshold": {
    "enabled": true,
    "cross_customer_threshold": 0.75,
    "same_customer_threshold": 0.92,
    "default_threshold": 0.97,
    "high_risk_threshold": 0.97,
    "medium_risk_threshold": 0.93
  },
  "risk_summary": {
    "cross_customer_suspect": 1,
    "same_customer_repeat": 0,
    "normal_low_risk": 4
  },
  "similar_results": [
    {
      "similarity": 0.8912,
      "relationship": "cross_customer",
      "risk_level": "medium",
      "risk_type": "cross_customer_suspect",
      "review_priority": "standard",
      "recommended_action": "Send to anti-fraud review; verify identity and loan context before approval."
    }
  ]
}
```

## 风控策略

策略文件：[src/risk_policy.py](src/risk_policy.py)

当前配置：

```yaml
retrieval:
  similarity_threshold: 0.97
  high_risk_threshold: 0.97
  medium_risk_threshold: 0.93
  dynamic_threshold:
    enabled: true
    fraud: 0.75
    same_customer: 0.92
```

解释：

| 场景 | 关系 | 阈值 | 风险含义 |
|---|---|---:|---|
| 跨客户命中 | `cross_customer` | 0.75 | 疑似冒用/套用，优先召回 |
| 同客户命中 | `same_customer` | 0.92 | 可能是续贷/复用，减少误报 |
| 当前可疑交易/高风险 | default/high | 0.97 | 原 MVP 人工审核校准阈值 |
| 中风险候选 | medium | 0.93 | 抽检/二审候选池 |

## 方案表述

推荐在答辩/文档中这样概括：

> 系统以 SigLIP2 作为主视觉语言表征模型，CLIP 作为 baseline。模型负责判断影像“像不像”，业务风控策略负责判断“在金融业务中危险不危险”。通过同客户/跨客户分层阈值，将单纯相似度检索转化为可解释的风险等级、复核优先级和处置建议。

## 注意事项

- 切换模型后必须重建 FAISS 索引；
- `embedding_dim` 必须与主模型输出一致，当前为 768；
- `annotations.csv` 中如果包含 `similar_group`，系统会用于判断同客户/跨客户关系；
- 如果没有业务关系字段，系统会退化为保守的跨客户风险策略；
- CLIP 相关命名在部分代码类名中保留是为了兼容旧调用，实际加载模型由 `config.yaml` 决定。
