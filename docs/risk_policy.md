# 分层阈值与图谱风控策略

系统将“影像是否相似”和“该相似关系是否构成业务风险”分开处理。SigLIP2 多模态视觉 embedding 与 FAISS Top-K 负责召回候选面签影像；业务字段和风险关系图谱负责定性、分层与排序。因此不能只使用一个统一阈值。

## 客户关系与比赛标注边界

线上判定的优先级是：`loan_id` 排除自身匹配，`customer_id`（或经治理的客户主键）区分同客户与跨客户。缺少客户主键时，系统只输出 `cross_customer_candidate`（高相似待客户关系核验），不会把它直接定性为跨客户欺诈。

比赛数据中的 `similar_group` 是脱敏后的 Ground Truth，仅用于离线统计 Precision、Recall、F1 和阈值校准，绝不作为 API、Gradio 或生产规则的输入字段。这样可避免标签泄漏。

| 阈值 | 作用 | 准确率 / 召回率 / 人工成本影响 |
|---|---|---|
| 0.97 high risk | 高置信自动升级、紧急复核 | 精确率最高，召回较低，人工成本最低 |
| 0.93 medium risk | 二审/抽检候选池 | 提升召回，带来适中的误报与复核量 |
| 0.92 same customer repeat | 同客户重复提交的运营/合规复核 | 同一客户历史复用本身常是正常续贷；较低阈值减少漏检，复核不直接认定欺诈 |
| 0.95 cross customer fraud | 跨客户疑似欺诈的反欺诈复核 | 跨客户误报代价高，采用更严格阈值以控制误报和反欺诈人工成本 |

同一个 0.94 的候选对，在同客户场景可作为 `same_customer_repeat` 进入合规复核，在跨客户场景则暂不作为欺诈命中；这正是统一阈值无法表达的业务语义。流水线会优先使用比赛 Ground Truth 生成阈值扫描报告；如需采用 F1 最优点，可加 `--use-calibrated-high-threshold`，并结合 Precision 下限与人工复核容量复核后上线。

## 图谱和综合欺诈分

批处理会把贷款/业务作为节点，把达到对应分层阈值的相似面签影像作为边，输出：

- `fraud_monitoring.csv`：逐候选对的风险类型、阈值、标签和可解释分量。
- `risk_graph_nodes.csv` 与 `risk_graph_edges.csv`：可直接导入图数据库或看板。
- `fraud_monitoring_summary.json`：命中、簇、跨产品和极高风险统计。

综合欺诈分为 `0.52×影像相似度 + 0.24×超阈值幅度 + 客户关系 + 跨产品 + 节点连接度 + 风险簇规模`，截断到 1。其不是替代阈值的黑盒分类器，而是已命中候选的复核排序依据。解释标签包括：跨客户高相似、同客户重复、跨产品复用、风险关系簇、高连接节点、极高综合欺诈分。

## 运行

```powershell
F:\Environment\conda_envs\pytorch\python.exe -m mvp.pipeline
F:\Environment\conda_envs\pytorch\python.exe -m streamlit run experiments\dashboard.py
F:\Environment\conda_envs\pytorch\python.exe api.py --port 8000
```

API 提供 `POST /classify`、`POST /search`、`POST /batch-search` 和 `GET /monitoring-report`。批量接口接受多文件上传并对每张图片完成分类、面签筛选和 Top-K 检索。
