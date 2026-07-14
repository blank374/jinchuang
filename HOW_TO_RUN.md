# 运行说明

本仓库保留两个可运行入口：一个用于最终业务演示，一个用于实验结果展示。

## 1. 业务演示端

用于最终演示：上传影像、自动识别影像类型、筛选面签照片、检索历史影像，并展示高相似可疑交易。

```powershell
cd D:\GitHub\jinchuang
F:\Environment\conda_envs\pytorch\python.exe main.py
```

启动后打开终端输出的 Gradio 地址，通常是：

```text
http://127.0.0.1:7860
```

如果 `7860` 端口已被占用，Gradio 会自动使用附近的其他端口。

## 2. MVP 实验看板

用于展示实验依据：训练集/验证集/测试集指标、阈值分析、检测汇总、高相似可疑交易和方法说明。

```powershell
cd D:\GitHub\jinchuang
F:\Environment\conda_envs\pytorch\python.exe -m streamlit run experiments/dashboard.py --server.address 127.0.0.1 --server.port 8501
```

打开：

```text
http://127.0.0.1:8501
```

## 重建历史影像索引

仓库中的 `checkpoints/` 目录已经包含可直接运行的历史 FAISS 索引。如果需要从影像数据集重新构建索引：

```powershell
cd D:\GitHub\jinchuang
F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<数据集目录>" --force
```

## 重建 MVP 实验结果

仓库中的 `outputs/mvp/` 目录已经包含实验结果。如果需要重新生成：

```powershell
cd D:\GitHub\jinchuang
.\experiments\run_mvp.ps1
```

整理后，实验专用入口统一放在 `experiments/` 目录下；最终业务演示入口仍保留在仓库根目录的 `main.py`。
