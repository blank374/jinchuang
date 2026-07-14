# How to Run

This repository has two runnable entry points.

## 1. Business Demo

Use this for the final demo: upload images, classify image type, filter signing photos, search historical images, and show suspicious high-similarity transactions.

```powershell
cd D:\GitHub\jinchuang
F:\Environment\conda_envs\pytorch\python.exe main.py
```

Open the Gradio URL printed by the terminal, usually:

```text
http://127.0.0.1:7860
```

If port 7860 is occupied, Gradio will choose another nearby port.

## 2. MVP Experiment Dashboard

Use this for experiment evidence: train/validation/test metrics, threshold analysis, detection summary, suspicious pairs, and method discussion.

```powershell
cd D:\GitHub\jinchuang
F:\Environment\conda_envs\pytorch\python.exe -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## Rebuild Historical Index

The committed `checkpoints/` directory already contains a runnable historical FAISS index. To rebuild it from an image dataset:

```powershell
cd D:\GitHub\jinchuang
F:\Environment\conda_envs\pytorch\python.exe ingest.py --data_dir "<dataset_dir>" --force
```

## Rebuild MVP Results

The committed `outputs/mvp/` directory already contains experiment outputs. To regenerate them:

```powershell
cd D:\GitHub\jinchuang
.\run_mvp.ps1
```

