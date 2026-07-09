"""
FastAPI REST 接口：复用 main.py 中的全局组件，提供 HTTP API 服务

启动:
    python api.py                  # http://localhost:8000
    python api.py --port 8080      # 自定义端口
    uvicorn api:app --host 0.0.0.0 --port 8000  # uvicorn 直接启动

接口:
    POST /classify      上传图片 → 分类
    POST /search        上传图片 → 分类 → 检索相似
    GET  /stats         索引统计信息
    GET  /health        健康检查
"""
import os
import sys
import argparse
import io
from contextlib import asynccontextmanager

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import yaml
import torch
import numpy as np
from PIL import Image
from pydantic import BaseModel

# 延迟初始化（首次请求时加载）

@asynccontextmanager
async def lifespan(app):
    print("API 服务启动中...")
    lazy_init()
    print("API 服务就绪")
    yield

app = FastAPI(
    title="金融影像智能相似度检测 API",
    description="基于多模态大模型的金融影像智能相似度检测服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 全局状态
extractor = None
searcher = None
classifier = None
preprocessor = None
config = None


def lazy_init():
    """懒加载：首次请求时初始化所有组件"""
    global extractor, searcher, classifier, preprocessor, config

    if extractor is not None:
        return

    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.model import CLIPFeatureExtractor
    from src.retrieval import SimilaritySearch
    from src.classifier import ImageClassifier
    from src.preprocessing import PreprocessingPipeline

    extractor = CLIPFeatureExtractor()
    searcher = SimilaritySearch(
        embedding_dim=config["model"]["embedding_dim"],
        index_type=config["retrieval"].get("index_type", "flat"),
    )
    classifier = ImageClassifier(
        model=extractor.model,
        processor=extractor.processor,
        categories=config["classifier"]["categories"],
        device=extractor.device,
    )
    preprocessor = PreprocessingPipeline(config.get("preprocessing", {}))

    searcher.load()


def preprocess_image(image: Image.Image):
    """统一预处理流程"""
    image = preprocessor(image)
    image = image.convert("RGB")
    img_tensor = torch.tensor(np.array(image.resize((224, 224))).transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return image, img_tensor



@app.get("/health")
async def health():
    """健康检查"""
    lazy_init()
    total = searcher.index.ntotal if searcher and searcher.index else 0
    return {
        "status": "ok",
        "index_size": total,
        "model": config["model"]["name"] if config else "unknown",
    }


@app.get("/stats")
async def stats():
    """索引统计信息"""
    lazy_init()
    if not searcher.index or searcher.index.ntotal == 0:
        return {"total": 0, "category_distribution": {}}

    total = searcher.index.ntotal
    from collections import Counter
    cat_counter = Counter()
    for m in searcher.metadata:
        cat_counter[m.get("cat_name", "未知")] += 1

    return {
        "total": total,
        "index_type": searcher.index_type,
        "embedding_dim": searcher.embedding_dim,
        "category_distribution": dict(cat_counter.most_common()),
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """上传一张图片，返回分类结果"""
    lazy_init()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"图片解析失败: {str(e)}")

    try:
        _, img_tensor = preprocess_image(image)
        cat_id, cat_name, scores = classifier.classify(img_tensor)
        is_sign, sign_conf = classifier.is_sign_photo(img_tensor)

        return {
            "filename": file.filename,
            "category_id": cat_id,
            "category_name": cat_name,
            "sign_confidence": round(float(sign_conf), 4),
            "is_sign_photo": bool(is_sign),
            "all_scores": {k: round(float(v), 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
        }
    except Exception as e:
        raise HTTPException(500, f"分类失败: {str(e)}")


@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = None):
    """上传图片，分类后检索相似影像

    Args:
        file: 上传的图片
        top_k: 返回最相似结果数（默认 config.app.top_k）
    """
    lazy_init()

    if top_k is None:
        top_k = config["app"]["top_k"]

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"图片解析失败: {str(e)}")

    try:
        _, img_tensor = preprocess_image(image)

        # 分类
        cat_id, cat_name, scores = classifier.classify(img_tensor)
        is_sign, sign_conf = classifier.is_sign_photo(img_tensor)

        # 特征提取
        with torch.no_grad():
            feat = extractor.extract(img_tensor)
        feat_np = feat.cpu().numpy().astype(np.float32)
        feat_norm = np.linalg.norm(feat_np)
        if feat_norm > 0:
            feat_np = feat_np / feat_norm

        # 检索
        results = searcher.search(feat_np[0], top_k=top_k)

        return {
            "filename": file.filename,
            "category_id": cat_id,
            "category_name": cat_name,
            "sign_confidence": round(float(sign_conf), 4),
            "is_sign_photo": bool(is_sign),
            "all_scores": {k: round(float(v), 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
            "similar_results": [
                {
                    "rank": i + 1,
                    "similarity": round(float(r["score"]), 4),
                    "is_suspicious": r["score"] >= config["retrieval"]["similarity_threshold"],
                    "biz_id": r["metadata"].get("biz_id", ""),
                    "category": r["metadata"].get("cat_name", ""),
                    "path": r["metadata"].get("path", ""),
                }
                for i, r in enumerate(results)
            ],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"检索失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="启动 API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    print(f"启动 API 服务: http://{args.host}:{args.port}")
    print(f"接口文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
