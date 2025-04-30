import os
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import time
from pathlib import Path

app = FastAPI(title="Product Search API")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
COLLECTION_NAME = "products"
model = SentenceTransformer('all-MiniLM-L6-v2')

client = QdrantClient("localhost", port=6333)

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    search_time_ms: float

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(STATIC_DIR / "index.html") as f:
        return f.read()

@app.get("/admin.html", response_class=HTMLResponse)
async def admin_ui():
    with open(STATIC_DIR / "admin.html") as f:
        return f.read()

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="search query in natural language"),
    limit: int = Query(10, description="maximum number of results to return"),
    offset: int = Query(0, description="number of results to skip")
):
    start_time = time.time()
    query_vector = model.encode(query).tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        offset=offset
    )
    
    results = []
    for scored_point in search_result:
        item = scored_point.payload
        results.append({
            "score": scored_point.score,
            "item": item,
        })
    
    search_time_ms = (time.time() - start_time) * 1000
    
    return SearchResponse(
        results=results,
        total=len(results),
        search_time_ms=search_time_ms
    )

@app.get("/health")
async def health_check():
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            collection_info = client.get_collection(COLLECTION_NAME)
            count_result = client.count(collection_name=COLLECTION_NAME)
            count = count_result.count if hasattr(count_result, 'count') else None
            detailed_info = vars(collection_info) if hasattr(collection_info, '__dict__') else collection_info
            
            return {
                "status": "ok",
                "vector_db": "connected",
                "collection": COLLECTION_NAME,
                "points_count": count,
                "collection_info": detailed_info
            }
        else:
            return {
                "status": "warning",
                "vector_db": "connected",
                "message": f"collection '{COLLECTION_NAME}' not found"
            }
    except Exception as e:
        return {
            "status": "error",
            "vector_db": "disconnected",
            "error": str(e)
        }

@app.get("/admin")
async def admin_panel():
    collections_info = []
    
    try:
        collections = client.get_collections()
        for collection in collections.collections:
            coll_name = collection.name
            try:
                count_result = client.count(collection_name=coll_name)
                count = count_result.count if hasattr(count_result, 'count') else 0
                collections_info.append({
                    "name": coll_name,
                    "count": count
                })
            except Exception as e:
                collections_info.append({
                    "name": coll_name,
                    "count": "Error",
                    "error": str(e)
                })
    except Exception as e:
        return {"error": str(e)}
    
    return {
        "collections": collections_info
    }

@app.get("/collection/{collection_name}/sample")
async def get_collection_sample(collection_name: str, limit: int = 10):
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if collection_name not in collection_names:
            return {"error": f"Collection '{collection_name}' not found"}
        
        points = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]
        
        return {
            "collection": collection_name,
            "sample_count": len(points),
            "points": [
                {
                    "id": point.id,
                    "payload": point.payload
                }
                for point in points
            ]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.search_api:app", host="0.0.0.0", port=8000, reload=True) 