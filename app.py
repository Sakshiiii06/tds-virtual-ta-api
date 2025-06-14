from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import uvicorn
import logging

# -------------------- Setup --------------------
app = FastAPI(title="TDS Virtual TA")
logging.basicConfig(level=logging.INFO)

# -------------------- Load Model & Index --------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")

# Load metadata (original posts)
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -------------------- Request Model --------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3  # Number of top results to return

# -------------------- Helper Function --------------------
def search_index(query: str, top_k: int = 3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        if idx < len(metadata):
            result = {
                "content": metadata[idx].get("content", ""),
                "title": metadata[idx].get("title", ""),
                "url": metadata[idx].get("url", "")
            }
            results.append(result)
    return results

# -------------------- API Route --------------------
@app.post("/ask")
async def ask_question(request: QueryRequest):
    logging.info(f"Received query: {request.question}")
    results = search_index(request.question, request.top_k)
    return {"question": request.question, "answers": results}

# -------------------- Optional Root --------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the TDS Virtual TA API!"}

# -------------------- Run with Uvicorn --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
