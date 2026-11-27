import os
import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from router.train_router import Router
import hybrid_rag
import uvicorn
import nest_asyncio
import time

listener = None

async def connect_ngrok():
    global listener
    listener = await ngrok.connect(8000, authtoken_from_env=True)
    print(f"🌍 API kamu sudah online di: {listener.url()}")

# model = model_module.load_model_and_tokenizer()

query_history = []
query_answer_history = []
query_rewrite_history = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_ngrok()
    yield 
    if listener:
        await listener.close()

app = FastAPI(lifespan=lifespan)

class queryRequest(BaseModel):
    query: str

class AblationRequest(BaseModel):
    query: str
    top_k: int = 5
    use_threshold: bool = True
router = Router("router_out/router.joblib")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def process_query(query: queryRequest):
    t0 = time.perf_counter()
    resultRag = hybrid_rag.main(query.query)
    server_time = time.perf_counter() - t0
    if resultRag[1] == False:
        return {"status": False, "message": resultRag[0], "server_time": server_time}
    return {"status": True, "message": resultRag[0], "server_time": server_time}

@app.post("/query/test")
def process_query(query: queryRequest):
    t0 = time.perf_counter()
    resultRag = hybrid_rag.main(query.query)
    server_time = time.perf_counter() - t0
    if resultRag[1] == False:
        return {"status": False, "message": resultRag[3], "server_time": server_time}
    return {"status": True, "message": resultRag[3], "server_time": server_time}

@app.post("/route")
def process_route(query: queryRequest):
    label, conf = router.route(query.query)
    return {"status": True, "message": label}

@app.post("/ablation/{mode}")
def process_ablation(mode: str, req: AblationRequest):
    if mode not in ["semantic", "bm25", "hybrid"]:
        return {"status": False, "message": "Mode harus 'semantic', 'bm25', atau 'hybrid'"}
    t0 = time.perf_counter()
    top_ids, avg_bm25, avg_semantic = hybrid_rag.retrieve_top_ids_ablation(
        req.query, 
        mode=mode,
        use_threshold=req.use_threshold # Matikan threshold sementara untuk melihat sebaran skor asli
    )
    server_time = time.perf_counter() - t0
    return {
        "status": True, 
        "message": top_ids,
        "avg_bm25": avg_bm25,
        "avg_semantic": avg_semantic,
        "server_time": server_time
    }

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)