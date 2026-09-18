import os
import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import hybrid_rag
import connect_model.model as model_module
import uvicorn
import nest_asyncio

listener = None

async def connect_ngrok():
    global listener
    listener = await ngrok.connect(8000, authtoken_from_env=True, domain="plating-ambition-mammal.ngrok-free.dev")
    print(f"🌍 API kamu sudah online di: {listener.url()}")

model = model_module.load_model_and_tokenizer()

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
    turn: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def process_query(query: queryRequest):
    if not model[0]:
        return {"query": query.query, "result": "Gagal memuat model dan tokenizer."}
    if query.turn == 0:
        resultRag = hybrid_rag.main(query.query)
        if resultRag[1] == False:
            return {"query": query.query, "result": resultRag[0]}
    rewrite_query = model_module.generate_rag_query_rewrite(model[1], model[2], query_rewrite_history, query.query, query_answer_history)
    if rewrite_query[0] == False:
        return {"query": query.query, "result": rewrite_query[1]}
    return {"query": query.query, "result": rewrite_query[1]}


if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)