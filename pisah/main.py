import os
import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import hybrid_rag
import uvicorn
import nest_asyncio

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
    turn: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def process_query(query: queryRequest):
    resultRag = hybrid_rag.main(query.query)
    if resultRag[1] == False:
        return {"status": False, "message": resultRag[0]}
    return {"status": True, "message": resultRag[0]}


if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)