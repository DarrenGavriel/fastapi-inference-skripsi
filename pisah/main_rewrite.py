import os

from requests import Request
import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import connect_model.model as model_module
import uvicorn
import nest_asyncio

listener = None

async def connect_ngrok():
    global listener
    listener = await ngrok.connect(8000, authtoken_from_env=True)
    print(f"🌍 API kamu sudah online di: {listener.url()}")

model_lora = model_module.load_model_and_tokenizer_lora()
model = model_module.load_model_and_tokenizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_ngrok()
    yield 
    if listener:
        await listener.close()

app = FastAPI(lifespan=lifespan)

class queryRequest(BaseModel):
    query: str
    history_query: list
    history_answer: list

class answerRequest(BaseModel):
    query: str
    RAG: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/rewrite")
def process_query(query: queryRequest):
    if not model[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    rewrite_query = model_module.generate_rag_query_rewrite(model[1], model[2], query.history_query, query.query, query.history_answer)
    if rewrite_query[0] == False:
        return {"status": False, "message": rewrite_query[1]}
    return {"status": True, "message": rewrite_query[1]}

@app.post("/answer")
def process_answer(query: answerRequest):
    if not model_lora[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    if query.RAG.strip() == "":
        return {"status": False, "message": "RAG tidak boleh kosong."}
    answer_query = model_module.generate_answer(model[1], model[2], query.query, query.RAG)
    if answer_query[0] == False:
        return {"status": False, "message": answer_query[1]}
    return {"status": True, "message": answer_query[1]}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)