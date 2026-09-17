import os
import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import hybrid_rag
import uvicorn
import nest_asyncio

def connect_ngrok():
    forwarder = ngrok.forward("localhost:8000", authtoken_from_env=True, domain="plating-ambition-mammal.ngrok-free.dev")
    print(f"🌍 API kamu sudah online di: {forwarder.url()}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_ngrok()
    yield 
    ngrok.disconnect()

app = FastAPI(lifespan=lifespan)

class queryRequest(BaseModel):
    query: str
    turn: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def process_query(query: queryRequest):
    if query.turn == 0:
        resultRag = hybrid_rag.main(query.query)
        if resultRag[1] == False:
            return {"query": query.query, "result": resultRag[0]}
    return {"query": query}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)