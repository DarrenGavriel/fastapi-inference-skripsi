from fastapi import FastAPI
from pydantic import BaseModel
import hybrid_rag

app = FastAPI()

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
