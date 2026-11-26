import os
from time import time

from requests import Request
import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import connect_model.model as model_module
from calc.extractor import extract
from calc.responder import buat_jawaban
import uvicorn
import nest_asyncio
import torch
from codecarbon import EmissionsTracker

listener = None

async def connect_ngrok():
    global listener
    listener = await ngrok.connect(8000, authtoken_from_env=True)
    print(f"🌍 API kamu sudah online di: {listener.url()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_ngrok()
    yield 
    if listener:
        await listener.close()

app = FastAPI(lifespan=lifespan)
benchmark_state = {
    "tracker": None,
    "model_load_time": 0.0,
}
class queryRequest(BaseModel):
    query: str
    history_query: list
    history_answer: list

class answerRequest(BaseModel):
    query: str
    RAG: str

class calculationRequest(BaseModel):
    query: str
    flag: str
    ekstraksi: dict
    nilai: float

class extractionRequest(BaseModel):
    query: str
    flag: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/rewrite")
def process_query(query: queryRequest):
    if not model[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    t0 = time.perf_counter()
    rewrite_query = model_module.generate_rag_query_rewrite(model[1], model[2], query.history_query, query.query, query.history_answer)
    server_time = time.perf_counter() - t0
    if rewrite_query[0] == False:
        return {"status": False, "message": rewrite_query[1], "query": "", "server_time": server_time}
    return {"status": True, "message": rewrite_query[1], "query": rewrite_query[2], "tps": rewrite_query[3], "server_time": server_time}

@app.post("/generate/answer")
def process_answer(query: answerRequest):
    if not model[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    if query.RAG.strip() == "":
        return {"status": False, "message": "RAG tidak boleh kosong."}
    t0 = time.perf_counter()
    answer_query = model_module.generate_answer(model[1], model[2], query.query, query.RAG)
    server_time = time.perf_counter() - t0
    if answer_query[0] == False:
        return {"status": False, "message": answer_query[1], "query": "", "server_time": server_time}
    return {"status": True, "message": answer_query[1], "query": answer_query[2], "tps": answer_query[3], "server_time": server_time}

@app.post("/extraction")
def process_extraction(query: extractionRequest):
    if not model[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    if query.flag.strip() == "":
        return {"status": False, "message": "Flag tidak boleh kosong."}
    def generate(system_prompt, user_text):
        return model_module.generate_plain(
            model[1],
            model[2],
            system_prompt,
            user_text,
        )
    t0 = time.perf_counter()
    extraction_info, raw_output = extract(query.flag, query.query, generate)
    server_time = time.perf_counter() - t0
    return {"status": True, "message": extraction_info, "raw_output": raw_output, "server_time": server_time}

@app.post("/generate/calculation")
def process_calculation(query: calculationRequest):
    if not model[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    if query.flag.strip() == "":
        return {"status": False, "message": "Flag tidak boleh kosong."}
    t0 = time.perf_counter()
    def generate(system_prompt, user_text):
        return model_module.generate_plain(
            model[1],
            model[2],
            system_prompt,
            user_text,
        )

    calculation_info = buat_jawaban(
        flag=query.flag, 
        query=query.query, 
        ekstraksi=query.ekstraksi, 
        nilai=query.nilai, 
        generate_jawaban=generate
    )
    
    alasan = calculation_info.get("alasan", "").strip()
    jawaban = calculation_info.get("jawaban", "")
    server_time = time.perf_counter() - t0
    if alasan == "":
        return {"status": True, "message": jawaban, "server_time": server_time}
    else:
        return {"status": False, "message": jawaban, "alasan": alasan, "server_time": server_time}

@app.post("/benchmark/start")
def start_benchmark(project_name: str = "eval_llm"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    tracker = EmissionsTracker(
        project_name=project_name,
        force_carbon_intensity_g_co2e_kwh=778.0,
        output_methods=[],
        log_level="error",
    )
    tracker.start()
    benchmark_state["tracker"] = tracker
    return {"status": True, "message": "Benchmark & CodeCarbon dimulai"}

@app.post("/benchmark/stop")
def stop_benchmark():
    tracker = benchmark_state.get("tracker")
    total_kwh = 0.0
    total_co2_kg = 0.0

    if tracker:
        tracker.stop()
        data = tracker.final_emissions_data
        total_kwh = data.energy_consumed
        total_co2_kg = data.emissions
        benchmark_state["tracker"] = None

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "status": True,
        "model_load_time_sec": benchmark_state["model_load_time"],
        "peak_vram_gb": peak_vram_gb,
        "total_joule": total_kwh * 3_600_000,       # Konversi kWh ke Joule
        "total_co2_grams": total_co2_kg * 1000,     # Konversi kg ke gram CO2e
    }
if __name__ == "__main__":
    model = model_module.load_model_and_tokenizer()
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)