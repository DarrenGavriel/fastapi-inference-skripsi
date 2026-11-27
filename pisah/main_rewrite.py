import os
import time

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
import pynvml
import threading

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
    "peak_vram_bytes": 0,
    "monitoring": False,
    "monitor_thread": None,
}

def _pantau_vram(gpu_index: int = 0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    while benchmark_state["monitoring"]:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.used > benchmark_state["peak_vram_bytes"]:
            benchmark_state["peak_vram_bytes"] = info.used
        time.sleep(0.1)  # Cek setiap 100 ms (beban CPU nyaris 0%)
    pynvml.nvmlShutdown()

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
    status, message, user_content, generated_token = model_module.generate_rag_query_rewrite(model[1], model[2], query.history_query, query.query, query.history_answer)
    server_time = time.perf_counter() - t0
    tps = generated_token / server_time if server_time > 0 else 0
    if status == False:
        return {"status": False, "message": message, "query": "", "server_time": server_time, "tps": tps}
    return {"status": True, "message": message, "query": user_content, "server_time": server_time, "tps": tps}

@app.post("/generate/answer")
def process_answer(query: answerRequest):
    if not model[0]:
        return {"status": False, "message": "Gagal memuat model dan tokenizer."}
    if query.query.strip() == "":
        return {"status": False, "message": "Query tidak boleh kosong."}
    if query.RAG.strip() == "":
        return {"status": False, "message": "RAG tidak boleh kosong."}
    t0 = time.perf_counter()
    status, message, user_content, generated_token = model_module.generate_answer(model[1], model[2], query.query, query.RAG)
    server_time = time.perf_counter() - t0
    tps = generated_token / server_time if server_time > 0 else 0
    if status == False:
        return {"status": False, "message": message, "query": "", "server_time": server_time, "tps": tps}
    return {"status": True, "message": message, "query": user_content, "tps": tps, "server_time": server_time}

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
    extraction_info, raw_output, generated_token = extract(query.flag, query.query, generate)
    server_time = time.perf_counter() - t0
    tps = generated_token / server_time if server_time > 0 else 0
    return {"status": True, "message": extraction_info, "raw_output": raw_output, "server_time": server_time, "tps": tps}

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
    generated_token = calculation_info.get("generated_token", 0)
    server_time = time.perf_counter() - t0
    tps = generated_token / server_time if server_time > 0 else 0
    if alasan == "":
        return {"status": True, "message": jawaban, "server_time": server_time, "tps": tps}
    else:
        return {"status": False, "message": jawaban, "alasan": alasan, "server_time": server_time, "tps": tps}

@app.post("/benchmark/start")
def start_benchmark(project_name: str = "eval_llm", gpu_index: int = 0):
    benchmark_state["peak_vram_bytes"] = 0
    benchmark_state["monitoring"] = True

    # Jalankan thread pemantau VRAM
    t = threading.Thread(target=_pantau_vram, args=(gpu_index,), daemon=True)
    t.start()
    benchmark_state["monitor_thread"] = t

    tracker = EmissionsTracker(
        project_name=project_name,
        force_carbon_intensity_g_co2e_kwh=778.0,
        output_methods=[],
        log_level="error",
    )
    tracker.start()
    benchmark_state["tracker"] = tracker
    return {"status": True, "message": "Benchmark, CodeCarbon, & VRAM monitor dimulai"}

@app.post("/benchmark/stop")
def stop_benchmark():
    # Hentikan pemantau VRAM
    benchmark_state["monitoring"] = False
    if benchmark_state["monitor_thread"]:
        benchmark_state["monitor_thread"].join(timeout=1.0)

    tracker = benchmark_state.get("tracker")
    total_kwh = 0.0
    total_co2_kg = 0.0
    if tracker:
        tracker.stop()
        data = tracker.final_emissions_data
        total_kwh = data.energy_consumed
        total_co2_kg = data.emissions
        benchmark_state["tracker"] = None

    peak_vram_gib = benchmark_state["peak_vram_bytes"] / (1024 ** 3)

    return {
        "status": True,
        "model_load_time_sec": benchmark_state["model_load_time"],
        "peak_vram_gb": round(peak_vram_gib, 3),
        "total_joule": total_kwh * 3_600_000,
        "total_co2_grams": total_co2_kg * 1000,
    }
if __name__ == "__main__":
    t_load = time.perf_counter()
    model = model_module.load_model_and_tokenizer()
    benchmark_state["model_load_time"] = time.perf_counter() - t_load
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)