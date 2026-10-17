"""
AWQ (W4A16) untuk Qwen/Qwen3.5-27B pakai llm-compressor (oneshot).

Install dulu (Qwen3.5 butuh versi baru):
    pip install -U transformers
    pip install -U llmcompressor   # atau install dari source kalau versinya < 0.11

Yang WAJIB kamu sesuaikan: bagian CONFIG dan fungsi build_text().
"""

import json
import random

from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# ----------------------------- CONFIG -----------------------------
MODEL_ID = "Qwen/Qwen3.5-27B"
SAVE_DIR = "Qwen3.5-27B-AWQ-W4A16"

QUERY_FILE = "queries.jsonl"      # satu baris = {"query": "..."}; ganti sesuai format datasetmu
NUM_CALIB_SAMPLES = 256           # 256-512 umum dipakai
MAX_SEQ_LEN = 2048                # samakan dengan panjang prompt RAG aslimu (query + top-5 chunk)
SEED = 42

# Prompt sistem harus sama dengan yang dipakai saat inferensi,
# karena skala AWQ disetel mengikuti distribusi data kalibrasi.
SYSTEM_PROMPT = "ISI_DENGAN_SYSTEM_PROMPT_ASISTENMU"
# -------------------------------------------------------------------


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["query"] for r in rows]


def build_text(tokenizer, query, context=""):
    """Bentuk teks kalibrasi semirip mungkin dengan prompt produksi.
    Kalau bisa, isi `context` dengan chunk hasil retrieval asli (top-5),
    jangan cuma query mentah."""
    user_content = f"Dokumen:\n{context}\n\nPertanyaan: {query}" if context else query
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    random.seed(SEED)

    # 1) Load model (Qwen3.5 itu multimodal, jadi pakai kelas ImageTextToText)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2) Siapkan dataset kalibrasi dari query domain pajakmu
    queries = load_queries(QUERY_FILE)
    random.shuffle(queries)
    queries = queries[:NUM_CALIB_SAMPLES]

    ds = Dataset.from_dict({"text": [build_text(tokenizer, q) for q in queries]})
    ds = ds.map(
        lambda ex: tokenizer(
            ex["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,  # chat template sudah memasukkan special token
        ),
        remove_columns=ds.column_names,
    )

    # 3) Recipe AWQ
    recipe = [
        AWQModifier(
            targets=["Linear"],
            scheme="W4A16_ASYM",  # alternatif: "W4A16" (simetris)
            ignore=[
                "lm_head",
                "re:.*visual.*",       # vision tower
                "re:.*linear_attn.*",  # layer GatedDeltaNet tetap BF16
                "re:.*mtp.*",          # head multi-token prediction
            ],
        )
    ]

    # 4) Jalankan kalibrasi + kuantisasi
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=len(queries),
        # Kalau muncul error saat sequential tracing, coba isi manual dengan
        # nama kelas decoder layer Qwen3.5, mis. sequential_targets=["<NamaDecoderLayer>"]
    )

    # 5) Simpan (format compressed-tensors, bisa langsung di-serve vLLM)
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True).save_pretrained(SAVE_DIR)

    # Bawa MTP head dari model asli (tidak ikut terkuantisasi)
    save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)

    print(f"Selesai. Tersimpan di: {SAVE_DIR}")


if __name__ == "__main__":
    main()