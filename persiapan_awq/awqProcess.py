"""
AWQ (W4A16) untuk Qwen/Qwen3.5-27B pakai llm-compressor (oneshot).
Dilengkapi Stratified Sampling berdasarkan 'flag' agar semua jenis task terwakili merata.
"""

from collections import defaultdict
import json
import random

from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

import tqdm

_orig_tqdm_init = tqdm.tqdm.__init__


def _quiet_grid_search_tqdm(self, *args, **kwargs):
  desc = kwargs.get("desc", "") or ""
  # Sembunyikan progress bar jika ia adalah proses "Grid search" atau "Smoothing"
  if "Grid search" in desc or "Smoothing" in desc:
    kwargs["disable"] = True
  # Paksa lebar bar supaya tidak melimpah ke baris baharu di terminal
  kwargs.setdefault("dynamic_ncols", True)
  _orig_tqdm_init(self, *args, **kwargs)


tqdm.tqdm.__init__ = _quiet_grid_search_tqdm

# ----------------------------- CONFIG -----------------------------
MODEL_ID = "Qwen/Qwen3.5-27B"
SAVE_DIR = "Qwen3.5-27B-AWQ-W4A16"

CALIB_FILE = "./merged_data.jsonl"  # Ganti dengan path file JSONL kamu
NUM_CALIB_SAMPLES = 256  # Total sampel kalibrasi (dibagi rata ke 10 flag)

# CATATAN: Jika 26680 bikin OOM saat proses AWQ, turunkan ke 4096 atau 2048.
# Menurunkan angka ini saat AWQ TIDAK akan mengurangi batas konteks 26680 saat inferensi nanti.
MAX_SEQ_LEN = 4096
SEED = 42
# -------------------------------------------------------------------


def load_stratified_calib_data(path, total_samples, seed=42):
    """Membaca JSONL dan mengambil sampel secara merata (stratified)

    berdasarkan key 'flag'.
    """
    rng = random.Random(seed)
    grouped = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
        row = json.loads(line)
        flag = row.get("flag", "unknown")
        grouped[flag].append({
            "content": row.get("content", ""),
            "system_prompt": row.get("system_prompt", ""),
            "flag": flag,
        })

    flags = list(grouped.keys())
    num_flags = len(flags)
    if num_flags == 0:
        raise ValueError("Dataset JSONL kosong atau tidak valid!")

    # Acak urutan data di dalam masing-masing flag
    for flag in flags:
        rng.shuffle(grouped[flag])

    # Hitung kuota dasar per flag
    base_quota = total_samples // num_flags
    selected = []
    leftovers = []

    print(
        f"\n--- Distribusi Data Kalibrasi per Flag (Total Flag: {num_flags}) ---"
    )
    for flag in flags:
        items = grouped[flag]
        take = min(len(items), base_quota)
        selected.extend(items[:take])
        # Simpan sisa data kalau-kalau butuh tambahan untuk menggenapi total_samples
        leftovers.extend(items[take:])
        print(f"Flag '{flag}': tersedia {len(items)} -> diambil {take} sampel")

    # Jika masih kurang dari total_samples (karena pembulatan bagi atau ada flag yang datanya sedikit)
    shortfall = total_samples - len(selected)
    if shortfall > 0 and leftovers:
        rng.shuffle(leftovers)
        extra = leftovers[:shortfall]
        selected.extend(extra)
        print(
            f"+ Tambahan {len(extra)} sampel dari sisa data agar genap"
            f" {len(selected)} sampel."
        )

    # Acak lagi urutan gabungan akhirnya supaya saat kalibrasi urutan flag-nya menyebar
    rng.shuffle(selected)
    print(f"Total akhir sampel kalibrasi: {len(selected)}\n")
    return selected


def build_text(tokenizer, sample):
    """Membentuk teks ChatML dari system_prompt dan content."""
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["content"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    random.seed(SEED)

    # 1) Load model & tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2) Siapkan dataset kalibrasi dengan Stratified Sampling per flag
    samples = load_stratified_calib_data(CALIB_FILE, NUM_CALIB_SAMPLES, SEED)
    formatted_texts = [build_text(tokenizer, s) for s in samples]

    ds = Dataset.from_dict({"text": formatted_texts})
    ds = ds.map(
        lambda ex: tokenizer(
            ex["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        ),
        remove_columns=ds.column_names,
    )

    # Info diagnostik panjang token aktual di dataset kamu
    lengths = [len(x) for x in ds["input_ids"]]
    print(
        f"Info Token Kalibrasi -> Min: {min(lengths)}, Rata-rata:"
        f" {sum(lengths)//len(lengths)}, Max: {max(lengths)}"
    )

    # 3) Recipe AWQ
    recipe = [
        AWQModifier(
            targets=["Linear"],
            scheme="W4A16_ASYM",
            ignore=[
                "lm_head",
                "re:.*visual.*",  # vision tower
                "re:.*linear_attn.*",  # layer GatedDeltaNet tetap BF16
                "re:.*mtp.*",  # head multi-token prediction
            ],
        )
    ]

    # 4) Jalankan kalibrasi + kuantisasi
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=len(samples),
    )

    # 5) Simpan hasil
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True).save_pretrained(
        SAVE_DIR
    )

    save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
    print(f"Selesai. Tersimpan di: {SAVE_DIR}")


if __name__ == "__main__":
    main()