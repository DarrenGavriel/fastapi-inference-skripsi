from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import re
import torch

def load_model_and_tokenizer():
    try:
        max_seq_length = 4056
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "Qwen/Qwen3.5-9B",
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "chatml",
        )
        return True, model, tokenizer
    except Exception as e:
        message = f"Terjadi kesalahan saat memuat model dan tokenizer: {e}"
        print(message)
        return False, None, None

def generate_rag_query_rewrite(model, tokenizer, chat_history, current_query, answer_history):
    try:
        tokenizer.chat_template = "{% set enable_thinking = false %}\n" + tokenizer.chat_template
        system_prompt = """Anda adalah sistem AI ahli dalam merumuskan ulang kueri pencarian dokumen.
        Tugas Anda: Ubah "Pertanyaan Saat Ini" menjadi SATU kalimat pertanyaan yang baku, formal, jelas, dan mandiri (standalone).

        Langkah Evaluasi (Kerjakan dalam hati):
        1. Cek apakah "Pertanyaan Saat Ini" sudah mandiri atau membutuhkan konteks (mengandung kata penunjuk atau implikasi kelanjutan seperti "terus", "kalau itu", "dendanya", dll).
        2. Jika MANDIRI: Cukup perbaiki tata bahasanya menjadi baku dan formal. Abaikan riwayat.
        3. Jika BUTUH KONTEKS: Analisis bagian <Riwayat Percakapan>.
        4. FILTERING & RESOLUSI AMBIGUITAS: Cari riwayat yang paling relevan secara substansi dengan topik "Pertanyaan Saat Ini". Abaikan riwayat yang tidak berhubungan (Out of Topic). Pastikan kata ambigu seperti "telat" merujuk pada topik utama yang logis (pajak/pelaporan SPT, bukan kantin).
        5. GABUNGKAN: Gunakan informasi dari riwayat yang relevan untuk melengkapi subjek, objek, dan konteks pada pertanyaan saat ini.

        Aturan Output:
        - Jawab LANGSUNG dengan hasil akhir berupa satu kalimat pertanyaan baku.
        - DILARANG memberikan kata pengantar, penjelasan, tanda kutip ekstra, atau analisis.
        - DILARANG menggunakan bahasa santai/gaul."""

        if not chat_history or not answer_history:
            formatted_history = "Tidak ada riwayat percakapan sebelumnya."
        else:
            history_lines = []
            for i, (user_msg, ast_msg) in enumerate(zip(chat_history, answer_history)):
                user_number = (i * 2) + 1
                assistant_number = (i * 2) + 2
                
                history_lines.append(f"{user_number}. User: {user_msg}")
                history_lines.append(f"{assistant_number}. Assistant: {ast_msg}")
            formatted_history = "\n".join(history_lines)
        pertanyaan_saat_ini = current_query.strip()
        user_content = f"""<Riwayat Percakapan>
        {formatted_history}
        </Riwayat Percakapan>

        <Pertanyaan Saat Ini>
        {pertanyaan_saat_ini}
        </Pertanyaan Saat Ini>

        Hasil:"""

        # 4. Susun Struktur Pesan Menggunakan Format Standar Chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False, # Set False agar mengembalikan string, bukan angka token
            add_generation_prompt = True, # Otomatis menambahkan <|im_start|>assistant di akhir
            enable_thinking = False,
        )
        inputs = tokenizer(
            text=[formatted_prompt],
            return_tensors="pt",
            add_special_tokens=False
        ).to("cuda")

        # 3. Generate Jawaban
        banned_token_ids = tokenizer(text="<think>", add_special_tokens=False)["input_ids"]
        outputs = model.generate(
            input_ids = inputs["input_ids"],
            max_new_tokens = 200,
            use_cache = True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            # Setting Instruct/Non-thinking
            temperature = 0.7,
            top_p = 0.8,
            repetition_penalty = 1.1,

            # KUNCI UTAMA: Larang model mengeluarkan tag <think>
            bad_words_ids = banned_token_ids
        )
        # 5. Tampilkan Hasil
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        # 6. Bersihkan tag <think> dan ambil hasil akhirnya saja
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)

        # Bersihkan juga sisa spasi kosong atau tag sistem lainnya jika ada
        clean_response = clean_response.replace('<|im_end|>', '').strip()
        print(clean_response)
        return True, clean_response
    except Exception as e:
        message = f"Terjadi kesalahan saat merumuskan ulang kueri: {e}"
        print(message)
        return False, message