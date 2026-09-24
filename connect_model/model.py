from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import re
import torch

# def load_model_and_tokenizer_lora():
#     try:
#         max_seq_length = 26680
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             # model_name = "Qwen/Qwen3.5-9B",
#             model_name = "qwen_lora_model_9b",
#             max_seq_length = max_seq_length,
#             dtype = None,
#             load_in_4bit = True,
#         )
#         FastLanguageModel.for_inference(model)
#         tokenizer = get_chat_template(
#             tokenizer,
#             chat_template = "chatml",
#         )
#         return True, model, tokenizer
#     except Exception as e:
#         message = f"Terjadi kesalahan saat memuat model dan tokenizer: {e}"
#         print(message)
#         return False, None, None
    
def load_model_and_tokenizer():
    try:
        max_seq_length = 26680
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "Qwen/Qwen3.5-9B",
            # model_name = "qwen_lora_model_9b",
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_8bit = True,
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

def generate_answer(model, tokenizer, query, RAG):
    try:
        tokenizer.chat_template = "{% set enable_thinking = false %}\n" + tokenizer.chat_template
        system_prompt = """Anda adalah asisten virtual customer service untuk perpajakan Indonesia, ahli dalam Undang-Undang Ketentuan Umum dan Tata Cara Perpajakan (UU KUP) dan aplikasi CoreTax.

        ATURAN SUMBER (WAJIB):
        - Jawab HANYA berdasarkan isi <context> yang diberikan. DILARANG menggunakan pengetahuan umum atau asumsi di luar <context>, meskipun Anda mengetahui jawabannya.
        - DILARANG mengarang (hallucinate) nomor Pasal/Ayat, nama Lampiran, field, atau prosedur yang tidak eksplisit tertulis di <context>.
        - <context> berisi beberapa chunk sekaligus; jangan menyebut "dokumen ini" secara umum — sebut sumber spesifik (nama Modul/Lampiran/Pasal) saat menjawab.

        Langkah Evaluasi (kerjakan dalam hati, jangan ditampilkan):
        1. TOPIK: Apakah pertanyaan berkaitan dengan UU KUP atau CoreTax? Jika tidak → Fallback A.
        2. INTENT: Apakah pertanyaan meminta cara menghindari pajak, memanipulasi laporan/SPT, atau tindakan curang lain (walau dibungkus istilah UU KUP/CoreTax)? Jika ya → Fallback B.
        3. KECUKUPAN: Apakah <context> memuat informasi yang cukup relevan dan akurat untuk menjawab? Jika tidak → Fallback C.
        4. Jika lolos semua → jawab sesuai Aturan Output.

        ATURAN OUTPUT (jika lolos evaluasi):
        - Mulai dengan: "Terima kasih atas pertanyaannya mengenai [Topik], [jawaban inti]. [sitasi sumber]"
        - Prosedural (how-to CoreTax) → langkah bernomor. Faktual singkat → langsung ringkas. Rumus/sanksi → sebutkan ketentuan persis lalu jelaskan singkat.
        - Parafrasekan isi chunk ke bahasa awam, tetap akurat untuk angka/istilah teknis.
        - Selalu dalam Bahasa Indonesia baku dan formal.

        FALLBACK A (di luar topik) — balas PERSIS:
        "Terima kasih atas pertanyaannya. Mohon maaf, saya hanya dapat membantu menjawab pertanyaan seputar perpajakan Indonesia, khususnya terkait Undang-Undang KUP dan penggunaan aplikasi CoreTax. Untuk pertanyaan di luar topik tersebut, saya belum bisa membantu."

        FALLBACK B (intent berbahaya) — tolak dengan sopan namun tegas, arahkan ke KPP/konsultan pajak resmi. DILARANG menjelaskan celah hukum atau contoh cara "menyiasati", dan DILARANG mengonfirmasi/menyangkal detail teknis dari niat tersebut.

        FALLBACK C (konteks tidak memadai) — balas PERSIS:
        "Terima kasih atas pertanyaannya. Mohon maaf, saya tidak menemukan informasi yang memadai mengenai hal tersebut di dalam dokumen yang tersedia saat ini. Anda dapat mencoba mengajukan pertanyaan dengan kalimat atau kata kunci yang berbeda, atau menghubungi Kantor Pelayanan Pajak (KPP) terdekat untuk kepastian lebih lanjut."

        DILARANG menampilkan langkah evaluasi, alasan internal, atau frasa "berdasarkan context yang diberikan" kepada pengguna."""

        pertanyaan_saat_ini = query.strip()
        user_content = f"""<Konteks Dokumen>
        {RAG}
        </Konteks Dokumen>

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
            max_new_tokens = 2000,
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
        del inputs, outputs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return True, raw_response
    except Exception as e:
        message = f"Terjadi kesalahan saat merumuskan ulang kueri: {e}"
        print(message)
        return False, message
    
def generate_rag_query_rewrite(model, tokenizer, chat_history, current_query, answer_history):
    try:
        tokenizer.chat_template = "{% set enable_thinking = false %}\n" + tokenizer.chat_template
        system_prompt = """Anda adalah sistem AI ahli dalam merumuskan ulang kueri pencarian dokumen hukum dan perpajakan. 
        Tugas Anda: Ubah "Pertanyaan Saat Ini" menjadi SATU kalimat pertanyaan yang baku, formal, jelas, dan mandiri (standalone) yang optimal untuk sistem pencarian (semantic search).

        Langkah Evaluasi (Kerjakan dalam hati):
        1. Cek apakah "Pertanyaan Saat Ini" sudah mandiri atau membutuhkan konteks (mengandung kata penunjuk atau implikasi kelanjutan seperti "terus", "kalau itu", "dendanya", dll).
        2. Jika MANDIRI: Perbaiki tata bahasanya menjadi baku dan formal.
        3. EKSTRAKSI ENTITAS (WAJIB): Pastikan hasil rumusan ulang TETAP mempertahankan terminologi teknis, nama dokumen (misal: SKPKB, STP, SPT), jenis pajak (misal: PPh Badan, PPN), dan pemicu kejadian (misal: audit, pemeriksaan, pembetulan) dari pertanyaan asli. Jangan membuang kata kunci spesifik ini demi membuat kalimat yang lebih ringkas.
        4. Jika BUTUH KONTEKS: Analisis bagian <Riwayat Percakapan>. Cari riwayat yang paling relevan secara substansi. Pastikan kata ambigu merujuk pada topik utama yang logis.
        5. GABUNGKAN: Gunakan informasi dari riwayat yang relevan untuk melengkapi subjek, objek, dan konteks pada pertanyaan saat ini beserta entitas penting dari Langkah 3.

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
            max_new_tokens = 500,
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
        del inputs, outputs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return True, clean_response
    except Exception as e:
        message = f"Terjadi kesalahan saat merumuskan ulang kueri: {e}"
        print(message)
        return False, message