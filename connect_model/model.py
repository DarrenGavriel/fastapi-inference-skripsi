import re
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
        model_path = "./Qwen3.5-27B-AWQ-W4A16"  # Path ke folder AWQ kamu
        max_seq_length = 8192

        # 1. Load Tokenizer (otomatis baca chat_template.jinja bawaan Qwen)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 2. Load Model dengan vLLM (pengganti FastLanguageModel)
        model = LLM(
            model=model_path,
            quantization="compressed-tensors",  # Format AWQ dari llm-compressor
            dtype="bfloat16",  # Gunakan bfloat16 (atau "float16")
            max_model_len=max_seq_length,
            max_num_seqs=2,
            # gpu_memory_utilization=0.95,
            kv_cache_memory_bytes = 1.5 * 1024 * 1024 * 1024,  # Batasi cache KV ke ~1.5GB
            kv_cache_dtype="fp8",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            limit_mm_per_prompt={
                "image": 0
            },  # Hemat VRAM karena fokus teks, bukan gambar
            trust_remote_code=True,
        )

        return True, model, tokenizer
    except Exception as e:
        message = f"Terjadi kesalahan saat memuat model dan tokenizer: {e}"
        print(message)
        return False, None, None

def generate_answer(model, tokenizer, query, RAG):
    try:
        tokenizer.chat_template = "{% set enable_thinking = false %}\n" + tokenizer.chat_template
        system_prompt = """Anda adalah asisten virtual customer service untuk perpajakan Indonesia yang mutlak hanya berkomunikasi dalam bahasa Indonesia, ahli dalam Undang-Undang Ketentuan Umum dan Tata Cara Perpajakan (UU KUP) dan aplikasi CoreTax.

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

        ATURAN GLOBAL MUTLAK:
        - JAWAB MUTLAK HANYA DALAM BAHASA INDONESIA. DILARANG KERAS menggunakan bahasa Mandarin (Zhongwen), Inggris, atau bahasa lainnya dalam skenario apa pun.
        - DILARANG menampilkan langkah evaluasi, alasan internal, atau frasa "berdasarkan context yang diberikan" kepada pengguna."""

        pertanyaan_saat_ini = query.strip()
        user_content = f"""<Konteks Dokumen>
        {RAG}
        </Konteks Dokumen>

        <Pertanyaan Saat Ini>
        {pertanyaan_saat_ini}
        </Pertanyaan Saat Ini>

        Hasil:"""
        print(f"System Prompt:\n{system_prompt}\n")
        print(repr(system_prompt))
        print(repr(user_content))
        print(f"User Content:\n{user_content}\n")
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
        think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
        logit_bias_dict = {tid: -100.0 for tid in think_token_ids}

        # Pengganti parameter di dalam model.generate()
        sampling_params = SamplingParams(
            max_tokens=5000,  # Pengganti max_new_tokens
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
            stop_token_ids=[tokenizer.eos_token_id],  # Pengganti eos_token_id
            bad_words=["<think>"],  # KUNCI 1: Larang string "<think>"
            logit_bias=logit_bias_dict,  # KUNCI 2: Beri penalti -100 pada token ID "<think>"
        )

        # Jalankan inferensi (use_cache=True sudah otomatis aktif di PagedAttention vLLM)
        outputs = model.generate([formatted_prompt], sampling_params, use_tqdm=False)

        # Ambil teks hasil generate saja (di vLLM, prompt input tidak ikut ter-output lagi!)
        generated_text = outputs[0].outputs[0].text
        return True, generated_text.strip(), repr(system_prompt)
    except Exception as e:
        message = f"Terjadi kesalahan saat merumuskan ulang kueri: {e}"
        print(message)
        return False, message
    
def generate_rag_query_rewrite(model, tokenizer, chat_history, current_query, answer_history):
    try:
        tokenizer.chat_template = "{% set enable_thinking = false %}\n" + tokenizer.chat_template
        system_prompt = """Anda adalah sistem AI ahli dalam merumuskan ulang kueri pencarian dokumen hukum dan perpajakan berbahasa Indonesia. 
        Tugas Anda: Ubah "Pertanyaan Saat Ini" menjadi SATU kalimat pertanyaan yang baku, formal, jelas, dan mandiri (standalone) yang optimal untuk sistem pencarian (semantic search).

        Langkah Evaluasi (Kerjakan dalam hati):
        1. Cek apakah "Pertanyaan Saat Ini" sudah mandiri atau membutuhkan konteks (mengandung kata penunjuk atau implikasi kelanjutan seperti "terus", "kalau itu", "dendanya", dll).
        2. Jika MANDIRI: Perbaiki tata bahasanya menjadi baku dan formal.
        3. EKSTRAKSI ENTITAS (WAJIB): Pastikan hasil rumusan ulang TETAP mempertahankan terminologi teknis, nama dokumen (misal: SKPKB, STP, SPT), jenis pajak (misal: PPh Badan, PPN), dan pemicu kejadian (misal: audit, pemeriksaan, pembetulan) dari pertanyaan asli. Jangan membuang kata kunci spesifik ini demi membuat kalimat yang lebih ringkas.
        4. Jika BUTUH KONTEKS: Analisis bagian <Riwayat Percakapan>. Cari riwayat yang paling relevan secara substansi. Pastikan kata ambigu merujuk pada topik utama yang logis.
        5. GABUNGKAN: Gunakan informasi dari riwayat yang relevan untuk melengkapi subjek, objek, dan konteks pada pertanyaan saat ini beserta entitas penting dari Langkah 3.

        Aturan Output:
        - JAWAB MUTLAK HANYA DALAM BAHASA INDONESIA. DILARANG KERAS menggunakan bahasa Mandarin (Zhongwen), Inggris, atau bahasa lainnya.
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
        print(f"System Prompt:\n{system_prompt}\n")
        print(repr(system_prompt))
        print(repr(user_content))
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
        think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
        logit_bias_dict = {tid: -100.0 for tid in think_token_ids}

        # Pengganti parameter di dalam model.generate()
        sampling_params = SamplingParams(
            max_tokens=5000,  # Pengganti max_new_tokens
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
            stop_token_ids=[tokenizer.eos_token_id],  # Pengganti eos_token_id
            bad_words=["<think>"],  # KUNCI 1: Larang string "<think>"
            logit_bias=logit_bias_dict,  # KUNCI 2: Beri penalti -100 pada token ID "<think>"
        )

        # Jalankan inferensi (use_cache=True sudah otomatis aktif di PagedAttention vLLM)
        outputs = model.generate([formatted_prompt], sampling_params, use_tqdm=False)

        # Ambil teks hasil generate saja (di vLLM, prompt input tidak ikut ter-output lagi!)
        generated_text = outputs[0].outputs[0].text
        return True, generated_text.strip(), repr(system_prompt)
    except Exception as e:
        message = f"Terjadi kesalahan saat merumuskan ulang kueri: {e}"
        print(message)
        return False, message
def generate_plain(model, tokenizer, systemQuery, query):
    try:
        print(f'System Prompt (repr): """{repr(systemQuery)}"""')
        print(f'User Content (repr): """{repr(query)}"""')
        messages = [
            {"role": "system", "content": systemQuery},
            {"role": "user", "content": query}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False, # Set False agar mengembalikan string, bukan angka token
            add_generation_prompt = True, # Otomatis menambahkan <|im_start|>assistant di akhir
            enable_thinking = False,
        )
        think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
        logit_bias_dict = {tid: -100.0 for tid in think_token_ids}

        # Pengganti parameter di dalam model.generate()
        sampling_params = SamplingParams(
            max_tokens=5000,  # Pengganti max_new_tokens
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
            stop_token_ids=[tokenizer.eos_token_id],  # Pengganti eos_token_id
            bad_words=["<think>"],  # KUNCI 1: Larang string "<think>"
            logit_bias=logit_bias_dict,  # KUNCI 2: Beri penalti -100 pada token ID "<think>"
        )

        # Jalankan inferensi (use_cache=True sudah otomatis aktif di PagedAttention vLLM)
        outputs = model.generate([formatted_prompt], sampling_params, use_tqdm=False)

        # Ambil teks hasil generate saja (di vLLM, prompt input tidak ikut ter-output lagi!)
        generated_text = outputs[0].outputs[0].text
        return generated_text.strip()
    except Exception as e:
        message = f"Terjadi kesalahan saat merumuskan ulang kueri: {e}"
        print(message)
        return {}