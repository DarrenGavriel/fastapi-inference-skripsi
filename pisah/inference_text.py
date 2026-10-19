# ini digunakan buat memanggil 2 API sekaligus, satu untuk rewrite query dan satu untuk answer query. Jadi bisa dipanggil dari 1 endpoint aja.
import requests
import os
from calc.hitung import hitung_sanksi_bunga, hitung_denda_spt, hitung_ppn, hitung_njop, hitung_pbb, hitung_pph_orang_pribadi, hitung_pph_badan

url_rag = "candle-spokesman-nimble.ngrok-free.dev"
url_model = "plating-ambition-mammal.ngrok-free.dev"
turn = 1

query_history = []
answer_history = []

global_answer_history = []
global_query_history = []
global_query_rewrite_history = []
global_rewrite_history = []
global_tps_history = []
global_rag_time = []

headers = {"Content-Type": "application/json"}

# Hubungkan flag dari router dengan fungsi hitung dan gunakan kwargs hasil
# extractor secara langsung. Nilai kwargs untuk pph_op sudah disesuaikan oleh
# calc.extractor.to_function_kwargs.
CALCULATION_MAPPING = {
    "sanksi_bunga": hitung_sanksi_bunga,
    "denda_spt": hitung_denda_spt,
    "ppn": hitung_ppn,
    "NJOP": hitung_njop,
    "PBB": hitung_pbb,
    "pph_op": hitung_pph_orang_pribadi,
    "pph_badan": hitung_pph_badan,
}

def calculate_from_extraction(flag, kwargs):
    calculator = CALCULATION_MAPPING.get(flag)
    if calculator is None:
        raise ValueError(f"Flag perhitungan tidak didukung: {flag}")
    if not isinstance(kwargs, dict):
        raise ValueError(f"Kwargs extractor tidak valid untuk flag: {flag}")
    return calculator(**kwargs)

print("=== Program POST Berjalan ===")
print("Membaca input dari dataset 'file.txt'...\n")

if not os.path.exists("./persiapan_awq/test/pertanyaan.txt"):
    print("File './persiapan_awq/test/pertanyaan.txt' tidak ditemukan. Pastikan file dataset sudah ada.")
else:
    with open("./persiapan_awq/test/pertanyaan.txt", "r", encoding="utf-8") as f:
        baris_query = f.readlines()

    for line in baris_query:
        query = line.strip()
        if not query:
            continue
            
        print(f"Masukkan query: {query}")
        
        if query.lower() == "keluar":
            print("Program dihentikan.")
            break
        elif query.lower() == "reset":
            query_history.clear()
            answer_history.clear()
            turn = 1
            print("Riwayat percakapan telah dihapus.\n")
            continue
            
        # Mengirim permintaan POST ke endpoint rewrite
        if turn == 1:
            data_rag = {"query": query}
            response_route = requests.post(f"https://{url_rag}/route", json=data_rag, headers=headers)
            response_route.raise_for_status()
            body_route = response_route.json()
            if body_route.get("status") == True:
                result_route = body_route.get("message")
            else:
                print(f"Route Error: {body_route.get('message')}\n")
                continue
            if result_route != "tanpa_hitung":
                print(f"Route Result: {result_route}\n")
                data_extract = {"query": query, "flag": result_route}
                response_extract = requests.post(f"https://{url_model}/extraction", json=data_extract, headers=headers)
                response_extract.raise_for_status()
                body_extract = response_extract.json()
                if body_extract["status"] == True:
                    extraction_info = body_extract["message"]
                    print(f"Extract Result: {extraction_info}\n")
                    if extraction_info["ok"] == True:
                        print(f"Extract Result: {extraction_info['kwargs']}\n")
                        calculation_result = calculate_from_extraction(
                            result_route, extraction_info["kwargs"]
                        )
                        data_answer_calc = {"query": query, "flag": result_route, "ekstraksi": extraction_info, "nilai": calculation_result}
                        print(f"Calculation Result: {data_answer_calc}\n")
                        response_answer_calc = requests.post(f"https://{url_model}/generate/calculation", json=data_answer_calc, headers=headers)
                        response_answer_calc.raise_for_status()
                        body_answer_calc = response_answer_calc.json()
                        if body_answer_calc["status"] == True:
                            result_answer_calc = body_answer_calc["message"]
                            query_history.append(query)
                            answer_history.append(result_answer_calc)
                            turn += 1
                            print(f"Calculation Result: {result_answer_calc}\n")
                        else:
                            print(f"Calculation Error: {body_answer_calc['message']}\n")
                    else:
                        questions = extraction_info.get("questions", [])
                        message = "data belum lengkap. Mohon jawab:"
                        print("Data belum lengkap. Mohon jawab:")
                        for index, question in enumerate(questions, start=1):
                            message += f"\n{index}. {question}"
                            print(f"{index}. {question}")
                        query_history.append(query)
                        answer_history.append(message)
                        turn += 1
                        continue
                continue
            
            response_rag = requests.post(f"https://{url_rag}/query", json=data_rag, headers=headers)
            response_rag.raise_for_status()
            body_rag = response_rag.json()
            result_rag = body_rag["message"]
            if body_rag["status"] == True:
                print(f"RAG Result: {result_rag}\n")
                data_answer = {"query": query, "RAG": result_rag}
                response_answer = requests.post(f"https://{url_model}/generate/answer", json=data_answer, headers=headers)
                print(f"Data Answer: {response_answer}\n")
                try:
                    response_answer.raise_for_status()
                except requests.exceptions.HTTPError as err:
                    print(f"Status Code: {response_answer.status_code}")
                    print(f"Pesan Error dari API: {response_answer.text}") # Ini akan menampilkan detail errornya
                    raise err
                body_answer = response_answer.json()
                result_answer = body_answer["message"]
                result_query = body_answer["query"] # Menggunakan get() untuk menghindari KeyError jika 'query' tidak ada
                result_tps = body_asnwer["tps"]
                # Menggunakan get() untuk menghindari KeyError jika 'query' tidak ada
                result_query = body_answer.get("query", query) 
                
                if body_answer["status"] == True:
                    query_history.append(query)
                    answer_history.append(result_answer)
                    turn += 1
                    print(f"Answer Result: {result_answer}\n")
                    global_answer_history.append(result_answer)
                    global_query_history.append(result_query)
                    global_tps_history.append(result_tps)
                else:
                    print(f"Answer Error: {result_answer}\n")
            else:
                print(f"RAG Error: {result_rag}\n")
        else:
            data_rewrite = {"query": query, "history_query": query_history, "history_answer": answer_history}
            response_rewrite = requests.post(f"https://{url_model}/rewrite", json=data_rewrite, headers=headers)
            response_rewrite.raise_for_status()
            body_rewrite = response_rewrite.json()
            if body_rewrite.get("status") is True:
                result_rewrite = body_rewrite.get("message")
                result_rag_time = body_rewrite.get("")
                print(f"Rewrite Result: {result_rewrite}\n")
                data_rag = {"query": result_rewrite}
                response_route = requests.post(f"https://{url_rag}/route", json=data_rag, headers=headers)
                response_route.raise_for_status()
                body_route = response_route.json()
                if body_route.get("status") is not True:
                    print(f"Route Error: {body_route.get('message')}\n")
                    continue
                result_route = body_route.get("message")
                
                # Ambil query rewrite dari body_rewrite dengan get()
                result_query_rewite = body_rewrite.get("query", query)
                global_query_rewrite_history.append(result_query_rewite)
                global_rewrite_history.append(result_rewrite)
                global_rag_time.append(result_rag_time)
                
                if result_route != "tanpa_hitung":
                    print(f"Route Result: {result_route}\n")
                    data_extract = {"query": result_rewrite, "flag": result_route}
                    response_extract = requests.post(f"https://{url_model}/extraction", json=data_extract, headers=headers)
                    response_extract.raise_for_status()
                    body_extract = response_extract.json()
                    if body_extract.get("status") is True:
                        extraction_info = body_extract["message"]
                        if extraction_info.get("ok") is True:
                            print(f"Extract Result: {extraction_info['kwargs']}\n")
                            calculation_result = calculate_from_extraction(
                                result_route, extraction_info["kwargs"]
                            )
                            print(f"Calculation Result: {calculation_result}\n")
                            data_answer_calc = {"query": result_rewrite, "flag": result_route, "ekstraksi": extraction_info, "nilai": calculation_result}
                            response_answer_calc = requests.post(f"https://{url_model}/generate/calculation", json=data_answer_calc, headers=headers)
                            response_answer_calc.raise_for_status()
                            body_answer_calc = response_answer_calc.json()
                            if body_answer_calc["status"] == True:
                                result_answer_calc = body_answer_calc["message"]
                                query_history.append(query)
                                answer_history.append(result_answer_calc)
                                turn += 1
                                print(f"Calculation Result: {result_answer_calc}\n")
                            else:
                                print(f"Calculation Error: {body_answer_calc['message']}\n")
                        else:
                            questions = extraction_info.get("questions", [])
                            message = "data belum lengkap. Mohon jawab:"
                            print("Data belum lengkap. Mohon jawab:")
                            for index, question in enumerate(questions, start=1):
                                message += f"\n{index}. {question}"
                                print(f"{index}. {question}")
                            query_history.append(query)
                            answer_history.append(message)
                            turn += 1
                            continue
                    else:
                        print(f"Extract Error: {body_extract.get('message')}\n")
                    continue
                else:
                    response_rag = requests.post(f"https://{url_rag}/query", json=data_rag, headers=headers)
                    response_rag.raise_for_status()
                    body_rag = response_rag.json()
                    result_rag = body_rag.get("message")
                    result_rag_time = body_rag.get("rag_time")
                    if body_rag.get("status") is True:
                        print(f"RAG Result: {result_rag}\n")
                        data_answer = {"query": result_rewrite, "RAG": result_rag}
                        response_answer = requests.post(f"https://{url_model}/generate/answer", json=data_answer, headers=headers)
                        response_answer.raise_for_status()
                        body_answer = response_answer.json()
                        result_answer = body_answer.get("message")
                        result_tps = body_answer.get("tps")

                        result_query = body_answer.get("query", result_rewrite)
                        
                        if body_answer.get("status") is True:
                            global_query_history.append(result_query)
                            global_answer_history.append(result_answer)
                            global_tps_history.append(result_tps)
                            
                            query_history.append(result_rewrite)
                            answer_history.append(result_answer)
                            turn += 1
                            print(f"Answer Result: {result_answer}\n")
                        else:
                            print(f"Answer Error: {result_answer}\n")
                    else:
                        print(f"RAG Error: {result_rag}\n")
            else:
                print(f"Rewrite Error: {body_rewrite.get('message')}\n")

    # =========================================================
    # LOGIKA PENULISAN FILE SETELAH LOOPING DATASET SELESAI
    # =========================================================
    print("Looping dataset selesai. Menyimpan hasil ke dalam file txt...")

    with open("global_query_history.txt", "w", encoding="utf-8") as f:
        for item in global_query_history:
            # Ganti baris baru di dalam string (jika ada) agar format tiap elemen rapi
            safe_item = str(item).replace('\n', ' ')
            f.write(f"{safe_item}\n")

    with open("global_answer_history.txt", "w", encoding="utf-8") as f:
        for item in global_answer_history:
            safe_item = str(item).replace('\n', ' ')
            f.write(f"{safe_item}\n")

    with open("global_query_rewrite_history.txt", "w", encoding="utf-8") as f:
        for item in global_query_rewrite_history:
            safe_item = str(item).replace('\n', ' ')
            f.write(f"{safe_item}\n")

    with open("global_rewrite_history.txt", "w", encoding="utf-8") as f:
        for item in global_rewrite_history:
            safe_item = str(item).replace('\n', ' ')
            f.write(f"{safe_item}\n")

    with open("global_tps_history.txt", "w", encoding="utf-8") as f:
        for item in global_tps_history:
            safe_item = str(item).replace('\n', ' ')
            f.write(f"{safe_item}\n")

    with open("global_rag_time.txt", "w", encoding="utf-8") as f:
        for item in global_rag_time:
            safe_item = str(item).replace('\n', ' ')
            f.write(f"{safe_item}\n")

    print("keenam file berhasil dibuat:")
    print("1. global_query_history.txt")
    print("2. global_answer_history.txt")
    print("3. global_query_rewrite_history.txt")
    print("4. global_rewrite_history.txt")
    print("5. global_tps_history.txt")
    print("6. global_rag_time.txt")