# ini digunakan buat memanggil 2 API sekaligus, satu untuk rewrite query dan satu untuk answer query. Jadi bisa dipanggil dari 1 endpoint aja.
import requests
from calc.hitung import hitung_sanksi_bunga, hitung_denda_spt, hitung_ppn, hitung_njop, hitung_pbb, hitung_pph_orang_pribadi, hitung_pph_badan

url_rag = "candle-spokesman-nimble.ngrok-free.dev"
url_model = "plating-ambition-mammal.ngrok-free.dev"
turn = 1

query_history = []
answer_history = []

headers = {"Content-Type": "application/json"}
print("=== Program POST Berjalan ===")
print("Ketik 'keluar' pada judul untuk menghentikan program.\n")

while True:
    query = input("Masukkan query: ")
    if query.lower() == "keluar":
        print("Program dihentikan.")
        break
    # Mengirim permintaan POST ke endpoint rewrite
    if turn == 1:
        data_rag = {"query": query}
        response_route = requests.post(f"http://{url_rag}/route", json=data_rag, headers=headers)
        result_route = response_route.message
        if result_route != "tanpa_hitung":
            print(f"Route Result: {result_route}\n")
            data_extract = {"query": query, "flag": result_route}
            response_extract = requests.post(f"http://{url_model}/extract", json=data_extract, headers=headers)
            response_extract.raise_for_status()
            body_extract = response_extract.json()
            if body_extract["status"] == True:
                result_extract = response_extract.message
                print(f"Extract Result: {result_extract}\n")
                extraction_info = body_extract["message"]
                if extraction_info["ok"] == True:
                    print(f"Extract Result: {extraction_info['kwargs']}\n")
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
                    print(f"Extract Error: {extraction_info['message']}\n")
                    continue
            continue
        response_rag = requests.post(f"http://{url_rag}/query", json=data_rag, headers=headers)
        result_rag = response_rag.message
        if response_rag.status == True:
            print(f"RAG Result: {result_rag}\n")
            data_answer = {"query": query, "RAG": result_rag}
            response_answer = requests.post(f"http://{url_model}/answer", json=data_answer, headers=headers)
            result_answer = response_answer.message
            if response_answer.status == True:
                query_history.append(query)
                answer_history.append(result_answer)
                turn += 1
                print(f"Answer Result: {result_answer}\n")
            else:
                print(f"Answer Error: {result_answer}\n")
        else:
            print(f"RAG Error: {result_rag}\n")
    else:
        data_rewrite = {"query": query, "history_query": query_history, "history_answer": answer_history}
        response_rewrite = requests.post(f"http://{url_model}/rewrite", json=data_rewrite, headers=headers)
        result_rewrite = response_rewrite.message
        if response_rewrite.status == True:
            print(f"Rewrite Result: {result_rewrite}\n")
            data_rag = {"query": result_rewrite}
            response_route = requests.post(f"http://{url_rag}/route", json=data_rag, headers=headers)
            result_route = response_route.message
            if result_route != "tanpa_hitung":
                print(f"Route Result: {result_route}\n")
                data_extract = {"query": query, "flag": result_route}
                response_extract = requests.post(f"http://{url_model}/extract", json=data_extract, headers=headers)
                response_extract.raise_for_status()
                body_extract = response_extract.json()
                if body_extract["status"] == True:
                    result_extract = response_extract.message
                    print(f"Extract Result: {result_extract}\n")
                    extraction_info = body_extract["message"]
                    if extraction_info["ok"] == True:
                        print(f"Extract Result: {extraction_info['kwargs']}\n")
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
                        print(f"Extract Error: {extraction_info['message']}\n")
                        continue
                continue
            else:
                response_rag = requests.post(f"http://{url_rag}/query", json=data_rag, headers=headers)
                result_rag = response_rag.message
                if response_rag.status == True:
                    print(f"RAG Result: {result_rag}\n")
                    data_answer = {"query": result_rewrite, "RAG": result_rag}
                    response_answer = requests.post(f"http://{url_model}/answer", json=data_answer, headers=headers)
                    if response_answer.status == True:
                        result_answer = response_answer.message
                        query_history.append(result_rewrite)
                        answer_history.append(result_answer)
                        turn += 1
                        print(f"Answer Result: {result_answer}\n")
                    else:
                        print(f"Answer Error: {result_answer}\n")
                else:
                    print(f"RAG Error: {result_rag}\n")