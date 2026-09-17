import json
import re
from typing import List, Dict, Any, Tuple, Optional

# =====================================================================
# 1. Import Class & Helper
# =====================================================================
from create_bm25.build_BM25 import TaxBM25Retriever
from create_chromadb.semantic_search import (
    load_embedding_model, 
    connect_chromadb, 
    embed_query, 
    calculate_similarity
)

# ==================================================
# 2. KONFIGURASI REGEX (IGNORE CASE)
# ==================================================
RE_PASAL = re.compile(r'pasal\s+([0-9]+[a-zA-Z]*)\b', re.IGNORECASE)
RE_AYAT = re.compile(r'ayat\s+([0-9]+[a-zA-Z]*)\b', re.IGNORECASE)

print("[*] Menyiapkan Semantic Search...")
semantic_model = load_embedding_model()
chroma_collection = connect_chromadb()

print("[*] Memuat index BM25 (harap tunggu)...")
bm25_retriever = TaxBM25Retriever()

try:
    bm25_retriever.load_index("./my_bm25_index") 
except FileNotFoundError:
    print("❌ Error: Folder index BM25 tidak ditemukan. Sesuaikan path-nya.")


# ==================================================
# 3. FUNGSI HYBRID RETRIEVAL LOKAL (DARI SKRIP 1)
# ==================================================
def extract_metadata(query: str) -> Tuple[Optional[str], Optional[str]]:
    pasal_match = RE_PASAL.search(query)
    ayat_match = RE_AYAT.search(query)
    return (pasal_match.group(1) if pasal_match else None, 
            ayat_match.group(1) if ayat_match else None)

def parse_chroma_results(chroma_res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    parsed = {}
    if not chroma_res or not chroma_res.get('ids') or not chroma_res['ids'][0]:
        return parsed

    for i in range(len(chroma_res['ids'][0])):
        doc_id = chroma_res['ids'][0][i]
        dist = chroma_res['distances'][0][i]
        meta = chroma_res['metadatas'][0][i]
        doc = chroma_res['documents'][0][i]
        
        parsed[doc_id] = {
            "id": doc_id,
            "similarity": calculate_similarity(dist),
            "metadata": meta,
            "document": doc,
            "is_exact_meta": False # Default penanda bypass threshold
        }
    return parsed

def metadata_search(collection, q_embedding, pasal, ayat, top_k) -> Dict[str, Dict[str, Any]]:
    where_clause = {}
    if pasal and ayat:
        where_clause = {"$and": [{"pasal": pasal}, {"ayat": ayat}]}
    elif pasal:
        where_clause = {"pasal": pasal}

    res = collection.query(
        query_embeddings=[q_embedding], n_results=top_k,
        where=where_clause, include=['documents', 'metadatas', 'distances']
    )
    return parse_chroma_results(res)

def semantic_search_custom(collection, q_embedding, top_k) -> Dict[str, Dict[str, Any]]:
    res = collection.query(
        query_embeddings=[q_embedding], n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return parse_chroma_results(res)

def execute_hybrid_search(query: str, q_embedding: List[float], collection, top_k: int) -> Tuple[List[Dict[str, Any]], str]:
    pasal, ayat = extract_metadata(query)
    sem_results = semantic_search_custom(collection, q_embedding, top_k)
    
    # RULE 1: PASAL + AYAT (PRIORITY MERGE)
    if pasal and ayat:
        meta_results = metadata_search(collection, q_embedding, pasal, ayat, top_k)
        final_docs = []
        
        # 1. Paksa HANYA hasil metadata yang masuk
        sorted_meta = sorted(meta_results.values(), key=lambda x: x['similarity'], reverse=True)
        for d in sorted_meta:
            d['is_exact_meta'] = True
            final_docs.append(d)
            
        # Blok kode "Sisipkan hasil semantic" HAPUS SAJA biar gak bocor ke pasal lain
                
        return final_docs[:top_k], "Rule 1 (Pasal+Ayat: Exact Match Only)"

    # RULE 2: HANYA PASAL (STANDARD MERGE)
    elif pasal and not ayat:
        meta_results = metadata_search(collection, q_embedding, pasal, None, top_k)
        final_docs = []
        
        # Hanya masukkan hasil yang benar-benar match dari metadata
        sorted_meta = sorted(meta_results.values(), key=lambda x: x['similarity'], reverse=True)
        for d in sorted_meta:
            d['is_exact_meta'] = True
            final_docs.append(d)
            
        return final_docs[:top_k], "Rule 2 (Pasal: Exact Match Only)"

    # RULE 3 & 4: HANYA AYAT / TANPA KEDUANYA (SEMANTIC ONLY)
    else:
        sorted_docs = sorted(sem_results.values(), key=lambda x: x['similarity'], reverse=True)
        return sorted_docs[:top_k], "Rule 3/4 (Semantic Only)"


# =====================================================================
# 4. FUNGSI BM25, RRF, & GET SEMANTIC (SESUAIKAN DENGAN SKRIP 2)
# =====================================================================
def get_bm25_results(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    # 1. Ekstrak target pasal dan ayat dari query pengguna
    pasal_target, ayat_target = extract_metadata(query)
    
    # 🐛 DEBUGGING: Cek apakah Regex berhasil nangkap angkanya!
    print(f"    -> [Filter Target] Mencari Pasal: '{pasal_target}', Ayat: '{ayat_target}'")
    
    fetch_size = 50 if pasal_target else top_k 
    raw_results = bm25_retriever.search(query, top_k=fetch_size)
    
    formatted_results = []
    
    for res in raw_results:
        metadata = res.get("metadata", {})
        
        # Bersihkan spasi dan nol di depan (misal " 04 " jadi "4")
        doc_pasal = str(metadata.get("pasal", "")).strip().lstrip("0")
        doc_ayat = str(metadata.get("ayat", "")).strip().lstrip("0")
        
        # 3. LOGIKA HARD FILTER MUTLAK
        if pasal_target and ayat_target:
            pasal_bersih = pasal_target.strip().lstrip("0")
            ayat_bersih = ayat_target.strip().lstrip("0")
            if doc_pasal != pasal_bersih or doc_ayat != ayat_bersih:
                continue # Buang kalau gak sama persis
                
        elif pasal_target and not ayat_target:
            pasal_bersih = pasal_target.strip().lstrip("0")
            if doc_pasal != pasal_bersih:
                continue # Buang kalau pasal gak sama persis
        
        # 4. Jika lolos filter, simpan
        teks = res.get("content", res.get("page_content", res.get("document", "")))
        formatted_item = {
            "id": res.get("chunk_id", res.get("id", "")),
            "score": res.get("score", 0.0), 
            "page_content": teks,
            "metadata": metadata
        }
        formatted_results.append(formatted_item)
        
        if len(formatted_results) >= top_k:
            break
            
    print(f"    -> [BM25] Lolos filter: {len(formatted_results)} dokumen.")
    return formatted_results

def get_semantic_results(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Mengambil hasil Semantic menggunakan Regex/Metadata Rules,
    dan menyesuaikan outputnya untuk digabung ke dalam RRF.
    """
    # 1. Generate embedding
    query_embedding = embed_query(query, semantic_model)
    
    # 2. Panggil fungsi hybrid lokal yang mengintegrasikan Regex (Rule 1-4)
    final_docs, strategy = execute_hybrid_search(query, query_embedding, chroma_collection, top_k)
    print(f"    -> [Semantic Strategy]: {strategy}")
    
    # 3. Format hasil ke bentuk yang kompatibel dengan RRF script
    formatted_results = []
    for doc in final_docs:
        formatted_item = {
            "id": doc["id"],
            "score": doc["similarity"],
            "page_content": doc["document"],
            "metadata": doc["metadata"],
            "is_exact_meta": doc.get("is_exact_meta", False) # Simpan penanda untuk filter rata-rata
        }
        formatted_results.append(formatted_item)
        
    return formatted_results

def rrf_fusion(bm25_results: List[Dict[str, Any]], semantic_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    k = 60
    rrf_scores = {}
    documents = {}

    # 1. Proses hasil BM25 dan simpan skor BM25 aslinya
    for index, doc in enumerate(bm25_results):
        rank = index + 1
        doc_id = doc["id"]
        score = 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        
        if doc_id not in documents:
            documents[doc_id] = {"page_content": doc["page_content"], "metadata": doc["metadata"]}
        # Simpan skor BM25
        documents[doc_id]["bm25_score"] = doc.get("score", 0.0)
            
    # 2. Proses hasil Semantic dan simpan skor Similarity aslinya
    for index, doc in enumerate(semantic_results):
        rank = index + 1
        doc_id = doc["id"]
        score = 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        
        if doc_id not in documents:
            documents[doc_id] = {"page_content": doc["page_content"], "metadata": doc["metadata"]}
        # Simpan skor Semantic (Similarity)
        documents[doc_id]["semantic_similarity"] = doc.get("score", 0.0)

    sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    # 3. Susun hasil akhir beserta skor aslinya
    final_results = []
    for doc_id, rrf_score in sorted_rrf[:top_k]:
        doc_info = documents[doc_id]
        final_doc = {
            "id": doc_id,
            "rrf_score": round(rrf_score, 5),
            "semantic_similarity": doc_info.get("semantic_similarity", None), # <--- Nilai Similarity ditambahkan
            "bm25_score": doc_info.get("bm25_score", None),                   # <--- Nilai BM25 ditambahkan (sebagai info tambahan)
            "page_content": doc_info["page_content"],
            "metadata": doc_info["metadata"]
        }
        final_results.append(final_doc)

    return final_results


# =====================================================================
# 5. MAIN LOOP
# =====================================================================
def main(question: str = None):
    print("="*50)
    print("      PENCARIAN GABUNGAN (BM25 + REGEX HYBRID SEMANTIC)      ")
    print("="*50)
    top_k_search = 5
    status = True
    if question:
        query = question.strip()
        # ---------------------------------
        # BM25 SEARCH
        # ---------------------------------
        print("[*] Menjalankan BM25 Search...")
        bm25_results = get_bm25_results(query, 10)
        if bm25_results:
            total_bm25_score = sum(doc['score'] for doc in bm25_results)
            rata_rata_bm25 = total_bm25_score / len(bm25_results)
            print(f"    -> Rata-rata Skor BM25: {rata_rata_bm25:.2f}")
            # Menggunakan threshold dari kodemu
            if rata_rata_bm25 < 6.1:
                print("    -> ⚠️ Rata-rata di bawah 6.1, hasil BM25 tidak digunakan.")
                bm25_results = []  
        # ---------------------------------
        # SEMANTIC (REGEX HYBRID) SEARCH
        # ---------------------------------
        print("\n[*] Menjalankan Semantic (Regex) Search...")
        semantic_results = get_semantic_results(query, 10)
        if semantic_results:
            has_exact_meta = any(doc.get("is_exact_meta", False) for doc in semantic_results)
            total_semantic_score = sum(doc['score'] for doc in semantic_results)
            rata_rata_semantic = total_semantic_score / len(semantic_results)
            print(f"    -> Rata-rata Skor Semantic: {rata_rata_semantic * 100:.2f}%")
            # Menggunakan threshold dari kodemu
            if rata_rata_semantic < 0.50 and not has_exact_meta:
                print("    -> ⚠️ Rata-rata di bawah 50%, hasil Semantic tidak digunakan.")
                semantic_results = []  
                status = False
            else:
                print("    -> ✅ Hasil Semantic diterima untuk RRF.")
        # ---------------------------------
        # RRF FUSION
        # ---------------------------------
        if not semantic_results and not bm25_results:
            print("\n❌ Pencarian dibatalkan: Rata-rata skor dari kedua metode di bawah standar minimum.")
            # Output default jika tidak ada hasil
            fallback_msg = 'Terima kasih atas pertanyaannya. Mohon maaf, saya hanya dapat membantu menjawab pertanyaan seputar perpajakan Indonesia, khususnya terkait Undang-Undang KUP dan penggunaan aplikasi CoreTax. Untuk pertanyaan di luar topik tersebut, saya belum bisa membantu.'
            # print(fallback_msg)
            return fallback_msg, status
        print("\n[*] Melakukan Reciprocal Rank Fusion (RRF)...")
        hybrid_results = rrf_fusion(bm25_results, semantic_results, top_k=top_k_search)
        print("\n" + "-"*50)
        print("HASIL AKHIR UNTUK DATASET LORA")
        print("-"*50)
        # 3. Kumpulkan semua chunk menjadi SATU BARIS dengan literal '\n'
        chunk_outputs = []
        for i, doc in enumerate(hybrid_results, start=1):
            # Ubah enter asli di dalam page_content menjadi teks literal '\n' (aman untuk JSON)
            content_safe = doc['page_content'].replace('\n', '\\n').replace('\r', '')
            chunk_outputs.append(f"Chunk {i}\\n{content_safe}")
        # Gabungkan antar chunk dengan literal '\n' juga
        final_output_string = "\\n".join(chunk_outputs)
        # Return hasil akhir
        return final_output_string, status
        # print(final_output_string)
    else:
        while True:
            try:
                print("\n" + "-"*50)
                query = input("Masukkan pertanyaan (atau ketik 'quit' untuk keluar): ").strip()
                if query.lower() in ["exit", "quit", "keluar"]:
                    print("Program dihentikan. Sampai jumpa!")
                    break
                if not query:
                    continue
                print(f"\n[*] MEMPROSES PERTANYAAN: '{query}'")
                # ---------------------------------
                # BM25 SEARCH
                # ---------------------------------
                print("[*] Menjalankan BM25 Search...")
                bm25_results = get_bm25_results(query, 10)
                if bm25_results:
                    total_bm25_score = sum(doc['score'] for doc in bm25_results)
                    rata_rata_bm25 = total_bm25_score / len(bm25_results)
                    print(f"    -> Rata-rata Skor BM25: {rata_rata_bm25:.2f}")
                    # Menggunakan threshold dari kodemu
                    if rata_rata_bm25 < 6.1:
                        print("    -> ⚠️ Rata-rata di bawah 6.1, hasil BM25 tidak digunakan.")
                        bm25_results = []  
                # ---------------------------------
                # SEMANTIC (REGEX HYBRID) SEARCH
                # ---------------------------------
                print("\n[*] Menjalankan Semantic (Regex) Search...")
                semantic_results = get_semantic_results(query, 10)
                if semantic_results:
                    has_exact_meta = any(doc.get("is_exact_meta", False) for doc in semantic_results)
                    total_semantic_score = sum(doc['score'] for doc in semantic_results)
                    rata_rata_semantic = total_semantic_score / len(semantic_results)
                    print(f"    -> Rata-rata Skor Semantic: {rata_rata_semantic * 100:.2f}%")
                    # Menggunakan threshold dari kodemu
                    if rata_rata_semantic < 0.60 and not has_exact_meta:
                        print("    -> ⚠️ Rata-rata di bawah 60%, hasil Semantic tidak digunakan.")
                        semantic_results = []  
                    else:
                        print("    -> ✅ Hasil Semantic diterima untuk RRF.")
                # ---------------------------------
                # RRF FUSION
                # ---------------------------------
                if not semantic_results and not bm25_results:
                    print("\n❌ Pencarian dibatalkan: Rata-rata skor dari kedua metode di bawah standar minimum.")
                    # Output default jika tidak ada hasil
                    fallback_msg = 'Terima kasih atas pertanyaannya. Mohon maaf, saya hanya dapat membantu menjawab pertanyaan seputar perpajakan Indonesia, khususnya terkait Undang-Undang KUP dan penggunaan aplikasi CoreTax. Untuk pertanyaan di luar topik tersebut, saya belum bisa membantu.'
                    print(fallback_msg)
                    # Jika kamu punya fungsi save_output_to_txt(fallback_msg), bisa di-uncomment di bawah ini:
                    # save_output_to_txt(fallback_msg)
                    continue
                print("\n[*] Melakukan Reciprocal Rank Fusion (RRF)...")
                hybrid_results = rrf_fusion(bm25_results, semantic_results, top_k=top_k_search)
                print("\n" + "-"*50)
                print("HASIL AKHIR UNTUK DATASET LORA")
                print("-"*50)
                # 3. Kumpulkan semua chunk menjadi SATU BARIS dengan literal '\n'
                chunk_outputs = []
                for i, doc in enumerate(hybrid_results, start=1):
                    # Ubah enter asli di dalam page_content menjadi teks literal '\n' (aman untuk JSON)
                    content_safe = doc['page_content'].replace('\n', '\\n').replace('\r', '')
                    chunk_outputs.append(f"Chunk {i}\\n{content_safe}")
                # Gabungkan antar chunk dengan literal '\n' juga
                final_output_string = "\\n".join(chunk_outputs)
                # Print hasil akhir
                print(final_output_string)
                # Jika kamu punya fungsi save_output_to_txt, uncomment baris di bawah ini:
                # save_output_to_txt(final_output_string)
            except KeyboardInterrupt:
                print("\nProgram dihentikan oleh pengguna. Sampai jumpa!")
                break
            except Exception as e:
                print(f"\n❌ Terjadi kesalahan: {e}")
                continue

if __name__ == "__main__":
    main(question=None)