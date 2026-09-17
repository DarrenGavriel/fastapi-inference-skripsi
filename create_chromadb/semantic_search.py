import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from xml.parsers.expat import model

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Error: Library 'chromadb' atau 'sentence-transformers' belum terinstal.")
    print("Jalankan: pip install chromadb sentence-transformers")
    sys.exit(1)

# ==================================================
# KONFIGURASI EVALUASI
# ==================================================
TOP_K = 5
SIMILARITY_THRESHOLD = 0.50
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "Combine_tax_knowledge"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Menambahkan cache directory di Drive D agar tidak memenuhi Drive C Anda
MODEL_CACHE_DIR = "D:/RAG_Cache/models" 

# ==================================================
# FUNGSI-FUNGSI EVALUASI
# ==================================================
def load_embedding_model() -> SentenceTransformer:
    """
    Memuat model embedding (BAAI/bge-m3).
    Menggunakan cache lokal jika tersedia untuk mempercepat proses.
    """
    print(f"[*] Memuat model {EMBEDDING_MODEL} (harap tunggu)...")
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)
    return model

def connect_chromadb() -> chromadb.Collection:
    """
    Membuat koneksi ke PersistentClient ChromaDB dan mengambil collection yang dituju.
    """
    db_path = Path(CHROMA_DB_PATH)
    if not db_path.exists():
        print(f"❌ Error: Folder database '{CHROMA_DB_PATH}' tidak ditemukan.")
        sys.exit(1)
        
    print(f"[*] Menghubungkan ke ChromaDB di '{CHROMA_DB_PATH}'...")
    client = chromadb.PersistentClient(path=str(db_path))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except ValueError:
        print(f"❌ Error: Collection '{COLLECTION_NAME}' tidak ditemukan di database.")
        sys.exit(1)

def embed_query(query: str, model: SentenceTransformer) -> List[float]:
    """
    Mengubah teks query pengguna menjadi vektor embedding.
    """
    # model.encode mengembalikan array NumPy, kita ubah ke list native Python
    embedding = model.encode(query, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    return embedding.tolist()

def calculate_similarity(distance: float) -> float:
    """
    Mengonversi nilai distance dari ChromaDB ke similarity (0.0 - 1.0).
    
    Penjelasan Metrik:
    Secara default, ChromaDB menggunakan metrik L2 (Squared L2) distance.
    Untuk vektor yang dinormalisasi (seperti output umum model embedding), 
    rumus konversi dari Squared L2 ke Cosine Similarity adalah:
    Cosine Similarity = 1 - (L2_Distance)
    """
    # Menghindari nilai di bawah 0 jika terjadi anomali pembulatan
    similarity = 1.0 - (distance)
    return max(0.0, min(1.0, similarity))

def similarity_search(
    query_embedding: List[float], 
    collection: chromadb.Collection
) -> Dict[str, Any]:
    """
    Melakukan pencarian Top-K dokumen paling relevan ke ChromaDB.
    Hanya meminta data dokumen, metadata, dan distances.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=['documents', 'metadatas', 'distances']
    )
    return results

def print_results(
    query: str, 
    results: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Menampilkan hasil retrieval lengkap dengan metadata dan page_content.
    Mengembalikan tuple (passed_count, filtered_count) untuk keperluan summary.
    """
    print("\n==================================================")
    print("QUERY")
    print(query)
    print(f"Embedding Model      : {EMBEDDING_MODEL}")
    print(f"Collection           : {COLLECTION_NAME}")
    print(f"Top-K                : {TOP_K}")
    print(f"Similarity Threshold : {SIMILARITY_THRESHOLD * 100:.0f}%")
    print("==================================================")

    # Mengekstrak list dari response ChromaDB
    # Chroma mengembalikan list of lists karena bisa mencari banyak query sekaligus
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]

    retrieved_count = len(documents)
    passed_count = 0
    filtered_count = 0

    if retrieved_count == 0:
        print("\n⚠ Tidak ada dokumen yang ditemukan di collection ini.")
        return passed_count, filtered_count

    for i in range(retrieved_count):
        rank = i + 1
        content = documents[i]
        meta = metadatas[i]
        distance = distances[i]
        
        # Hitung persentase similarity
        sim_score = calculate_similarity(distance)
        sim_percent = sim_score * 100
        
        # Cek kelolosan threshold
        if sim_score >= SIMILARITY_THRESHOLD:
            status = "✅ Lolos Threshold"
            passed_count += 1
        else:
            status = "❌ Tidak Lolos Threshold"
            filtered_count += 1
        print(f"\n==================================================")
        print(f"Rank {rank}")
        print(f"==================================================")
        print(f"\nSimilarity : {sim_percent:.2f}%")
        print(f"Status     : {status}\n")
        
        print("Metadata")
        print("---------")
        # Format key metadata agar sejajar dan mudah dibaca (padding 14 karakter)
        for key, value in meta.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key:<14} : {value if value is not None else '-'}")
        print("\nContent")
        print("-------")
        print(f"{content}\n")
    return passed_count, filtered_count

def print_summary(
    retrieved_count: int, 
    passed_count: int, 
    filtered_count: int
) -> None:
    """
    Menampilkan ringkasan dari proses retrieval.
    """
    print("==================================================")
    print("Summary")
    print("==================================================")
    print(f"Top-K Requested      : {TOP_K}")
    print(f"Retrieved            : {retrieved_count}")
    print(f"Passed Threshold     : {passed_count}")
    print(f"Filtered Out         : {filtered_count}")
    print("==================================================\n")

def main():
    """
    Fungsi utama program. Akan berjalan dalam loop interaktif 
    agar Anda dapat menguji berbagai query secara berturut-turut.
    """
    print("="*50)
    print("      TOOLS EVALUASI RAG (SIMILARITY SEARCH)      ")
    print("="*50)
    
    # 1. Persiapan Model & Database (dilakukan sekali di awal)
    model = load_embedding_model()
    collection = connect_chromadb()
    
    # 2. Loop interaktif
    while True:
        try:
            print("\n" + "-"*50)
            query = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()
            
            if query.lower() in ['exit', 'quit', 'keluar']:
                print("Program dihentikan. Sampai jumpa!")
                break
                
            if not query:
                continue

            # 3. Proses Retrieval (DENGAN CHECKPOINT DEBUG)
            print("[Debug] 1. Teks diterima. Memulai proses embedding model...")
            query_embedding = embed_query(query, model)
            
            print("[Debug] 2. Embedding berhasil dibuat! Melakukan pencarian ke ChromaDB...")
            results = similarity_search(query_embedding, collection)
            
            print("[Debug] 3. Data ditemukan! Menyiapkan hasil untuk ditampilkan...")
            
            # 4. Tampilkan Hasil
            passed, filtered = print_results(query, results)
            
            # Hitung retrieved dari panjang data asli
            retrieved = len(results.get('documents', [[]])[0])
            
            # 5. Tampilkan Summary
            print_summary(retrieved, passed, filtered)
            
        except KeyboardInterrupt:
            # Menangani jika user menekan Ctrl+C
            print("\nProgram dihentikan oleh pengguna. Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Terjadi kesalahan saat melakukan pencarian: {e}")

if __name__ == "__main__":
    main()