# KALAU MAU FILL ULANG CHROMA DB PAKE INI SAYANG LANGSUNG JSONL SEMUANYA MASUK
# PLUS METADATA YANG LIST ITU PAKAI > yaaa

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Membaca seluruh dokumen dari file JSONL hasil preprocessing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File {filepath} tidak ditemukan!")

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_text_content(doc: Dict[str, Any]) -> str:
    """
    Mencari field yang berisi teks dokumen secara dinamis.
    Mendukung 'page_content', 'document', 'text', 'content', dll.
    """
    for key in ["page_content", "document", "text", "content"]:
        if key in doc and doc[key]:
            return str(doc[key])
    return ""

def clean_metadata(raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Membersihkan metadata secara dinamis agar sesuai dengan syarat ChromaDB.
    ChromaDB hanya menerima str, int, float, atau bool.
    """
    cleaned = {}
    for k, v in raw_metadata.items():
        # ChromaDB tidak menerima null/None, jadi kita lewati field ini
        if v is None:
            continue
        
        # Jika berupa list, gabungkan dengan separator
        elif isinstance(v, list):
            cleaned[k] = " > ".join(map(str, v))
            
        # Jika berupa dictionary/object bersarang, ubah jadi string JSON
        elif isinstance(v, dict):
            cleaned[k] = json.dumps(v, ensure_ascii=False)
            
        # Tipe data dasar yang diterima ChromaDB (str, int, float, bool)
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
            
        # Tipe data lainnya diubah paksa ke string
        else:
            cleaned[k] = str(v)
            
    return cleaned

def validate_document(doc: Dict[str, Any], index: int) -> Tuple[bool, str, str]:
    """
    Melakukan validasi eksistensi id, teks konten (dinamis), dan metadata.
    Mengembalikan (is_valid, warning_msg, text_content).
    """
    doc_id = doc.get("id")
    metadata = doc.get("metadata")
    text_content = get_text_content(doc)

    if not doc_id:
        return False, f"Baris {index}: 'id' kosong atau tidak ditemukan.", ""
    
    if not text_content.strip():
        return False, f"Baris {index} (ID: {doc_id}): Field teks (page_content/document) kosong.", ""
        
    if not metadata or not isinstance(metadata, dict):
        return False, f"Baris {index} (ID: {doc_id}): 'metadata' kosong atau format salah.", ""

    return True, "", text_content

def build_chroma_index(
    jsonl_path: str,
    persist_directory: str,
    collection_name: str,
    model_name: str,
    cache_folder: str
):
    print("\n==================================================")
    print("🔄 MEMULAI PROSES INDEXING CHROMADB")
    print("==================================================")
    
    start_time = time.time()

    # 1. Load JSONL
    print(f"📂 Membaca file: {jsonl_path}...")
    raw_data = load_jsonl(jsonl_path)
    print(f"📊 Ditemukan {len(raw_data)} dokumen di dalam JSONL.")

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    failed_count = 0

    print("⚙️ Memvalidasi data dan menyusun struktur...")
    for idx, item in enumerate(raw_data, start=1):
        is_valid, warning_msg, text_content = validate_document(item, idx)
        
        if not is_valid:
            print(f"   ⚠️ WARNING: {warning_msg} -> Dilewati.")
            failed_count += 1
            continue

        doc_id = item["id"]
        # Panggil fungsi clean_metadata untuk menangani list, dict, dan null secara dinamis
        metadata_cleaned = clean_metadata(item.get("metadata", {}))

        ids.append(doc_id)
        documents.append(text_content)
        metadatas.append(metadata_cleaned)

    success_count = len(ids)
    print(f"   ✅ Dokumen valid: {success_count} | ❌ Gagal/Dilewati: {failed_count}")

    if success_count == 0:
        print("❌ Tidak ada dokumen yang valid. Program dihentikan.")
        return

    # 2. Proses Embedding
    print(f"\n🧠 Memuat Model Embedding: {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    
    print("[Mulai Embeddings] Menghitung vektor embeddings...")
    embeddings_array = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=4
    )
    embeddings = embeddings_array.tolist()

    # 3. Proses Insert ke ChromaDB
    print(f"\n💾 Menghubungkan ke ChromaDB di: {persist_directory}")
    client = chromadb.PersistentClient(path=persist_directory)
    
    collection = client.get_or_create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"📦 Melakukan upsert data ke collection: {collection_name}...")
    
    batch_size_db = 1000
    for i in range(0, len(ids), batch_size_db):
        end_idx = min(i + batch_size_db, len(ids))
        
        collection.upsert(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            embeddings=embeddings[i:end_idx]
        )
        print(f"   ✔️ Batch tersimpan: {end_idx} / {len(ids)}")

    elapsed_time = time.time() - start_time
    print("==================================================")
    print("🎉 INDEXING CHROMADB SELESAI 🎉")
    print(f"✅ Total Embedding Sukses : {success_count}")
    print(f"⏱️ Waktu Proses           : {elapsed_time:.2f} detik")
    print("==================================================")

if __name__ == "__main__":
    JSONL_SOURCE = "../processed_documents.jsonl"
    CHROMA_PERSIST_DIR = "../chroma_db"
    CHROMA_COLLECTION = "Combine_tax_knowledge"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    CACHE_FOLDER = "D:/RAG_Cache/models" 
    
    try:
        build_chroma_index(
            jsonl_path=JSONL_SOURCE,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION,
            model_name=EMBEDDING_MODEL,
            cache_folder=CACHE_FOLDER
        )
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat eksekusi: {e}")