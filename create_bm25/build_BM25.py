# python build_BM25.py ./jsonl/processed_documents.jsonl --save-dir my_bm25_index
import json
import re
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from uuid import uuid4

# Pastikan Anda telah menginstal bm25s dan codecarbon: 
# pip install bm25s codecarbon
import bm25s
# from codecarbon import EmissionsTracker

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class TaxBM25Retriever:
    """
    Sistem Retrieval Lexical menggunakan BM25 untuk Dokumen Perpajakan.
    Desain modular untuk mempermudah integrasi dengan Hybrid Retrieval / RRF nantinya.
    """

    def __init__(self):
        self.bm25: bm25s.BM25 = None
        self.corpus: List[Dict[str, Any]] = []
        self.id_mapping: Dict[int, str] = {}

    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Membaca dataset dari file JSONL.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"File dataset tidak ditemukan: {filepath}")

        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    
                    # Hapus embedding agar tidak memakan memori (sesuai format baru)
                    if "embedding" in doc:
                        del doc["embedding"]
                        
                    documents.append(doc)
        
        return documents

    def load_jsonl_files(self, filepaths: List[str]) -> List[Dict[str, Any]]:
        """
        Membaca dan menggabungkan beberapa file JSONL menjadi satu list dokumen.
        Menjamin setiap dokumen memiliki `id` yang unik dengan menambahkan prefix
        filename bila terjadi duplikasi.
        """
        merged: List[Dict[str, Any]] = []
        seen_ids = set()

        for fp in filepaths:
            path = Path(fp)
            if not path.is_file():
                raise FileNotFoundError(f"File dataset tidak ditemukan: {fp}")

            docs = self.load_jsonl(fp)
            for doc in docs:
                # Pastikan doc memiliki id
                orig_id = doc.get("id")
                if not orig_id:
                    new_id = f"gen-{uuid4().hex}"
                    doc["id"] = new_id
                else:
                    new_id = str(orig_id)

                # Jika id sudah ada, tambahkan prefix nama file sampai unik
                if new_id in seen_ids:
                    prefix = path.stem
                    candidate = f"{prefix}::{new_id}"
                    # jika masih tabrakan (sangat jarang), tambahkan uuid
                    if candidate in seen_ids:
                        candidate = f"{candidate}::{uuid4().hex}"
                    new_id = candidate
                    doc["id"] = new_id

                seen_ids.add(new_id)
                merged.append(doc)

        return merged

    def preprocess_for_bm25(self, text: str) -> str:
        """
        Melakukan normalisasi ringan sebelum tokenisasi untuk BM25.
        Teks asli di corpus tidak akan diubah, hanya digunakan untuk indexing.
        """
        # 1. Ubah teks menjadi lowercase
        text = text.lower()

        # 2. Hapus karakter yang tidak berguna (contoh: HTML comment <!-- -->)
        text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
        
        # 3. Rapikan whitespace berlebih (spasi berturut-turut, tab, newline berlebih)
        # Diubah menjadi satu spasi saja karena struktur layout tidak relevan untuk BM25 Indexing
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def custom_tokenize(self, text: str) -> List[str]:
        """
        Tokenizer NLP ringan berbasis Regex (Regular Expression).
        
        ALASAN PEMILIHAN:
        - text.split() terlalu kasar dan akan menempelkan tanda baca pada kata (misal "login." menjadi "login.").
        - Tokenizer ini (r'[a-z0-9\-]+') secara spesifik hanya akan mengambil huruf (a-z), angka (0-9), 
          dan tanda hubung (-). 
        - Sehingga istilah seperti "drag-and-drop", "1", "2", "spt", "pph", dan singkatan tetap utuh.
        - Titik, koma, atau kurung akan otomatis terbuang, namun angkanya/katanya tetap selamat.
        """
        # Ekstrak semua kumpulan huruf, angka, dan tanda hubung
        tokens = re.findall(r'[a-z0-9\-]+', text)
        return tokens

    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Membangun index BM25 dari sekumpulan dokumen dan membuat mapping ID.
        """
        start_time = time.time()
        print(f"[BM25] Memulai proses indexing untuk {len(documents)} dokumen...")

        self.corpus = documents
        
        # Mapping index internal BM25 (integer) ke Chunk ID (string)
        self.id_mapping = {i: doc["id"] for i, doc in enumerate(documents)}

        # Persiapkan data untuk indexing (Preprocessing + Tokenisasi)
        # Tidak menggunakan stemming & stopwords (sesuai instruksi v1)
        tokenized_corpus = []
        
        for i, doc in enumerate(documents):
            # Tampilkan progress setiap 1000 dokumen
            if (i + 1) % 1000 == 0:
                print(f"   [BM25] Progress: {i + 1} / {len(documents)} dokumen diproses...")
                
            # Menggunakan key "document" (format baru) atau fallback ke "page_content" (format lama)
            text_content = doc.get("document", doc.get("page_content", ""))
            clean_text = self.preprocess_for_bm25(text_content)
            tokens = self.custom_tokenize(clean_text)
            tokenized_corpus.append(tokens)

        print("   [BM25] Membangun internal struktur...")
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokenized_corpus)

        elapsed = time.time() - start_time
        print(f"[BM25] Indexing selesai dalam {elapsed:.2f} detik.")

    def save_index(self, save_dir: str):
        """
        Menyimpan index BM25, corpus, dan mapping ID ke dalam folder khusus.
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Simpan struktur BM25 menggunakan bawaan library
        bm25_index_dir = path / "index"
        self.bm25.save(str(bm25_index_dir))

        # 2. Simpan corpus asli untuk pengembalian dokumen saat search
        with open(path / "corpus.json", 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)

        # 3. Simpan id mapping
        with open(path / "id_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(self.id_mapping, f, ensure_ascii=False, indent=2)

        print(f"[BM25] Index beserta mapping berhasil disimpan di folder: {save_dir}")

    def load_index(self, save_dir: str):
        """
        Memuat kembali index BM25, corpus, dan mapping ID dari folder yang sudah ada.
        """
        path = Path(save_dir)
        bm25_index_dir = path / "index"
        corpus_path = path / "corpus.json"
        mapping_path = path / "id_mapping.json"

        if not (bm25_index_dir.exists() and corpus_path.exists() and mapping_path.exists()):
            raise FileNotFoundError(f"Komponen index tidak lengkap di dalam folder {save_dir}")

        # 1. Muat objek BM25
        # parameter load_corpus=False karena kita me-manage corpus sendiri via JSON
        self.bm25 = bm25s.BM25.load(str(bm25_index_dir), load_corpus=False)

        # 2. Muat corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)

        # 3. Muat id mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            # Json key selalu dibaca sebagai string, konversi kembali ke int
            str_mapping = json.load(f)
            self.id_mapping = {int(k): v for k, v in str_mapping.items()}

        print(f"[BM25] Berhasil memuat index BM25. Total dokumen: {len(self.corpus)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Melakukan pencarian BM25, mengembalikan dict terstruktur untuk hybrid/RRF.
        """
        if self.bm25 is None:
            raise ValueError("Index BM25 belum dibangun atau dimuat.")

        start_time = time.time()
        
        # Pastikan query diproses menggunakan aturan normalisasi & tokenizer yang SAMA dengan indexing
        clean_query = self.preprocess_for_bm25(query)
        query_tokens = self.custom_tokenize(clean_query)

        # BM25 retrieve mengharapkan batch query (list of lists), jadi kita bungkus dalam list `[query_tokens]`
        # Mengembalikan array numpy berisi index internal dan score
        indices, scores = self.bm25.retrieve([query_tokens], k=top_k)
        
        results = []
        # Mengambil hasil iterasi dari query pertama/satu-satunya (indeks [0])
        for doc_internal_idx, score in zip(indices[0], scores[0]):
            internal_idx = int(doc_internal_idx)
            
            # Abaikan jika hasil tidak relevan / id kosong
            if internal_idx not in self.id_mapping:
                continue
                
            chunk_id = self.id_mapping[internal_idx]
            original_doc = self.corpus[internal_idx]
            
            # 💡 SOLUSI: Cek 'page_content' dulu. Jika tidak ada (None), ambil 'document'. 
            # Jika keduanya tidak ada, kembalikan string kosong "".
            teks = original_doc.get("page_content")
            if not teks:
                teks = original_doc.get("document", "")
            
            results.append({
                "chunk_id": chunk_id,
                "score": float(score),
                # Gunakan key netral "content" agar seragam saat diproses di tahap hybrid/RRF
                "content": teks,
                "metadata": original_doc.get("metadata", {})
            })

        elapsed = time.time() - start_time
        print(f"[BM25] Pencarian selesai dalam {elapsed:.4f} detik. Ditemukan {len(results)} dokumen.")
        
        return results

# =========================================================
# BLOK EKSEKUSI (Bisa dimatikan saat di-import modul lain)
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build BM25 index from one or more JSONL files (each line is a JSON doc)."
    )
    parser.add_argument(
        'jsonl_files', nargs='*',
        help='Paths to JSONL files to index. If omitted, uses processed_documents.jsonl'
    )
    parser.add_argument('--save-dir', '-s', default='bm25_index', help='Directory to save the BM25 index')

    args = parser.parse_args()
    files = args.jsonl_files or ["./jsonl/processed_documents.jsonl"]

    retriever = TaxBM25Retriever()

    # ---------------------------------------------------------
    # MODE 1: MEMBANGUN INDEX BARU (mendukung banyak file)
    # ---------------------------------------------------------
    try:
        print("\n--- TAHAP 1: MEMBANGUN INDEX ---")
        
        # Inisialisasi CodeCarbon untuk melacak fase Indexing
        # tracker_index = EmissionsTracker(
        #     project_name="BM25_Indexing",
        #     force_carbon_intensity_g_co2e_kwh=778.0, # Menggunakan standar emisi Indonesia (~0.778 kgCO2e/kWh)
        #     output_dir = "CodeCarbon",
        #     output_file = "bm25_index_emissions.csv",
        #     log_level="error" # Mengurangi log spam dari codecarbon
        # )
        
        # tracker_index.start() # Mulai rekam daya hardware
        
        if len(files) == 1:
            docs = retriever.load_jsonl(files[0])
        else:
            print(f"Menggabungkan {len(files)} file JSONL untuk indexing: {files}")
            docs = retriever.load_jsonl_files(files)

        retriever.build_index(docs)
        retriever.save_index(args.save_dir)
        
        # emissions_index = tracker_index.stop() # Hentikan rekam daya
        # print(f"[CodeCarbon] Emisi Tahap Indexing: {emissions_index:.6f} kg CO2e")

    except FileNotFoundError as e:
        print(f"[BM25] {e}")
        print("Silakan jalankan preprocess_rag.py terlebih dahulu untuk membuat file JSONL.")
        exit(1)

    # ---------------------------------------------------------
    # MODE 2: MENGUJI LOADING & SEARCH
    # ---------------------------------------------------------
    print("\n--- TAHAP 2: MENGUJI LOAD & SEARCH ---")
    
    # Inisialisasi CodeCarbon terpisah untuk melacak fase Pencarian (Inferensi)
    # tracker_search = EmissionsTracker(
    #     project_name="BM25_Search_Inference",
    #     force_carbon_intensity_g_co2e_kwh=778.0, 
    #     output_dir = "CodeCarbon",
    #     output_file = "bm25_search_emissions.csv",
    #     log_level="error"
    # )

    # Inisialisasi instance baru untuk simulasi bahwa file dijalankan terpisah
    test_retriever = TaxBM25Retriever()
    test_retriever.load_index(args.save_dir)

    test_query = "Tata cara pendaftaran wajib pajak PMSE"
    print(f"\nQuery Pencarian: '{test_query}'")

    # tracker_search.start() # Mulai rekam daya pencarian
    
    # Lakukan pencarian 
    # (Untuk metrik inference speed di skripsi, kamu bisa mengukur waktu/token di bagian ini)
    top_results = test_retriever.search(query=test_query, top_k=3)
    
    # emissions_search = tracker_search.stop() # Hentikan rekam daya
    # print(f"[CodeCarbon] Emisi Tahap Pencarian: {emissions_search:.8f} kg CO2e")

    # Cetak hasil ke layar
    print("\n--- HASIL PENCARIAN ---")
    for rank, res in enumerate(top_results, start=1):
        print(f"\nPeringkat: {rank}")
        print(f"ID Chunk : {res['chunk_id']}")
        print(f"BM25 Score: {res['score']:.4f}")
        print(f"Metadata : Dokumen - {res['metadata'].get('document')}")
        
        # Menampilkan cuplikan konten agar terminal tidak terlalu penuh (disesuaikan dengan key document)
        text_content = res.get('document', '')
        content_preview = text_content.replace('\n', ' ')[:100]
        print(f"Isi      : {content_preview}...")