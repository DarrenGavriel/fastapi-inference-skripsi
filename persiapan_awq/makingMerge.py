import json
import re

# ================= KONFIGURASI (HARDCODED) =================
INPUT_FILE = "temp_file.txt"            # Ganti dengan nama file txt kamu
OUTPUT_FILE = "merged_data.jsonl"       # Nama file output JSONL
FLAG_VALUE = "pph_badan"                # Ganti sesuai kebutuhan (bisa int 1/0 atau string "1")
# ===========================================================

def clean_text(text):
    # 1. Hapus tanda kutip sisa CSV di paling awal/akhir baris (jika ada)
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = re.sub(r"['\"]+$", "", text) # Bersihkan sisa kutip ganda di ujung seperti '""

    # 2. Ubah teks literal \n atau \\n menjadi newline (Enter) asli
    text = text.replace('\\n', '\n')

    # 3. Bersihkan invisible characters (seperti non-breaking space \xa0 dari PDF/Web)
    text = text.replace('\xa0', ' ').replace('\u200b', '')

    # 4. Normalisasi tanda kutip lengkung (Word) menjadi kutip lurus standar
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)

    # 5. Rapikan spasi berlebih TAPI jangan hapus Enter (\n)
    # Menggunakan [ \t]+ agar hanya spasi dan tab mendatar yang digabung
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Hapus spasi yang menempel sebelum atau sesudah Enter
    text = re.sub(r' *\n *', '\n', text)

    # 7. Batasi maksimal 2 Enter berturut-turut (agar tidak ada jarak kosong terlalu jauh)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def process_file():
    added_count = 0
    
    # Gunakan mode "r" untuk baca txt, dan mode "a" (append) untuk nambahin dari paling bawah
    with open(INPUT_FILE, mode="r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, mode="a", encoding="utf-8") as outfile:
        
        for line in infile:
            # Lewati baris kosong di file txt
            if not line.strip():
                continue
            
            cleaned_content = clean_text(line)
            
            # Struktur 2 kolom: content dan flag
            data = {
                "content": cleaned_content,
                "flag": FLAG_VALUE
            }
            
            # ensure_ascii=False agar karakter simbol tidak berubah jadi kode unicode
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            added_count += 1

    print(f"Selesai! Berhasil menambahkan {added_count} baris data dengan flag={FLAG_VALUE} ke '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    process_file()