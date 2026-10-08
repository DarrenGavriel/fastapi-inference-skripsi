"""
Ekstraksi parameter untuk fungsi hitung_* dari query pengguna.

Alur:
    router (flag) -> build_system_prompt(flag) -> LLM -> JSON mentah
    -> validate_and_clean (Python memeriksa) -> to_function_kwargs
    -> fungsi hitung_*  ATAU  tanya balik ke pengguna kalau ada yang kosong

Pembagian tugas:
    LLM    : membaca kalimat, menentukan angka mana untuk field apa, memetakan status PTKP
    Python : memastikan angka benar-benar ada di teks pengguna, memeriksa nilai yang diizinkan,
             mengalikan 12 (bulanan -> tahunan), mencari nilai PTKP, dan menghitung pajak

Jalankan self-test (tanpa model):  python extractor.py
"""
import json
import math
import re

# ----------------------------------------------------------------------------
# Nilai yang diizinkan
# ----------------------------------------------------------------------------
PTKP_STATUSES = [
    "TK/0", "TK/1", "TK/2", "TK/3",
    "K/0", "K/1", "K/2", "K/3",
    "K/I/0", "K/I/1", "K/I/2", "K/I/3",
]
JENIS_SPT = ["tahunan_badan", "tahunan_op", "masa_ppn", "masa_lainnya"]
PERIODE = ["bulanan", "tahunan"]

# Field yang diisi LLM untuk tiap flag (nama flag = label dari router, persis sama)
FIELDS = {
    "sanksi_bunga": ["pajak_kurang_bayar", "jumlah_bulan"],
    "denda_spt": ["jenis_spt"],
    "ppn": ["dpp"],
    "NJOP": ["luas_bumi", "luas_bangunan", "njop_bumi_per_m2", "njop_bangunan_per_m2"],
    "PBB": ["njop_total"],
    "pph_op": ["penghasilan", "periode", "status_ptkp"],
    "pph_badan": ["pkp"],
}
INT_FIELDS = {"jumlah_bulan"}
ZERO_OK = {"luas_bangunan", "njop_bangunan_per_m2"}  # boleh 0 kalau tanah kosong
ENUM_FIELDS = {"jenis_spt": JENIS_SPT, "periode": PERIODE}

# ----------------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------------
BASE_PROMPT = """Kamu adalah ekstraktor parameter untuk kalkulator pajak Indonesia.
Tugasmu hanya membaca pertanyaan pengguna lalu mengeluarkan SATU objek JSON sesuai skema di bawah. Kamu tidak menghitung pajak dan tidak menjawab pertanyaan.

Aturan:
1. Keluarkan JSON saja. Tanpa penjelasan, tanpa teks lain, tanpa markdown.
2. Gunakan hanya field yang ada di skema.
3. Kalau sebuah informasi tidak disebut jelas oleh pengguna, isi null. Jangan menebak, jangan memakai nilai bawaan, jangan mengarang angka.
4. Jangan menghitung apa pun (tidak menjumlah, tidak mengalikan, tidak mengubah satuan waktu). Salin angka sesuai yang disebut pengguna.
5. Nominal ditulis sebagai angka penuh tanpa titik, koma, atau "Rp". Contoh: "10 juta" -> 10000000, "10jt" -> 10000000, "500 ribu" -> 500000, "10,5 juta" -> 10500000, "Rp 10.000.000" -> 10000000."""

FLAG_BLOCKS = {
    "sanksi_bunga": """Tugas: ekstrak parameter sanksi bunga.
Skema: {"pajak_kurang_bayar": angka atau null, "jumlah_bulan": bilangan bulat atau null}
- pajak_kurang_bayar: nominal pajak yang kurang dibayar, dalam rupiah.
- jumlah_bulan: lama keterlambatan atau kekurangan, dalam bulan. Isi hanya jika pengguna menyebut jumlah bulan. Jika pengguna hanya memberi tanggal, isi null.

Contoh:
Pengguna: pajak kurang bayar saya 15 juta, telat 3 bulan, bunganya berapa?
{"pajak_kurang_bayar": 15000000, "jumlah_bulan": 3}
Pengguna: hitung sanksi bunga untuk kekurangan bayar 2 juta
{"pajak_kurang_bayar": 2000000, "jumlah_bulan": null}""",

    "denda_spt": """Tugas: ekstrak jenis SPT untuk menghitung denda keterlambatan lapor.
Skema: {"jenis_spt": "tahunan_op" atau "tahunan_badan" atau "masa_ppn" atau "masa_lainnya" atau null}
- tahunan_op: SPT Tahunan orang pribadi (OP, pribadi, 1770).
- tahunan_badan: SPT Tahunan badan (PT, CV, perusahaan, badan usaha, 1771).
- masa_ppn: SPT Masa PPN.
- masa_lainnya: SPT Masa selain PPN (misalnya PPh 21, PPh 23, PPh 4 ayat 2, PPh 25).
- Jika pengguna hanya menulis "SPT" atau "SPT Masa" tanpa jelas jenisnya, isi null.

Contoh:
Pengguna: SPT Tahunan PT saya telat lapor, dendanya berapa?
{"jenis_spt": "tahunan_badan"}
Pengguna: Kalau SPT Masa selain PPN terlambat, denda-nya berapa?
{"jenis_spt": "masa_lainnya"}
Pengguna: hitung denda telat lapor SPT
{"jenis_spt": null}""",

    "ppn": """Tugas: ekstrak parameter PPN.
Skema: {"dpp": angka atau null}
- dpp: Dasar Pengenaan Pajak, yaitu harga jual, nilai transaksi, atau nilai penyerahan sebelum PPN, dalam rupiah.
- Jika pengguna menyebut harga "sudah termasuk PPN", isi null (rumusnya berbeda).

Contoh:
Pengguna: hitung PPN untuk penjualan 10 juta
{"dpp": 10000000}
Pengguna: berapa PPN-nya?
{"dpp": null}
Pengguna: harga 11,1 juta sudah termasuk PPN, PPN-nya berapa?
{"dpp": null}""",

    "NJOP": """Tugas: ekstrak parameter perhitungan NJOP.
Skema: {"luas_bumi": angka atau null, "luas_bangunan": angka atau null, "njop_bumi_per_m2": angka atau null, "njop_bangunan_per_m2": angka atau null}
- luas_bumi, luas_bangunan: luas dalam meter persegi, angka saja.
- njop_bumi_per_m2, njop_bangunan_per_m2: NJOP (harga/nilai) per meter persegi, dalam rupiah.
- Jika pengguna menyebut tidak ada bangunan (tanah kosong), isi luas_bangunan 0 dan njop_bangunan_per_m2 0.

Contoh:
Pengguna: hitung NJOP tanah 200 m2 NJOP-nya 1,5 juta per meter, bangunan 100 m2 NJOP 3 juta per meter
{"luas_bumi": 200, "luas_bangunan": 100, "njop_bumi_per_m2": 1500000, "njop_bangunan_per_m2": 3000000}
Pengguna: hitung NJOP tanah kosong 300 m2, 2 juta per m2
{"luas_bumi": 300, "luas_bangunan": 0, "njop_bumi_per_m2": 2000000, "njop_bangunan_per_m2": 0}
Pengguna: hitung NJOP rumah saya, luas tanah 120 m2
{"luas_bumi": 120, "luas_bangunan": null, "njop_bumi_per_m2": null, "njop_bangunan_per_m2": null}""",

    "PBB": """Tugas: ekstrak parameter PBB.
Skema: {"njop_total": angka atau null}
- njop_total: total NJOP (bumi dan bangunan) dalam rupiah, persis seperti yang disebut pengguna. Jangan menjumlahkan sendiri.

Contoh:
Pengguna: NJOP total rumah saya 500 juta, PBB-nya berapa?
{"njop_total": 500000000}
Pengguna: hitung PBB tanah 100 m2
{"njop_total": null}""",

    "pph_op": """Tugas: ekstrak parameter PPh Orang Pribadi.
Skema: {"penghasilan": angka atau null, "periode": "bulanan" atau "tahunan" atau null, "status_ptkp": string atau null}
- penghasilan: penghasilan yang disebut pengguna (gaji, penghasilan, pendapatan), dianggap sudah neto. Tulis angka persis seperti yang disebut, jangan dikali 12.
- periode: "bulanan" jika disebut per bulan/sebulan/bulanan, "tahunan" jika disebut per tahun/setahun/tahunan, null jika tidak disebut.
- status_ptkp: salah satu dari TK/0, TK/1, TK/2, TK/3, K/0, K/1, K/2, K/3, K/I/0, K/I/1, K/I/2, K/I/3.
  TK = tidak kawin (belum menikah, lajang, janda, duda). K = kawin. K/I = kawin dan penghasilan istri digabung dengan suami.
  Angka terakhir = jumlah tanggungan (misalnya anak), maksimal 3. Jika lebih dari 3, tulis 3.
  Jika pengguna menyebut belum menikah/lajang tanpa menyebut tanggungan, isi TK/0.
  Jika pengguna menyebut menikah tanpa menyebut jumlah anak/tanggungan, isi null.
  Jika status pernikahan tidak disebut, isi null.

Contoh:
Pengguna: gaji neto saya 8 juta sebulan, menikah anak 2, hitung PPh
{"penghasilan": 8000000, "periode": "bulanan", "status_ptkp": "K/2"}
Pengguna: penghasilan neto setahun 120 juta, belum menikah
{"penghasilan": 120000000, "periode": "tahunan", "status_ptkp": "TK/0"}
Pengguna: hitung PPh OP saya, penghasilan 10jt
{"penghasilan": 10000000, "periode": null, "status_ptkp": null}""",

    "pph_badan": """Tugas: ekstrak parameter PPh Badan.
Skema: {"pkp": angka atau null}
- pkp: Penghasilan Kena Pajak, dalam rupiah. Isi hanya jika pengguna menyebut PKP / penghasilan kena pajak / laba kena pajak. Jika hanya menyebut omzet, laba, atau peredaran bruto, isi null.

Contoh:
Pengguna: PKP PT kami 2 miliar, PPh badan-nya berapa?
{"pkp": 2000000000}
Pengguna: hitung PPh badan, omzet kami 5 miliar
{"pkp": null}""",
}


def build_system_prompt(flag):
    if flag not in FLAG_BLOCKS:
        raise ValueError(f"Flag '{flag}' tidak punya blok ekstraksi (flag valid: {list(FLAG_BLOCKS)})")
    return BASE_PROMPT + "\n\n" + FLAG_BLOCKS[flag]


# ----------------------------------------------------------------------------
# Parser angka (deterministik): membaca SEMUA angka yang mungkin ada di teks
# ----------------------------------------------------------------------------
_UNIT_MULT = {
    "juta": 1e6, "jt": 1e6, "ribu": 1e3, "rb": 1e3, "k": 1e3,
    "miliar": 1e9, "milyar": 1e9, "triliun": 1e12,
}
_NUM_RE = re.compile(
    r"(?<![\d.,])(\d[\d.,]*\d|\d)(?:\s*(juta|jt|ribu|rb|miliar|milyar|triliun|k)(?:an)?\b)?",
    re.IGNORECASE,
)
_WORD_NUM = {
    "sebulan": 1, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11, "dua belas": 12,
}
_WORD_RE = re.compile(r"\b(" + "|".join(sorted(_WORD_NUM, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def _interpretations(raw):
    """Semua cara masuk akal membaca '10.500' / '10,5' / '1.234,56'. Format Indonesia diutamakan."""
    out = set()
    try:
        if raw.isdigit():
            out.add(float(raw))
        elif "." in raw and "," in raw:
            if raw.rfind(",") > raw.rfind("."):  # 1.234,56 (gaya Indonesia)
                out.add(float(raw.replace(".", "").replace(",", ".")))
            else:  # 1,234.56 (gaya Inggris)
                out.add(float(raw.replace(",", "")))
        else:
            sep = "." if "." in raw else ","
            parts = raw.split(sep)
            if all(len(p) == 3 for p in parts[1:]):  # pemisah ribuan: 10.000.000
                out.add(float("".join(parts)))
            if len(parts) == 2:  # desimal: 10,5 / 10.5
                out.add(float(parts[0] + "." + parts[1]))
    except ValueError:
        pass
    return out


def number_matches(text, include_words=True):
    """Satu himpunan interpretasi angka untuk SETIAP angka yang muncul di teks."""
    out = []
    for m in _NUM_RE.finditer(text):
        mult = _UNIT_MULT[m.group(2).lower()] if m.group(2) else 1
        vals = {round(v * mult, 6) for v in _interpretations(m.group(1))}
        if vals:
            out.append(vals)
    if include_words:
        for m in _WORD_RE.finditer(text):
            out.append({float(_WORD_NUM[m.group(1).lower()])})
    return out


def number_candidates(text):
    """Himpunan semua angka yang bisa dibaca dari teks pengguna."""
    cands = set()
    for s in number_matches(text):
        cands |= s
    return cands


# ----------------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------------
def normalize_status(value):
    """'k2', 'TK 0', 'K/I/1' -> bentuk baku; None kalau tidak ada di daftar."""
    if not isinstance(value, str):
        return None
    s = re.sub(r"[\s_\-]+", "/", value.strip().upper())
    m = re.fullmatch(r"(TK|K)/?(I)?/?([0-3])", s)
    if not m:
        return None
    status = f"{m.group(1)}/I/{m.group(3)}" if m.group(2) else f"{m.group(1)}/{m.group(3)}"
    return status if status in PTKP_STATUSES else None


def _check_number(value, cands, allow_zero, as_int):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or value < 0:
        return None
    value = float(value)
    if value == 0:
        return 0 if allow_zero else None
    if not any(math.isclose(value, c, rel_tol=1e-9) for c in cands):
        return None  # angka tidak ditemukan di teks pengguna
    if as_int:
        return int(value) if value.is_integer() else None
    return int(value) if value.is_integer() else value


def validate_and_clean(flag, query, raw):
    """
    query = teks pengguna yang dipakai untuk mencari angka. Sebaiknya gabungkan
    teks ASLI semua turn + hasil rewrite, supaya angka tidak lolos kalau rewrite salah mengubahnya.
    Return: (clean, problems). Nilai yang gagal diperiksa menjadi None, lalu ditanyakan ulang.
    """
    cands = number_candidates(query)
    clean, problems = {}, []
    for f in FIELDS[flag]:
        v = raw.get(f)
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            v = None
        if v is None:
            clean[f] = None
            continue

        if f == "status_ptkp":
            ok, why = normalize_status(v), "status PTKP tidak dikenal"
        elif f in ENUM_FIELDS:
            low = v.strip().lower() if isinstance(v, str) else None
            ok, why = (low if low in ENUM_FIELDS[f] else None), "nilai tidak ada di daftar yang diizinkan"
        else:
            ok = _check_number(v, cands, f in ZERO_OK, f in INT_FIELDS)
            why = "angka tidak ditemukan di teks pengguna atau tidak valid"

        if ok is None:
            problems.append({"field": f, "value": v, "alasan": why})
        clean[f] = ok
    return clean, problems


# ----------------------------------------------------------------------------
# Adaptor ke fungsi hitung_* dan pertanyaan balik
# ----------------------------------------------------------------------------
def to_function_kwargs(flag, clean):
    if flag == "pph_op":
        p, per = clean["penghasilan"], clean["periode"]
        setahun = None
        if p is not None and per == "bulanan":
            setahun = p * 12
        elif p is not None and per == "tahunan":
            setahun = p
        return {"penghasilan_neto_setahun": setahun, "status_ptkp": clean["status_ptkp"]}
    return {f: clean[f] for f in FIELDS[flag]}


QUESTIONS = {
    "pajak_kurang_bayar": "Berapa jumlah pajak yang kurang dibayar (dalam rupiah)?",
    "jumlah_bulan": "Berapa bulan lama keterlambatannya?",
    "jenis_spt": "SPT jenis apa? (SPT Tahunan OP, SPT Tahunan Badan, SPT Masa PPN, atau SPT Masa selain PPN)",
    "dpp": "Berapa nilai transaksinya sebelum PPN (DPP)?",
    "luas_bumi": "Berapa luas tanahnya (m²)?",
    "luas_bangunan": "Berapa luas bangunannya (m²)? Jika tidak ada bangunan, tulis 0.",
    "njop_bumi_per_m2": "Berapa NJOP tanah per m²?",
    "njop_bangunan_per_m2": "Berapa NJOP bangunan per m²? Jika tidak ada bangunan, tulis 0.",
    "njop_total": "Berapa total NJOP-nya (tanah + bangunan)?",
    "penghasilan_neto_setahun": "Berapa penghasilan netonya, dan apakah itu per bulan atau per tahun?",
    "status_ptkp": "Bagaimana status Anda: sudah menikah atau belum, dan berapa jumlah tanggungan?",
    "pkp": "Berapa Penghasilan Kena Pajak (PKP)-nya?",
}


def build_questions(missing, clean):
    qs = []
    for arg in missing:
        if arg == "penghasilan_neto_setahun" and clean.get("penghasilan") is not None:
            qs.append("Penghasilan tersebut per bulan atau per tahun?")
        else:
            qs.append(QUESTIONS[arg])
    return qs


# ----------------------------------------------------------------------------
# Parsing output LLM + fungsi utama
# ----------------------------------------------------------------------------
def parse_llm_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def extract(flag, query, generate, source_text=None, retries=1):
    """
    flag        : label dari router (mis. "pph_op")
    query       : query yang dikirim ke LLM (sudah di-rewrite kalau multi-turn)
    generate    : fungsi (system_prompt, user_text) -> string output LLM
    source_text : (opsional) teks asli semua turn, untuk pencarian angka. Default = query.
    """
    system = build_system_prompt(flag)
    raw = None
    for _ in range(retries + 1):
        raw = parse_llm_json(generate(system, query))
        if raw is not None:
            break
    if raw is None:
        return {"ok": False, "alasan": "output LLM bukan JSON valid", "kwargs": None,
                "clean": None, "missing": [], "questions": [], "problems": []}

    clean, problems = validate_and_clean(flag, source_text or query, raw)
    kwargs = to_function_kwargs(flag, clean)
    missing = [k for k, v in kwargs.items() if v is None]
    return {
        "ok": not missing,
        "kwargs": kwargs,
        "clean": clean,  # nilai hasil validasi (dipakai untuk menjelaskan periode bulanan/tahunan)
        "missing": missing,
        "questions": build_questions(missing, clean),
        "problems": problems,  # catat ke log: berguna untuk memperbaiki prompt/parser
    }


def make_unsloth_generate(model, tokenizer, max_new_tokens=200):
    """Contoh pembungkus untuk model Unsloth (BELUM diuji dengan model sungguhan)."""
    import torch

    def generate(system_prompt, user_text):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


# ----------------------------------------------------------------------------
# Self-test (tanpa model)
# ----------------------------------------------------------------------------
def _selftest():
    c = number_candidates
    assert 10_000_000 in c("hitung PPN 10 juta")
    assert 10_000_000 in c("Rp10jt")
    assert 10_000_000 in c("Rp 10.000.000")
    assert 10_500_000 in c("10,5 juta")
    assert 10_500_000 in c("10.5 juta")
    assert 500_000 in c("500 ribu") and 500_000 in c("500rb")
    assert 2_000_000_000 in c("PKP 2 miliar")
    assert 3 in c("telat 3 bulan") and 3 in c("telat tiga bulan")
    assert {10500, 10.5} <= c("luas 10.500")
    assert 100 in c("tanah 100 m2")

    assert normalize_status("k2") == "K/2"
    assert normalize_status("TK 0") == "TK/0"
    assert normalize_status("k/i/1") == "K/I/1"
    assert normalize_status("TK/I/0") is None
    assert normalize_status("K/4") is None

    # kasus normal pph_op
    q = "gaji neto saya 8jt sebulan, menikah anak 2"
    raw = {"penghasilan": 8000000, "periode": "bulanan", "status_ptkp": "k2"}
    clean, problems = validate_and_clean("pph_op", q, raw)
    assert not problems and clean["status_ptkp"] == "K/2"
    assert to_function_kwargs("pph_op", clean) == {
        "penghasilan_neto_setahun": 96_000_000, "status_ptkp": "K/2"}

    # LLM mengarang angka -> ditolak jadi None
    raw_bad = {"penghasilan": 9000000, "periode": "bulanan", "status_ptkp": "K/2"}
    clean, problems = validate_and_clean("pph_op", q, raw_bad)
    assert clean["penghasilan"] is None and problems[0]["field"] == "penghasilan"

    # tanah kosong: 0 diperbolehkan untuk field bangunan
    q = "hitung NJOP tanah kosong 300 m2, 2 juta per m2"
    raw = {"luas_bumi": 300, "luas_bangunan": 0, "njop_bumi_per_m2": 2000000, "njop_bangunan_per_m2": 0}
    clean, problems = validate_and_clean("NJOP", q, raw)
    assert not problems and clean["luas_bangunan"] == 0

    # enum denda_spt
    clean, problems = validate_and_clean("denda_spt", "SPT Masa selain PPN telat", {"jenis_spt": "masa_lainnya"})
    assert clean["jenis_spt"] == "masa_lainnya" and not problems
    clean, problems = validate_and_clean("denda_spt", "x", {"jenis_spt": "spt_apa_ini"})
    assert clean["jenis_spt"] is None and problems

    # extract() end-to-end dengan LLM palsu
    fake = lambda system, user: '```json\n{"penghasilan": 10000000, "periode": null, "status_ptkp": null}\n```'
    res = extract("pph_op", "hitung PPh OP saya, penghasilan 10jt", fake)
    assert not res["ok"] and "penghasilan_neto_setahun" in res["missing"]
    assert res["questions"][0] == "Penghasilan tersebut per bulan atau per tahun?"

    # output bukan JSON -> gagal dengan jelas setelah retry
    res = extract("ppn", "hitung PPN 10 juta", lambda s, u: "maaf saya tidak tahu")
    assert res["ok"] is False and res["alasan"]

    # prompt tersusun
    assert "ekstrak parameter PPN" in build_system_prompt("ppn")
    print("Semua self-test lolos.")


if __name__ == "__main__":
    _selftest()