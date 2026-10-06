import math
from decimal import Decimal

# ==========================================
# KONSTANTA & FALLBACK MESSAGE
# ==========================================

FALLBACK_MESSAGE = "Maaf, untuk pertanyaan Anda silakan berkonsultasi dengan ahli pajak."

# 1. Konstanta Sanksi Bunga (Asumsi/Simulasi)
TARIF_BUNGA_2023 = 0.0099

# 2. Konstanta Denda SPT
TARIF_DENDA_SPT = {
    "tahunan_badan": 1_000_000.0,
    "tahunan_op": 100_000.0,
    "masa_ppn": 500_000.0,
    "masa_lainnya": 100_000.0,
}

# 3. Konstanta PPN
TARIF_PPN_2023 = 0.11

# 4. Konstanta NJOP Dasar (Asumsi/Simulasi)
DEFAULT_NJOP_BUMI_PER_M2 = 2_000_000.0
DEFAULT_NJOP_BANGUNAN_PER_M2 = 1_500_000.0

# 5. Konstanta PBB
NJOPTKP = 12_000_000.0
TARIF_PBB = 0.005

# 6. Konstanta Penghasilan Tidak Kena Pajak (PTKP)
NILAI_PTKP = {
    "TK/0": 54_000_000.0,
    "TK/1": 58_500_000.0,
    "TK/2": 63_000_000.0,
    "TK/3": 67_500_000.0,
    "K/0": 58_500_000.0,
    "K/1": 63_000_000.0,
    "K/2": 67_500_000.0,
    "K/3": 72_000_000.0,
}

# 7. Konstanta PPh Badan
TARIF_PPH_BADAN = 0.22


# ==========================================
# FUNGSI-FUNGSI PERHITUNGAN PAJAK
# ==========================================

def hitung_sanksi_bunga(pajak_kurang_bayar: float | None = None, jumlah_bulan: int | None = None) -> float | str:
    """
    Tujuan: Menghitung sanksi bunga administrasi karena kurang bayar.
    Parameter:
        - pajak_kurang_bayar (float): Nominal pajak yang belum dibayar.
        - jumlah_bulan (int): Lama keterlambatan dalam bulan.
    Rumus:
        Sanksi = pajak kurang bayar × tarif bunga per bulan × jumlah bulan
    Asumsi/Ketentuan:
        - Menggunakan tarif asumsi 0,99% (bukan klaim tarif tunggal resmi sepanjang 2023).
        - Jumlah bulan maksimal adalah 24 bulan.
    """
    if pajak_kurang_bayar is None: 
        return "Tolong informasikan mengenai nominal pajak kurang bayar."
    if jumlah_bulan is None:
        return "Tolong informasikan mengenai lama keterlambatan dalam bulan."
    if pajak_kurang_bayar < 0 or jumlah_bulan < 0:
        return "Tolong informasikan nilai pajak kurang bayar dan lama keterlambatan ulang."
    
    bulan_efektif = min(jumlah_bulan, 24)
    
    # Menggunakan Decimal untuk menghindari masalah presisi floating-point
    sanksi = Decimal(str(pajak_kurang_bayar)) * Decimal(str(TARIF_BUNGA_2023)) * Decimal(bulan_efektif)
    return float(sanksi)



def hitung_denda_spt(jenis_spt: str | None = None) -> float | str:
    """
    Tujuan: Mendapatkan nominal denda keterlambatan pelaporan SPT.
    Parameter:
        - jenis_spt (str): Jenis SPT ("tahunan_badan", "tahunan_op", "masa_ppn", "masa_lainnya").
    Rumus:
        Lookup langsung dari kamus/tabel denda baku.
    Asumsi/Ketentuan:
        - Tidak ada perhitungan matematis. Hanya pemetaan nilai statis.
    """
    if jenis_spt is None:
        return "Tolong informasikan apakah anda mengajukan SPT tahunan badan, tahunan orang pribadi, masa PPN, atau masa lainnya."
    if jenis_spt not in TARIF_DENDA_SPT:
        return FALLBACK_MESSAGE
    
    return TARIF_DENDA_SPT[jenis_spt]

def hitung_ppn(dpp: float | None = None) -> float | str:
    """
    Tujuan: Menghitung Pajak Pertambahan Nilai (PPN).
    Parameter:
        - dpp (float): Dasar Pengenaan Pajak (nilai dasar transaksi).
    Rumus:
        PPN = 11% × DPP
    Asumsi/Ketentuan:
        - Berbasis aturan SDSN 2023 (tarif 11%).
        - Tidak menggunakan DPP Nilai Lain.
    """
    if dpp is None:
        return "Tolong informasikan mengenai nilai dasar pengenaan pajak (DPP)-nya."
    if dpp < 0:
        return "Tolong informasikan nilai DPP yang valid."
    
    ppn = Decimal(str(dpp)) * Decimal(str(TARIF_PPN_2023))
    return float(ppn)


def hitung_njop(
    luas_bumi: float | None = None, 
    luas_bangunan: float | None = None, 
    njop_bumi_per_m2: float | None = None, 
    njop_bangunan_per_m2: float | None = None
) -> float | str:
    """
    Tujuan: Menghitung total Nilai Jual Objek Pajak (NJOP).
    Parameter:
        - luas_bumi (float): Luas tanah dalam m2.
        - luas_bangunan (float): Luas bangunan dalam m2.
        - njop_bumi_per_m2 (float): Harga NJOP tanah per m2.
        - njop_bangunan_per_m2 (float): Harga NJOP bangunan per m2.
    Rumus:
        NJOP total = (luas bumi × NJOP/m2) + (luas bangunan × NJOP/m2)
    Asumsi/Ketentuan:
        - Nilai default digunakan untuk sekadar asumsi/simulasi apabila data wilayah tidak ada.
    """
    if luas_bumi is None: 
        return "Tolong informasikan mengenai luas tanah (m2)-nya."
    if luas_bangunan is None:
        return "Tolong informasikan mengenai luas bangunan (m2)-nya."
    if njop_bumi_per_m2 is None:
        njop_bumi_per_m2 = DEFAULT_NJOP_BUMI_PER_M2
    if njop_bangunan_per_m2 is None:
        njop_bangunan_per_m2 = DEFAULT_NJOP_BANGUNAN_PER_M2

    if luas_bumi < 0 or luas_bangunan < 0 or njop_bumi_per_m2 < 0 or njop_bangunan_per_m2 < 0:
        return FALLBACK_MESSAGE
    
    total_bumi = Decimal(str(luas_bumi)) * Decimal(str(njop_bumi_per_m2))
    total_bangunan = Decimal(str(luas_bangunan)) * Decimal(str(njop_bangunan_per_m2))
    njop_total = total_bumi + total_bangunan
    
    return float(njop_total)



def hitung_pbb(njop_total: float | None = None) -> float | str:
    """
    Tujuan: Menghitung Pajak Bumi dan Bangunan (PBB) Terutang.
    Parameter:
        - njop_total (float): Total NJOP (Bumi + Bangunan).
    Rumus:
        NJKP = (NJOP total - NJOPTKP) × Persentase NJKP (20% atau 40%)
        PBB = NJKP × 0,5%
    Asumsi/Ketentuan:
        - Menggunakan model perhitungan lama PBB Pusat sebelum diubah ke tarif daerah penuh, 
          untuk kebutuhan model simulasi pada proyek 2023.
        - Jika NJOP > 1M menggunakan 40%, jika tidak 20%.
    """
    if njop_total is None:
        return "Tolong informasikan mengenai total NJOP anda."
    if njop_total < 0:
        return FALLBACK_MESSAGE
    
    if njop_total <= NJOPTKP:
        return 0.0
        
    persentase_njkp = 0.40 if njop_total > 1_000_000_000.0 else 0.20
    
    njkp = (Decimal(str(njop_total)) - Decimal(str(NJOPTKP))) * Decimal(str(persentase_njkp))
    pbb = njkp * Decimal(str(TARIF_PBB))
    
    return float(pbb)




def hitung_pph_orang_pribadi(penghasilan_neto_setahun: float | None = None, status_ptkp: str | None = None) -> float | str:
    """
    Tujuan: Menghitung PPh terutang untuk Wajib Pajak Orang Pribadi.
    Parameter:
        - penghasilan_neto_setahun (float): Total penghasilan neto dalam 1 tahun.
        - status_ptkp (str): Kode PTKP (contoh: "TK/0", "K/1").
    Rumus:
        PKP = Penghasilan Neto - PTKP (dibulatkan ke bawah ke ribuan)
        Pajak = PKP × Tarif Progresif Pasal 17
    Asumsi/Ketentuan:
        - Menggunakan lapisan tarif Pasal 17 UU HPP (2023).
        - 5% (0-60jt), 15% (60jt-250jt), 25% (250jt-500jt), 30% (500jt-5M), 35% (>5M).
    """
    if penghasilan_neto_setahun is None:
        return "Tolong informasikan mengenai penghasilan neto setahun anda."
    if status_ptkp is None:
        return "Tolong informasikan mengenai status PTKP anda."
    if penghasilan_neto_setahun < 0 or status_ptkp not in NILAI_PTKP:
        return FALLBACK_MESSAGE
    
    ptkp = NILAI_PTKP[status_ptkp]
    pkp = penghasilan_neto_setahun - ptkp
    
    if pkp <= 0:
        return 0.0
    
    # Membulatkan PKP ke bawah menjadi ribuan penuh
    pkp_dibulatkan = math.floor(pkp / 1000) * 1000
    pkp_sisa = Decimal(str(pkp_dibulatkan))
    
    pajak_total = Decimal("0.0")
    
    # Lapisan 1: 5% (Maks Rp60.000.000)
    lapisan_1 = min(pkp_sisa, Decimal("60000000.0"))
    if lapisan_1 > 0:
        pajak_total += lapisan_1 * Decimal("0.05")
        pkp_sisa -= lapisan_1

    # Lapisan 2: 15% (Rp60.000.000 s.d. Rp250.000.000 -> Range: 190.000.000)
    lapisan_2 = min(pkp_sisa, Decimal("190000000.0"))
    if lapisan_2 > 0:
        pajak_total += lapisan_2 * Decimal("0.15")
        pkp_sisa -= lapisan_2
        
    # Lapisan 3: 25% (Rp250.000.000 s.d. Rp500.000.000 -> Range: 250.000.000)
    lapisan_3 = min(pkp_sisa, Decimal("250000000.0"))
    if lapisan_3 > 0:
        pajak_total += lapisan_3 * Decimal("0.25")
        pkp_sisa -= lapisan_3
        
    # Lapisan 4: 30% (Rp500.000.000 s.d. Rp5.000.000.000 -> Range: 4.500.000.000)
    lapisan_4 = min(pkp_sisa, Decimal("4500000000.0"))
    if lapisan_4 > 0:
        pajak_total += lapisan_4 * Decimal("0.30")
        pkp_sisa -= lapisan_4
        
    # Lapisan 5: 35% (Di atas Rp5.000.000.000)
    if pkp_sisa > 0:
        pajak_total += pkp_sisa * Decimal("0.35")
        
    return float(pajak_total)


def hitung_pph_badan(pkp: float | None = None) -> float | str:
    """
    Tujuan: Menghitung PPh terutang untuk Wajib Pajak Badan.
    Parameter:
        - pkp (float): Penghasilan Kena Pajak.
    Rumus:
        PPh Badan = 22% × PKP
    Asumsi/Ketentuan:
        - Menggunakan tarif umum 22% (berlaku di 2023).
    """
    if pkp is None:
        return "Tolong informasikan mengenai Penghasilan Kena Pajak (PKP) anda."
    if pkp < 0:
        return FALLBACK_MESSAGE
    
    pajak = Decimal(str(pkp)) * Decimal(str(TARIF_PPH_BADAN))
    return float(pajak)


# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("=== TESTING SANKSI BUNGA ===")
    # Expected: 10,000,000 * 0.0099 * 3 = 297,000.0
    print(hitung_sanksi_bunga(10_000_000.0, 3)) 
    print(hitung_sanksi_bunga(-500.0, 2)) # Fallback message test
    
    print("\n=== TESTING DENDA SPT ===")
    # Expected: 1,000,000.0
    print(hitung_denda_spt("tahunan_badan"))
    print(hitung_denda_spt("tidak_diketahui")) # Fallback message test

    print("\n=== TESTING PPN ===")
    # Expected: 6,000,000 * 11% = 660,000.0
    print(hitung_ppn(6_000_000.0))
    print(hitung_ppn(-1000.0)) # Fallback message test

    print("\n=== TESTING NJOP ===")
    # Expected: (120 * 2,000,000) + (80 * 1,500,000) = 360,000,000.0
    print(hitung_njop(120.0, 80.0))

    print("\n=== TESTING PBB ===")
    # Expected: NJOP 360M <= 1Milyar -> 20%. NJKP = (360M - 12M) * 20% = 69.6M. PBB = 69.6M * 0.5% = 348,000.0
    print(hitung_pbb(360_000_000.0))

    print("\n=== TESTING PPh ORANG PRIBADI ===")
    # PKP = 120M - 63M (K/1) = 57M. 
    # Pajak: 57M * 5% = 2,850,000.0
    print(hitung_pph_orang_pribadi(120_000_000.0, "K/1"))
    print(hitung_pph_orang_pribadi(120_000_000.0, "TK/99")) # Fallback message test

    print("\n=== TESTING PPh BADAN ===")
    # Expected: 800,000,000 * 22% = 176,000,000.0
    print(hitung_pph_badan(800_000_000.0))