"""
Router klasifikasi: BGE-M3 (dibekukan) + Logistic Regression.

Tugas: menentukan apakah sebuah teks butuh perhitungan, dan rumus yang mana
(8 flag hitungan + 1 kelas 'tidak_butuh_hitungan').

Cara pakai
----------
Install dulu (di Colab biasanya tinggal baris pertama):
    pip install sentence-transformers scikit-learn pandas joblib matplotlib

Latih + evaluasi:
    python train_router.py train --csv data.csv --out router_out

Latih + evaluasi, lalu latih ulang di SEMUA data untuk dipakai produksi:
    python train_router.py train --csv data.csv --out router_out --final

Coba prediksi:
    python train_router.py predict --model router_out/router.joblib "hitung PPN 10 juta"

CSV wajib punya dua kolom: `contoh` (teks) dan `flag` (label).
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

MODEL_NAME = "BAAI/bge-m3"
TEXT_COL = "contoh"
LABEL_COL = "flag"
FALLBACK_LABEL = "tanpa_hitung"  # dipakai kalau confidence di bawah threshold
THRESHOLD = 0.6
MAX_SEQ_LEN = 128
SEED = 42


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
def load_encoder(device="cpu"):
    import torch
    from sentence_transformers import SentenceTransformer

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if device == "cuda":
        model.half()  # fp16: lebih cepat dan hemat memori di GPU
    print(f"[encoder] {MODEL_NAME} dimuat di {device}")
    return model


def embed(model, texts, batch_size=32):
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise SystemExit(
            f"Kolom {sorted(missing)} tidak ada di CSV. "
            f"Kolom yang ditemukan: {list(df.columns)}. "
            f"Ubah TEXT_COL/LABEL_COL di bagian atas script kalau namanya beda."
        )

    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""]

    # Teks yang sama tapi label beda = data bermasalah, tampilkan supaya bisa dibenahi
    n_labels = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    conflicts = n_labels[n_labels > 1].index.tolist()
    if conflicts:
        print(f"[data] PERINGATAN: {len(conflicts)} teks punya label bertentangan:")
        for t in conflicts[:10]:
            print(f"        - {t!r}")

    before = len(df)
    df = df.drop_duplicates(subset=TEXT_COL, keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"[data] {before - len(df)} baris duplikat dibuang")

    counts = df[LABEL_COL].value_counts()
    print("[data] jumlah contoh per kelas:")
    print(counts.to_string())

    if counts.min() < 5:
        raise SystemExit(
            f"Kelas '{counts.idxmin()}' hanya punya {counts.min()} contoh. "
            "Minimal 5 per kelas supaya bisa dibagi train/test dan cross-validation."
        )
    if counts.min() < 15:
        print("[data] PERINGATAN: ada kelas dengan < 15 contoh, hasil evaluasi akan kasar.")
    return df


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    texts = df[TEXT_COL].values
    labels = df[LABEL_COL].values

    # Test set dipisah dulu, stratified. Seed tetap supaya split bisa dipakai ulang
    # (misalnya untuk membandingkan dengan IndoBERT pada data yang sama).
    tr_text, te_text, y_tr, y_te = train_test_split(
        texts, labels, test_size=args.test_size, stratify=labels, random_state=SEED
    )
    pd.DataFrame({TEXT_COL: tr_text, LABEL_COL: y_tr}).to_csv(out / "train_split.csv", index=False)
    pd.DataFrame({TEXT_COL: te_text, LABEL_COL: y_te}).to_csv(out / "test_split.csv", index=False)
    print(f"[split] train={len(tr_text)}  test={len(te_text)}")

    encoder = load_encoder()
    t0 = time.perf_counter()
    X_tr = embed(encoder, tr_text)
    X_te = embed(encoder, te_text)
    print(f"[embed] {len(tr_text) + len(te_text)} teks selesai dalam {time.perf_counter() - t0:.1f} detik")

    # Cari nilai C terbaik lewat cross-validation di train set (test set tidak disentuh)
    min_count = pd.Series(y_tr).value_counts().min()
    cv = StratifiedKFold(n_splits=int(min(5, min_count)), shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        LogisticRegression(max_iter=3000, class_weight="balanced"),
        param_grid={"C": [0.1, 1, 10, 100]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    clf = grid.best_estimator_
    print(f"[train] C terbaik = {grid.best_params_['C']}  (CV macro-F1 = {grid.best_score_:.3f})")

    # Evaluasi di test set
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    class_names = list(clf.classes_)

    print("\n=== HASIL DI TEST SET ===")
    print(f"accuracy : {accuracy_score(y_te, pred):.3f}")
    print(f"macro-F1 : {f1_score(y_te, pred, average='macro'):.3f}\n")
    print(classification_report(y_te, pred, digits=3, zero_division=0))

    cm = confusion_matrix(y_te, pred, labels=class_names)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / "confusion_matrix.csv")
    save_confusion_plot(cm, class_names, out / "confusion_matrix.png")

    # Kalimat yang salah ditebak: ini bahan utama untuk menambah hard negative
    conf = proba.max(axis=1)
    wrong = np.where(pred != y_te)[0]
    print(f"=== {len(wrong)} KALIMAT SALAH DITEBAK ===")
    rows = []
    for i in wrong:
        print(f"- {te_text[i]!r}\n    asli={y_te[i]}  tebakan={pred[i]}  conf={conf[i]:.2f}")
        rows.append({TEXT_COL: te_text[i], "asli": y_te[i], "tebakan": pred[i], "conf": round(float(conf[i]), 3)})
    pd.DataFrame(rows).to_csv(out / "kesalahan_test.csv", index=False)

    # Efek confidence threshold: makin tinggi, makin sedikit yang dijawab router
    print("\n=== EFEK THRESHOLD ===")
    print("threshold | dijawab router | akurasi yang dijawab")
    for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = conf >= t
        cover = mask.mean()
        acc = (pred[mask] == y_te[mask]).mean() if mask.any() else float("nan")
        print(f"   {t:.1f}    |     {cover:6.1%}     |       {acc:6.1%}")

    # Simpan model
    if args.final:
        X_all = np.vstack([X_tr, X_te])
        y_all = np.concatenate([y_tr, y_te])
        clf = LogisticRegression(
            C=grid.best_params_["C"], max_iter=3000, class_weight="balanced"
        ).fit(X_all, y_all)
        print("\n[final] classifier dilatih ulang di semua data (train + test)")
        print("        angka evaluasi di atas tetap berasal dari model yang hanya dilatih di train")

    joblib.dump(
        {"clf": clf, "model_name": MODEL_NAME, "threshold": THRESHOLD, "max_seq_len": MAX_SEQ_LEN},
        out / "router.joblib",
    )
    print(f"\n[simpan] semua output ada di folder: {out.resolve()}")


def save_confusion_plot(cm, class_names, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label asli")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:  # plot hanya tambahan, jangan sampai menggagalkan training
        print(f"[plot] confusion matrix tidak bisa digambar: {e}")


# ----------------------------------------------------------------------------
# Inference (bisa diimport ke pipeline RAG)
# ----------------------------------------------------------------------------
class Router:
    """
    router = Router("router_out/router.joblib")
    label, conf = router.route("hitung PPN 10 juta")

    Kalau confidence < threshold, label dikembalikan sebagai FALLBACK_LABEL
    (artinya: lanjut ke jalur LLM biasa).

    PENTING: embedding untuk classifier harus dibuat dengan cara yang sama
    seperti saat training (sentence-transformers, normalize=True). Kalau kamu
    mau memakai ulang encoder dari pipeline RAG, pastikan hasilnya setara.
    """

    def __init__(self, path, threshold=None, encoder=None):
        bundle = joblib.load(path)
        self.clf = bundle["clf"]
        self.threshold = bundle["threshold"] if threshold is None else threshold
        self.encoder = encoder if encoder is not None else load_encoder()

    def route(self, text):
        x = embed(self.encoder, [text])
        p = self.clf.predict_proba(x)[0]
        i = int(p.argmax())
        label, conf = str(self.clf.classes_[i]), float(p[i])
        return (label if conf >= self.threshold else FALLBACK_LABEL), conf


def predict(args):
    router = Router(args.model, threshold=args.threshold)
    print(f"threshold = {router.threshold}")
    router.route("pemanasan")  # panggilan pertama selalu lebih lambat, jangan diukur
    for text in args.texts:
        t0 = time.perf_counter()
        label, conf = router.route(text)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{text!r}\n    -> {label}  (conf={conf:.2f}, {ms:.1f} ms)")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Router klasifikasi BGE-M3 + Logistic Regression")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="latih dan evaluasi")
    p_train.add_argument("--csv", required=True, help="path file CSV (kolom: contoh, flag)")
    p_train.add_argument("--out", default="router_out", help="folder output")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--final", action="store_true", help="setelah evaluasi, latih ulang di semua data untuk disimpan")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="coba prediksi satu atau beberapa kalimat")
    p_pred.add_argument("--model", default="../router_out/router.joblib")
    p_pred.add_argument("--threshold", type=float, default=None)
    p_pred.add_argument("texts", nargs="+")
    p_pred.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()