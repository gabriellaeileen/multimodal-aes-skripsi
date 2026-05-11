# Multimodal AES Skripsi

Repositori source code dan dataset untuk skripsi
"Strategi Dekomposisi Pipeline Berbasis Konversi Modalitas
Menggunakan Gemma-3 dan RoBERTa untuk Mengatasi Kesenjangan
Semantik pada Penilaian Esai Multimodal Otomatis" (Gultom, 2026).

## Struktur

| Folder                      | Deskripsi                                      |
|-----------------------------|------------------------------------------------|
| `Baseline CLIP/`            | Implementasi baseline CLIP dengan sliding win. |
| `Kriteria Tekstual/`        | RoBERTa MTL untuk 8 kriteria tekstual.         |
| `Kriteria Multimodal/`      | RoBERTa STL + Gemma-3 + BERTScore.             |
| `Ablasi 1 (RQ2)/`           | Integrasi fitur visual pada kriteria tekstual. |
| `Ablasi 2 (RQ3)/`           | STL vs MTL pada kriteria multimodal.           |
| `Ablasi 3 (RQ4)/`           | Kontribusi optimasi grid search.               |
| `Ablasi 4 (RQ5)/`           | Dampak BERTScore sebagai jembatan semantik.    |
| `Ablasi Pooling/`           | Pemilihan strategi pooling pendahuluan.        |
| `Uji Signifikansi (t-test)/`| Paired Sample T-Test pembanding CLIP vs usulan.|

## Data

- `data.csv` — Dataset EssayJudge lengkap (1.054 entri).
- `train_df.csv` (867 baris) dan `test_df.csv` (187 baris) — Hasil Group Shuffle Split 80:20.
- `short_gemma_bertscore.csv` — Cache deskripsi visual Gemma-3 (45 kata) + BERTScore F1.

## Urutan Eksekusi

1. (Sudah disediakan) Hasil split data di `train_df.csv` / `test_df.csv`.
2. (Sudah disediakan) Cache Gemma-3 di `short_gemma_bertscore.csv`.
3. Jalankan `Baseline CLIP/` untuk reproduksi baseline.
4. Jalankan `Kriteria Tekstual/` dan `Kriteria Multimodal/` untuk metode usulan.
5. Jalankan folder Ablasi 1–4 dan Ablasi Pooling.
6. Jalankan `Uji Signifikansi (t-test)/` untuk validasi statistik.

## Lingkungan

- Python 3.10.19, Conda 25.9.1, PyTorch 2.10.0 (XPU backend)
- Perangkat: Lenovo Yoga 7 14IML9, Intel Core Ultra 7 155H, 16 GB LPDDR5x, Intel Arc Graphics
- Random seed = 42