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

## Lingkungan Eksperimen

Lingkungan referensi yang dipakai pada saat penelitian:

| Komponen           | Versi / Spesifikasi                                      |
|--------------------|----------------------------------------------------------|
| Sistem Operasi     | Windows 11 Home Single Language 25H2                     |
| Perangkat Keras    | Lenovo Yoga 7 14IML9 — Intel Core Ultra 7 155H, 16 GB LPDDR5x |
| GPU                | Intel Arc Graphics (integrated) via PyTorch **XPU** backend |
| Conda              | 25.9.1                                                   |
| Python             | 3.10.19                                                  |
| PyTorch            | 2.10.0+xpu                                               |
| Random Seed        | 42 (di-pin di seluruh notebook)                          |

## Setup Environment

Tersedia dua jalur instalasi. Pilih salah satu sesuai preferensi.

### Opsi A — Conda (direkomendasikan, sesuai naskah)

Cocok untuk replikasi 1:1 dengan lingkungan penelitian, terutama jika
memakai GPU Intel Arc (XPU backend).

```bash
# 1. Buat environment baru
conda env create -f environment.yml

# 2. Aktifkan
conda activate skripsi-aes

# 3. Daftarkan kernel Jupyter (agar dapat dipilih di notebook)
python -m ipykernel install --user --name skripsi-aes --display-name "Python (skripsi-aes)"

# 4. Jalankan Jupyter
jupyter notebook
```

### Opsi B — pip (alternatif, tanpa Conda)

Cocok jika tidak memakai Conda atau memakai GPU non-Intel (NVIDIA/CPU).
Lihat catatan PyTorch di `requirements.txt` untuk penyesuaian backend.

```bash
# 1. Buat virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install dependensi
pip install -r requirements.txt

# 3. Daftarkan kernel & jalankan Jupyter
python -m ipykernel install --user --name skripsi-aes --display-name "Python (skripsi-aes)"
jupyter notebook
```

### Catatan Backend PyTorch

- **Intel Arc / XPU** — `environment.yml` sudah memakai `torch==2.10.0+xpu`.
  Pastikan driver Intel Arc terbaru sudah ter-install.
- **NVIDIA / CUDA** — Ganti baris `torch==2.10.0+xpu` (di `environment.yml`)
  atau `torch==2.10.0` (di `requirements.txt`) dengan versi PyTorch CUDA yang
  sesuai, lalu install ulang. Pipeline lain tidak terpengaruh.
- **CPU only** — Bisa dijalankan, tapi `Ablasi 1 (RQ2)`, `Kriteria Multimodal`,
  dan `Baseline CLIP` akan lambat (training berbasis Transformer).

### API Key Gemma-3

Notebook `Kriteria Multimodal/extract short description.ipynb` memanggil
Google AI Studio API untuk ekstraksi deskripsi visual (`gemma-3-27b-it`).
Karena hasil ekstraksi sudah di-*cache* di `short_gemma_bertscore.csv`,
notebook ini **tidak perlu dijalankan ulang** untuk reproduksi hasil utama.
Jika ingin menjalankan ulang, isi `API_KEY` di sel pertama notebook tersebut.

## Urutan Eksekusi

1. (Sudah disediakan) Hasil split data di `train_df.csv` / `test_df.csv`.
2. (Sudah disediakan) Cache Gemma-3 + BERTScore di `short_gemma_bertscore.csv`.
3. Jalankan `Baseline CLIP/` untuk reproduksi baseline.
4. Jalankan `Kriteria Tekstual/` dan `Kriteria Multimodal/` untuk metode usulan.
5. Jalankan folder `Ablasi 1` sampai `Ablasi 4` dan `Ablasi Pooling`.
6. Jalankan `Uji Signifikansi (t-test)/` untuk validasi statistik.

## Dependensi Utama

Dependensi terkunci pada versi spesifik untuk menjamin reproducibility.
Daftar lengkap ada di `environment.yml` dan `requirements.txt`.

| Library                | Versi          | Fungsi                                                          |
|------------------------|----------------|-----------------------------------------------------------------|
| `torch`                | 2.10.0+xpu     | Framework deep learning, akselerasi XPU                          |
| `transformers`         | 5.0.1.dev0     | RoBERTa, CLIP, AutoTokenizer/AutoModel                          |
| `google-generativeai`  | 0.8.6          | API Gemma-3 untuk ekstraksi visual                              |
| `bert-score`           | 0.3.13         | Evaluasi keselarasan semantik (RoBERTa-Large)                   |
| `scikit-learn`         | 1.7.2          | `GroupShuffleSplit`, `GroupKFold`, `cohen_kappa_score`          |
| `scipy`                | 1.15.3         | Paired T-Test, Wilcoxon, skewness, kurtosis                     |
| `pandas`               | 2.3.3          | Manipulasi tabel dataset                                        |
| `numpy`                | 2.2.6          | Komputasi numerik                                               |
| `pillow`               | 12.0.0         | Pembacaan gambar (`.jpg`)                                       |
| `tqdm`                 | 4.67.1         | Progress bar                                                    |
| `matplotlib`           | 3.10.8         | Visualisasi QWK                                                 |
