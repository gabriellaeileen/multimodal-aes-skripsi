# Multimodal AES Skripsi

Repositori source code dan dataset untuk skripsi
"Strategi Dekomposisi Pipeline Berbasis Konversi Modalitas
Menggunakan Gemma-3 dan RoBERTa untuk Mengatasi Kesenjangan
Semantik pada Penilaian Esai Multimodal Otomatis".

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
| `scripts/`                  | Script persiapan dataset (unduh gambar).       |

## Data

- `data.csv` — Dataset EssayJudge lengkap (1.054 entri).
- `train_df.csv` (867 baris) dan `test_df.csv` (187 baris) — Hasil Group Shuffle Split 80:20.
- `short_gemma_bertscore.csv` — Cache deskripsi visual Gemma-3 (45 kata) + BERTScore F1.

> **Catatan gambar referensi**: Berkas gambar mentah EssayJudge **tidak
> disertakan** dalam repositori ini karena pertimbangan ukuran dan lisensi
> redistribusi. Lihat bagian [Persiapan Gambar](#persiapan-gambar) untuk
> instruksi pengunduhan.

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

Disediakan dua jalur instalasi. Pilih salah satu sesuai preferensi.

### Opsi A — Conda (direkomendasikan, sesuai naskah)

Cocok untuk replikasi 1:1 dengan lingkungan penelitian, terutama jika
memakai GPU Intel Arc (XPU backend). Driver Intel Arc terbaru sebaiknya
sudah ter-install.

```powershell
# 1. Buat environment dari spec
conda env create -f environment.yml

# 2. Aktifkan
conda activate skripsi-aes

# 3. (Opsional) daftarkan kernel Jupyter agar muncul di notebook
python -m ipykernel install --user --name skripsi-aes --display-name "Python (skripsi-aes)"

# 4. Jalankan Jupyter
jupyter notebook
```

`environment.yml` sudah menambahkan `--extra-index-url` ke PyTorch XPU,
sehingga `torch==2.10.0+xpu` beserta seluruh runtime Intel oneAPI
(`dpcpp-cpp-rt`, `intel-cmplr-*`, `onemkl-sycl-*`, `intel-openmp`, `mkl`,
`tbb`, `triton-xpu`, dst.) akan terinstal otomatis sebagai dependensi.

### Opsi B — pip + venv (alternatif, tanpa Conda)

Cocok jika tidak memakai Conda atau memakai backend non-XPU.

```powershell
# 1. Buat virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / macOS

# 2. Install PyTorch sesuai backend Anda (pilih SALAH SATU)
#   - Intel XPU (sesuai naskah):
pip install --extra-index-url https://download.pytorch.org/whl/xpu torch==2.10.0
#   - NVIDIA CUDA 12.x:
# pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.10.0
#   - CPU only:
# pip install torch==2.10.0

# 3. Install dependency selain PyTorch
pip install -r requirements.txt

# 4. (Opsional) daftarkan kernel & jalankan Jupyter
python -m ipykernel install --user --name skripsi-aes --display-name "Python (skripsi-aes)"
jupyter notebook
```

### Verifikasi Instalasi

```python
import torch
print(torch.__version__)          # harus: 2.10.0+xpu
print(torch.xpu.is_available())   # harus: True (untuk Intel Arc)
```

Notebook secara otomatis memilih device dengan urutan: `cuda` → `xpu` → `mps`
→ `cpu`. Tidak perlu mengubah kode jika memakai backend yang berbeda.

### API Key Gemma-3

Notebook `Kriteria Multimodal/extract short description.ipynb` memanggil
Google AI Studio API untuk ekstraksi deskripsi visual (`gemma-3-27b-it`).
Karena hasil ekstraksi sudah di-*cache* di `short_gemma_bertscore.csv`,
notebook ini **tidak perlu dijalankan ulang** untuk reproduksi hasil utama.
Jika ingin menjalankan ulang, isi `API_KEY` di sel pertama notebook tersebut.

## Persiapan Gambar

Berkas gambar referensi EssayJudge **tidak disertakan** dalam repositori
karena alasan ukuran dan lisensi redistribusi. Hanya `Baseline CLIP/` yang
membutuhkan gambar mentah pada saat eksekusi; seluruh notebook lain
mengandalkan cache fitur visual di `short_gemma_bertscore.csv`.

### Unduh otomatis dari sumber resmi

```bash
python scripts/download_images.py
```

Hasil unduhan disimpan di direktori `images/` pada akar repositori. Script
ini idempoten: gambar yang sudah ada di lokal akan dilewati. Notebook
`Baseline CLIP/` mencari gambar pada direktori `images/` secara default.

## Urutan Eksekusi

1. (Sudah disediakan) Hasil split data di `train_df.csv` / `test_df.csv`.
2. (Sudah disediakan) Cache Gemma-3 + BERTScore di `short_gemma_bertscore.csv`.
3. (Opsional, hanya untuk Baseline CLIP) Unduh gambar:
   `python scripts/download_images.py`
4. Jalankan `Baseline CLIP/` untuk reproduksi baseline.
5. Jalankan `Kriteria Tekstual/` dan `Kriteria Multimodal/` untuk metode usulan.
6. Jalankan folder `Ablasi 1` sampai `Ablasi 4` dan `Ablasi Pooling`.
7. Jalankan `Uji Signifikansi (t-test)/` untuk validasi statistik.

> Notebook urutan 5–7 dapat dijalankan **tanpa** menyelesaikan langkah 3
> karena tidak memerlukan gambar mentah.

## Dependensi Utama

Daftar lengkap dengan versi terkunci ada di `environment.yml` dan
`requirements.txt`. Hanya library yang benar-benar di-import oleh notebook
yang dicantumkan; library lain di lingkungan pengembangan penulis (mis.
`accelerate`, `spacy`, `peft`, `easyocr`) tidak relevan untuk repo ini.

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
| `tabulate`             | 0.10.0         | Render tabel hasil grid search (`pandas.to_markdown`)           |
| `ipykernel`            | 7.2.0          | Kernel Jupyter untuk eksekusi notebook                          |
