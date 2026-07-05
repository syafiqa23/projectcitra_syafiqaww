# Sistem Klasifikasi Tingkat Degradasi Vegetasi Gambut Menggunakan Citra UAV dan Deep Learning

## Deskripsi Proyek

Proyek ini merupakan implementasi penelitian klasifikasi tingkat degradasi vegetasi gambut menggunakan citra UAV (Unmanned Aerial Vehicle) dan metode Deep Learning berbasis Convolutional Neural Network (CNN).

Penelitian membandingkan tiga model transfer learning:

* MobileNetV2
* ResNet50
* ResNet101

Model terbaik kemudian diimplementasikan ke dalam aplikasi web menggunakan Streamlit sehingga pengguna dapat melakukan prediksi langsung terhadap citra UAV.

---

# Struktur Folder

```text
PROJECTCITRA/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── bare/
│   ├── heavily_grazed/
│   └── softly_grazed/
│
├── models/
│   ├── mobilenetv2_best.pth
│   ├── resnet50_best.pth
│   └── resnet101_best.pth
│
├── results/
│
├── compare_models.py
├── train_mobilenet.py
├── train_resnet50.py
├── train_resnet101.py
└── requirements.txt
```

---

# Dataset

Dataset terdiri dari 3 kelas vegetasi gambut:

| Kelas          | Deskripsi                                |
| -------------- | ---------------------------------------- |
| bare           | Area vegetasi sangat rendah atau terbuka |
| heavily_grazed | Area vegetasi dengan degradasi tinggi    |
| softly_grazed  | Area vegetasi dengan degradasi rendah    |

Jumlah dataset:

```text
Total : 3000 citra UAV

bare             : 1000
heavily_grazed   : 1000
softly_grazed    : 1000
```

---

# Arsitektur Model

Penelitian menggunakan pendekatan Transfer Learning dengan model:

1. MobileNetV2
2. ResNet50
3. ResNet101

Tahapan pelatihan:

* Data Augmentation
* Transfer Learning
* Fine Tuning
* Evaluasi Model
* Perbandingan Performa Model

---

# Hasil Penelitian

| Model       | Accuracy |
| ----------- | -------- |
| MobileNetV2 | 94.50%   |
| ResNet50    | 94.67%   |
| ResNet101   | 93.33%   |

Model terbaik:

🏆 **ResNet50**

---

# Persyaratan Sistem

## Minimum

* Windows 10 / Windows 11
* Python 3.11
* RAM 8 GB
* Storage 5 GB

## Direkomendasikan

* RAM 16 GB
* SSD
* NVIDIA RTX Series

---

# Instalasi Menggunakan Miniconda

## 1. Clone atau Download Project

Letakkan project pada folder:

```text
D:\projectcitra
```

atau lokasi lain sesuai kebutuhan.

---

## 2. Membuat Environment

Buka terminal:

```bash
conda create -n peatland_torch python=3.11 -y
```

Aktifkan:

```bash
conda activate peatland_torch
```

---

# Instalasi Untuk Pengguna GPU NVIDIA

Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

Verifikasi:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Output:

```text
True
```

Cek GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Contoh:

```text
NVIDIA GeForce RTX 3060 Laptop GPU
```

---

# Instalasi Untuk Pengguna Tanpa GPU NVIDIA

Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

Verifikasi:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Output:

```text
False
```

Hal ini normal apabila laptop tidak memiliki GPU NVIDIA.

Aplikasi dan model tetap dapat digunakan.

---

# Install Library Pendukung

Install seluruh dependency:

```bash
pip install -r requirements.txt
```

Apabila requirements.txt belum lengkap:

```bash
pip install streamlit
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install pillow
pip install scikit-learn
```

---

# Verifikasi Instalasi

Jalankan:

```bash
python -c "import torch; print(torch.__version__)"
```

Contoh:

```text
2.5.1
```

---

# Training Model

## MobileNetV2

```bash
python train_mobilenet.py
```

Output:

```text
models/mobilenetv2_best.pth
```

---

## ResNet50

```bash
python train_resnet50.py
```

Output:

```text
models/resnet50_best.pth
```

---

## ResNet101

```bash
python train_resnet101.py
```

Output:

```text
models/resnet101_best.pth
```

---

# Membandingkan Model

Jalankan:

```bash
python compare_models.py
```

Script ini digunakan untuk membandingkan:

* Accuracy
* Precision
* Recall
* F1-Score

ketiga model yang digunakan dalam penelitian.

---

# Menjalankan Frontend

Aktifkan environment:

```bash
conda activate peatland_torch
```

Masuk ke folder project:

```bash
cd D:\projectcitra
```

Jalankan aplikasi:

```bash
python -m streamlit run app/app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

---

# Cara Menggunakan Aplikasi

1. Buka browser.
2. Akses:

```text
http://localhost:8501
```

3. Upload citra UAV.
4. Tunggu proses inferensi.
5. Sistem akan menampilkan:

* Kelas Prediksi
* Confidence Score
* Tingkat Degradasi
* Probabilitas Tiap Kelas

---

# Interpretasi Tingkat Degradasi

| Kelas          | Tingkat Degradasi |
| -------------- | ----------------- |
| Bare           | Tinggi            |
| Heavily Grazed | Sedang            |
| Softly Grazed  | Rendah            |

---

# Menjalankan Tanpa Training

Apabila hanya ingin mencoba aplikasi:

Pastikan folder berikut tersedia:

```text
models/
├── mobilenetv2_best.pth
├── resnet50_best.pth
└── resnet101_best.pth
```

Kemudian jalankan:

```bash
python -m streamlit run app/app.py
```

Training ulang tidak diperlukan.

---

# Pengujian Prediksi

Untuk memvalidasi model:

1. Pilih beberapa gambar dari dataset.
2. Upload ke aplikasi.
3. Bandingkan label asli dengan hasil prediksi.

Contoh:

| Nama File | Label Asli     | Prediksi       |
| --------- | -------------- | -------------- |
| img01.png | bare           | bare           |
| img02.png | heavily_grazed | heavily_grazed |
| img03.png | softly_grazed  | softly_grazed  |

---

# Troubleshooting

## Error: No module named torch

Pastikan environment aktif:

```bash
conda activate peatland_torch
```

Verifikasi:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Error: Model tidak ditemukan

Pastikan file model tersedia:

```text
models/
├── mobilenetv2_best.pth
├── resnet50_best.pth
└── resnet101_best.pth
```

---

## Error: CUDA tidak terdeteksi

Jalankan:

```bash
nvidia-smi
```

Kemudian:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Apabila hasil:

```text
False
```

maka aplikasi tetap dapat digunakan menggunakan CPU.

---

## Training Sangat Lambat

Kurangi batch size pada script training.

Contoh:

```python
BATCH_SIZE = 16
```

ubah menjadi:

```python
BATCH_SIZE = 8
```

atau:

```python
BATCH_SIZE = 4
```

---

# Untuk Dosen Penguji

Apabila hanya ingin mencoba hasil penelitian:

1. Install Python atau Miniconda.
2. Install dependency dari requirements.txt.
3. Jalankan:

```bash
python -m streamlit run app/app.py
```

4. Upload citra UAV.

Training ulang tidak diperlukan karena model terbaik telah tersedia pada folder:

```text
models/
```

---

# Teknologi Yang Digunakan

* Python 3.11
* PyTorch
* TorchVision
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Pillow

---

# Lisensi

Proyek ini dibuat untuk keperluan akademik, penelitian, dan tugas akhir mahasiswa.
