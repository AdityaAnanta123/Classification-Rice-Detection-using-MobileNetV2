# 🍚 Rice Detection with MobileNetV2
This project is an implementation of **Rice Image Classification** using **TensorFlow & Keras** with the **MobileNetV2** architecture.
The main goal of this project is to detect rice types based on images with high accuracy using deep learning.
---

## 📂 Project Structure
```

├── data/                      # Dataset (raw & processed)
│   └── raw/                   # Dataset raw (Rice Image Dataset)
│   └── processed/             # Dataset preprocessed
│   └── download_data.py       # Import data from kaggle 
├── preprocessing/             # Modul preprocessing
│   └── preprocessed.py        # Data pipeline & augmentation
├── model/                     # Model deep learning
│   └── mobilenet_model.py     # Definition architectur MobileNetV2
├── training/                  # Script for training
│   ├── train.py               # Main training pipeline
│   └── utils.py               # Helper (plot, save class map, dll)
├── artifacts/                 # Result training & evaluasi
│   ├── models/                # Best model & final model
│   ├── plots/                 # Grafik training history
│   └── class_map.json         # Mapping kelas
├── test.py                    # Script for testing model
└── README.md                  # Dokumentasi proyek

````

---

## 📊 Dataset
Dataset yang digunakan:  
**[Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)** (5 kelas beras):
- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

Dataset akan otomatis di-download dan diekstrak oleh script `data_download.py`:
```bash
python data_download.py
````

---

## 🏗️ Model Architecture

Model utama menggunakan **MobileNetV2** yang sudah ter-pretrained di ImageNet, lalu ditambahkan beberapa lapisan kustom:

* Base Model: `MobileNetV2 (include_top=False)`
* Global Average Pooling
* Dense Layer (512 → 256 units) + BatchNorm + Dropout
* Output Layer: Softmax dengan 5 kelas

Visualisasi summary model:

```
Input → MobileNetV2 → GAP → Dense(512) → Dense(256) → Softmax(5)
```

---

## 🔄 Data Preprocessing & Augmentation

Pipeline preprocessing (di `preprocessed.py`):

* Resize → 224x224
* Normalization (0–1)
* Augmentation:

  * Random Flip
  * Random Rotation
  * Random Zoom
  * Random Height & Width shift

---

## 🚀 Training

Training dijalankan menggunakan script `train.py`:

```bash
python training/train.py --data_dir data/raw/Rice_Image_Dataset --epochs 20 --batch_size 32
```

### Callbacks yang digunakan:

* **ModelCheckpoint** → simpan model terbaik (`best_model.keras`)
* **EarlyStopping** → hentikan training jika val_loss tidak membaik
* **ReduceLROnPlateau** → turunkan learning rate otomatis
* **CSVLogger** → log training ke file `.csv`

Output training tersimpan di folder `artifacts/`.

---

## 📈 Hasil Training (Contoh)

* Training Accuracy: 99%
* Validation Accuracy: 97–98%
* Test Accuracy: ~95%

Grafik training & validation loss/accuracy ada di `artifacts/plots/training_history.png`.

---

## 🧪 Testing

Gunakan `test.py` untuk evaluasi model di dataset test:

```bash
python test.py --model artifacts/models/best_model.keras --data_dir data/raw/Rice_Image_Dataset
```

Hasil evaluasi:

* Confusion Matrix
* Classification Report (precision, recall, f1-score)
* Test Accuracy

---

## ⚙️ Requirements

Install dependency:

```bash
pip install -r requirements.txt
```

Daftar utama:

* TensorFlow
* scikit-learn
* matplotlib
* seaborn
* numpy

---

## 📌 Catatan

* Gunakan GPU untuk mempercepat training (MobileNetV2 cukup ringan).
* Jika dataset besar, batch_size bisa diperkecil (misalnya 16).
* Model akhir tersimpan di:

  * `artifacts/models/final_model.keras`
  * `artifacts/models/best_model.keras`

---

## 👨‍💻 Author

Proyek ini dikembangkan sebagai bagian dari portofolio deep learning computer vision untuk klasifikasi gambar.

```

---

Mau saya bikinkan juga **`requirements.txt`** biar langsung tinggal `pip install -r requirements.txt` untuk setup?
```
