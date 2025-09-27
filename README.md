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
├── testing/                   # Testing Model in Website
│   └── app.py                 # Script for testing model
├── requirement.txt            # Requirement Library that used in this Project 
└── README.md                  # Documentation Project

````

---

## 📊 Dataset
Dataset that used in this project:  
**[Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)** (5 class rice):
- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

Dataset will automatically downloaded and extracted by script `download_data.py`:
```bash
python download_data.py
````

---

## 🏗️ Model Architecture

Main model using  **MobileNetV2** that already pre-trained in ImageNet,  then added some custom layers:

* Base Model: `MobileNetV2 (include_top=False)`
* Global Average Pooling
* Dense Layer (512 → 256 units) + BatchNorm + Dropout
* Output Layer: Softmax dengan 5 kelas

Visualization summary model:

```
Input → MobileNetV2 → GAP → Dense(512) → Dense(256) → Softmax(5)
```

---

## 🔄 Data Preprocessing & Augmentation

Pipeline preprocessing (in `preprocessed.py`):

* Resize → 224x224
* Normalization (0–1)
* Augmentation:

  * Random Flip
  * Random Rotation
  * Random Zoom
  * Random Height & Width shift

---

## 🚀 Training

Training running used script  `train.py`:

```bash
python training/train.py --data_dir data/raw/Rice_Image_Dataset --epochs 20 --batch_size 32
```

### Callbacks that used:

* **ModelCheckpoint** → saved best model  (`best_model.keras`)
* **EarlyStopping** → stop training if val_loss not improving 
* **ReduceLROnPlateau** → reduce the automatic learning rate
* **CSVLogger** → log training to file `.csv`

Output training saved in folder `artifacts/`.

---

## 📈 Result Training 

* Training Accuracy: 99%
* Validation Accuracy: 97–98%
* Test Accuracy: ~95%

Grafik training & validation loss/accuracy it's in the  `artifacts/plots/training_history.png`.

---

## 🧪 Testing

used `app.py` to evaluate model in dataset test:

```bash
python app.py --model artifacts/models/best_model.keras --data_dir data/raw/Rice_Image_Dataset
```


## ⚙️ Requirements

Install dependency:

```bash
pip install -r requirements.txt
```

Main Library in this project:

* TensorFlow
* scikit-learn
* matplotlib
* seaborn
* numpy

---
```

