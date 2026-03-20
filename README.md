# ✏️ QuickDraw Google - CNN Hand Drawing Recognition
> Real-time hand drawing recognition using CNN + MediaPipe webcam

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Train a CNN model on the [Google QuickDraw dataset](https://quickdraw.withgoogle.com/data) to recognize 15 hand-drawn categories. The system uses **MediaPipe** to track the index finger via webcam, letting users draw in the air and get real-time predictions.

### Model Results
![TensorBoard](assets/tensorboard.png)

---

## 🗂️ Project Structure

```
.
├── model_CNN_QuickDraw.py   # CNN architecture definition
├── quickDrawDataset.py      # Custom PyTorch Dataset
├── train_Quick_Draw.py      # Training script with TensorBoard logging
├── inference_QD.py          # Single image inference
├── paint.py                 # Real-time webcam drawing app (MediaPipe)
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | Custom CNN (2 Conv + 3 FC layers) |
| Dataset | Google QuickDraw (numpy bitmap) |
| Hand Tracking | MediaPipe Hand Landmarker |
| Framework | PyTorch |
| Logging | TensorBoard |
| Language | Python 3.10 |

---

## 🎯 Categories (15 classes)

`Airplane` `Angel` `Apple` `Axe` `Bat` `Book` `Boomerang` `Camera` `Cup` `Fish` `Flower` `Mushroom` `Radio` `Sun` `Sword`

---

## 🚀 How to Run

### 1. Clone repo
```bash
git clone https://github.com/TrieuNguyenTai/quickdraw-cnn.git
cd quickdraw-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download QuickDraw `.npy` files and place them in `dataset_Quick_Draw/`:
- [Google QuickDraw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap)

### 4. Train model
```bash
python train_Quick_Draw.py
```

### 5. Run real-time webcam app
```bash
python paint.py
```

### 6. Single image inference
```bash
python inference_QD.py -p "test/sun.png" -t "./train_model_QuickDraw/quickdraw"
```

### 7. Or run with Docker
```bash
docker build -t quickdraw-cnn .
docker run quickdraw-cnn
```

---

## 🎮 Controls (paint.py)

| Key | Action |
|-----|--------|
| `S` | Start / Stop drawing |
| `P` | Predict current drawing |
| `D` | Clear canvas |
| `C` | Change challenge |
| `ESC` | Quit |

---

## 📷 Demo

### Webcam Drawing App
![Demo](assets/demo.png)

### Demo Video
> *(Add demo video link here — YouTube / Google Drive)*

---

## 👤 Author

**Trieu Nguyen Tai** — Hanoi University of Mining and Geology, Faculty of IT