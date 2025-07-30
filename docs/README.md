
# SABE-YOLO: Structure-Aware and Boundary-Enhanced YOLO for Weld Seam Instance Segmentation

This repository contains the official implementation of **SABE-YOLO**, a novel instance segmentation model designed for accurate and efficient weld seam segmentation in complex industrial environments. The model integrates **Structure-Aware Fusion Module (SAFM)** and **Boundary-Enhanced Aggregation Module (C2f-BEAM)**, and introduces an **Inner-MPDIoU loss** to better handle elongated structures and fuzzy boundaries. This project is built upon the Ultralytics YOLOv8 framework. We thank the Ultralytics team for their excellent work.

🔗 Ultralytics GitHub: https://github.com/ultralytics/ultralytics

---

## 📌 Highlights

- 🔺 Accurate segmentation of complex weld structures
- 📦 Lightweight: 6.6M parameters, 18.3 GFLOPs, 127 FPS
- 📐 Generalizes to crack segmentation with zero-shot inference

---

## 📁 Contents

- `ultralytics/nn/Addmodules/`: Core modules (SAFM->Starnet.py, C2f-BEAM->MEEIEM.py)
- `aug-pipline.py`: Custom image augmentations
- `sabe-yolo.pt`: Pre-trained weights
- `test_images/`: Five weld images for inference demo
- `test.py`: Run segmentation on demo images

---

## 🧩 Environment Setup

```bash
conda create -n sabe-yolo python=3.8 -y
conda activate sabe-yolo
pip install -r requirements.txt
```

Required packages include `torch`, `opencv-python`, `numpy`, `albumentations`, `matplotlib`, etc.

---

## 🔧 Pretrained Weights

Download the trained model weights:

```
sabe-yolo.pt
```

---

## 🧪 Demo Inference

We provide **5 weld seam images** in the `test_images/` folder for visualization.

### Run inference:

```bash
python test.py
```

- Results will be saved to `weld_seg1/weld_seg1/`

---

## 🧠 Data Augmentation Pipeline

The full data augmentation strategy used during training is implemented in `aug_pipeline.py`. It includes:

- Gaussian noise injection
- Brightness/contrast adjustment
- Random cropping and flipping
- Rotation, translation, cutout

---

## 🧾 Dataset Notes

> ⚠️ The original weld dataset is not publicly released due to industrial confidentiality.


---

## 📊 Evaluation

| Model           | AP(50–95) | Params | GFLOPs | FPS  |
|----------------|-----------|--------|--------|------|
| SABE-YOLO       | **46.3**  | 6.6M   | 18.3   | 127  |
| YOLOv8n-seg     | 43.3      | 3.3M   | 12   | 213  |
| Mask DINO       | 36.9      | 52M    | 280    | 15   |



## 📬 Contact

For questions, please contact:  
📧 wenruiguet@126.com  

