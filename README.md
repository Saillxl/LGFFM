# LGFFM: A Localized and Globalized Frequency Fusion Model for Ultrasound Image Segmentation

[![Paper](https://img.shields.io/badge/IEEE-TMI%202025-blue)](https://ieeexplore.ieee.org/document/11129883)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](./LICENSE)

📄 Published in **IEEE Transactions on Medical Imaging (TMI)**  
🔗 [Paper Link](https://ieeexplore.ieee.org/document/11129883)  

---

## 📖 Introduction

![framework](./LGFFM/framework.png)

Accurate segmentation of ultrasound images plays a critical role in disease screening and diagnosis. Recently, neural network-based methods have shown great promise, but still face challenges due to the inherent characteristics of ultrasound images—such as **low resolution, speckle noise, and artifacts**.

Moreover, ultrasound segmentation spans a wide range of scenarios, including **organ segmentation** (e.g., cardiac, fetal head) and **lesion segmentation** (e.g., breast cancer, thyroid nodules), which makes the task highly diverse and complex. Existing methods are often tailored for specific cases, limiting their flexibility and generalization.

To address these challenges, we propose **LGFFM (Localized and Globalized Frequency Fusion Model)**, a novel framework for ultrasound image segmentation:

- **Parallel Bi-Encoder (PBE):** integrates Local Feature Blocks (LFB) and Global Feature Blocks (GLB) to enhance feature extraction.  
- **Frequency Domain Mapping Module (FDMM):** captures texture information, particularly high-frequency details like edges.  
- **Multi-Domain Fusion (MDF):** effectively integrates features across different domains for more robust segmentation.  

We evaluate LGFFM on **eight public ultrasound datasets across four categories**. Results show that LGFFM **outperforms state-of-the-art methods** in both segmentation accuracy and generalization performance.  

---

## 📂 Clone Repository

```bash
git clone https://github.com/Saillxl/LGFFM.git
cd LGFFM/
```

---

## 📑 Dataset Preparation

The dataset should follow the format below. 

```
**Example: BUSI dataset**
BUSI/
├─ img.yaml─ - /data/project/BUSI/img_dir/malignant (17).png
|            - /data/project/BUSI/img_dir/benign (42).png 
|            - /data/project/BUSI/img_dir/malignant (109).png
├─ ann.yaml─ - /data/project/BUSI/ann_dir/malignant (17).png
|            - /data/project/BUSI/ann_dir/benign (42).png 
|            - /data/project/BUSI/ann_dir/malignant (109).png 
```
⚠️Note: Masks are binary (0 and 1).

---

## ⚙️ Requirements
We recommend creating a clean environment:

```
conda create -n LGFFM python=3.10
conda activate LGFFM
pip install -r requirements.txt
```

## 🚀 Training
1. Download pre-trained weights from the official (not 2.1) [SAM2 large repository](https://github.com/facebookresearch/sam2?tab=readme-ov-file). Place the weights in the "checkpoints" folder. If you want to place it elsewhere, modify the parameter hiera_path in train.py.
2. Run train.py
```
python train
```

## 🧪 Testing
