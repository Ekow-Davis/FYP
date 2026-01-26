# Brain Tumor & Alzheimer’s MRI Classification using LEAD-CNN

## Project Overview

This project implements and extends the methodology described in the research article:

> **“LEAD-CNN: A lightweight convolutional neural network with dimension reduction for multi-class brain tumor classification using MRI images”**
> Springer – International Journal of Machine Learning and Cybernetics
> [https://link.springer.com/article/10.1007/s13042-025-02637-6](https://link.springer.com/article/10.1007/s13042-025-02637-6)

The goal is to reproduce the preprocessing pipeline and model design proposed in the paper and apply it to:

1. **Brain tumor MRI classification** (primary task)
2. **Alzheimer’s disease MRI classification** (cross-dataset generalization study)

The project follows a rigorous data preparation workflow:

> Raw MRI images → Cleaning & standardization → Dataset splitting → Augmentation → Model training & evaluation

This ensures reproducibility, fairness in model comparison, and strong generalization.

---

## Dataset Sources

### Brain Tumor Dataset (Primary)

Kaggle – Brain Tumor MRI Dataset
[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Training](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Training)

Contains 7,023 MRI images in four classes:

* Glioma
* Meningioma
* Pituitary tumor
* Normal

Original image sizes: 512×512 and 236×236

---

### Alzheimer’s Dataset (Secondary – Future Extension)

An MRI-based Alzheimer’s dataset will be introduced later to:

* Evaluate cross-domain performance of LEAD-CNN
* Test robustness of preprocessing and feature extraction pipeline
* Validate suitability of the architecture for general MRI classification tasks

---

## Methodology Summary (Based on the Article)

The implemented pipeline mirrors the research paper:

### 1. Image Standardization

All MRI images are resized to **224 × 224 pixels** to:

* Ensure compatibility with CNN architectures
* Reduce computational cost
* Maintain uniform input dimensions

---

### 2. Dataset Distribution

Images are merged and re-split into:

* Training set (≈ 80%)
* Validation set (≈ 10%)
* Test set (≈ 10%)

Splitting is performed per class to preserve class balance.

---

### 3. Preprocessing

Using Keras/TensorFlow compatible formatting:

* Pixel normalization
* Batch loading
* Directory-based class structure
* Generator-ready formatting

---

### 4. Data Augmentation

To reduce overfitting and increase generalization:

* 90° rotation
* Horizontal flipping

Augmentation is performed **after cleaning** and stored separately to preserve dataset integrity.

---

### 5. Feature Extraction & Classification

Using the proposed **LEAD-CNN architecture**:

* Multiple convolution layers (3×3 kernels)
* LeakyReLU activation (prevents dead neurons)
* Max pooling layers
* Dropout regularization
* Custom dimension reduction block
* Fully connected layers with Softmax output

This allows automatic extraction of both low-level and high-level MRI features without manual engineering.

---

## Project Folder Structure

```
final-year-project/
│
├── data/
│   ├── raw_data/
│   │   ├── brain_tumor/
│   │   │   ├── Training/
│   │   │   │   ├── glioma/
│   │   │   │   ├── meningioma/
│   │   │   │   ├── pituitary/
│   │   │   │   └── notumor/
│   │   │   └── Testing/
│   │   │       ├── glioma/
│   │   │       ├── meningioma/
│   │   │       ├── pituitary/
│   │   │       └── notumor/
│   │   │
│   │   └── alzheimer/
│   │       └── (future dataset – raw MRI images)
│   │
│   ├── cleaned_data/
│   │   ├── train/
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── pituitary/
│   │   │   └── notumor/
│   │   │
│   │   ├── val/
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── pituitary/
│   │   │   └── notumor/
│   │   │
│   │   └── test/
│   │       ├── glioma/
│   │       ├── meningioma/
│   │       ├── pituitary/
│   │       └── notumor/
│   │
│   ├── augmented_data/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │       └── (same class folders as cleaned_data)
│   │
│   └── dataset_stats/
│       └── split_summary.csv
│
├── notebooks/
│   ├── 01_data_merge_and_cleaning.ipynb
│   ├── 02_dataset_splitting.ipynb
│   ├── 03_data_augmentation.ipynb
│   ├── 04_dataset_verification.ipynb
│   └── 
│
├── models/
│   ├── lead_cnn/
│   │   ├── architecture.py
│   │   ├── dimension_reduction_block.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── saved_weights/
│   │
│   ├── baselines/
│   │   ├── resnet/
│   │   ├── densenet/
│   │   ├── xception/
│   │   └── mobilenet/
│   │
│   └── experiments/
│       └── logs/
│
├── scripts/
│   ├── clean_dataset.py
│   ├── augment_dataset.py
│   ├── split_dataset.py
│   └── visualize_samples.py
│
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── plots/
│   └── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Folder Descriptions

### `data/raw_data/`

Contains untouched original datasets:

* Brain tumor MRI data from Kaggle
* Future Alzheimer’s MRI dataset

No preprocessing is done here.

---

### `data/cleaned_data/`

Contains:

* Resized images (224×224)
* Merged training + testing data
* Newly split into train/val/test

This dataset is used as the **baseline training dataset**.

---

### `data/augmented_data/`

Contains artificially expanded datasets generated from `cleaned_data` using:

* 90° rotation
* Horizontal flipping

Used to train more robust models.

---

### `notebooks/`

Jupyter notebooks for:

* Dataset merging
* Cleaning & resizing
* Splitting
* Augmentation
* Visual verification
* Statistics

Primary environment for preprocessing experiments.

---

### `models/`

Contains:

* LEAD-CNN implementation
* Pretrained baseline models
* Saved weights
* Training and evaluation scripts

---

### `scripts/`

Reusable command-line utilities for:

* Cleaning datasets
* Splitting datasets
* Augmentation
* Visualization

---

### `results/`

Stores:

* Accuracy, precision, recall, F1 scores
* Confusion matrices
* Training curves
* Final evaluation reports

---

## Python Environment

* **Python version:** 3.12
* Image processing: OpenCV
* Data handling: NumPy, Pandas
* Visualization: Matplotlib, Seaborn
* Utilities: tqdm, scikit-learn

---

## requirements.txt (Example)

```
python==3.12
numpy==2.2.6
pandas==2.2.2
opencv-python==4.10.0.84
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
tensorflow==2.16.1
tqdm==4.66.4
```

---

## Planned Extensions

* Train LEAD-CNN on Alzheimer’s MRI data
* Compare performance across MRI domains
* Test generalization without architecture modification
* Evaluate robustness of preprocessing and augmentation strategy
* Deploy trained model as a lightweight clinical inference tool* (TBD)

---

## Research Motivation

This project aims to:

* Validate a lightweight CNN architecture for medical imaging
* Demonstrate reproducible preprocessing pipelines
* Explore cross-disease MRI classification
* Reduce reliance on manual feature engineering
* Support resource-constrained medical environments

---

## Citation
This Project is inspired by and if you use this project or methodology, cite:

> Khan, S. U. R., Asif, S., Bilal, O., & Rehman, H. U.
> *LEAD-CNN: A lightweight convolutional neural network with dimension reduction for multi-class brain tumor classification using MRI images.*
> International Journal of Machine Learning and Cybernetics, 2025.

---