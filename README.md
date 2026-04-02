# Hybrid CNN-Attention Model for COPD Detection from Chest X-Rays

## Description

This repository contains the complete implementation of a hybrid attention-augmented deep convolutional neural network for binary classification of chest X-ray images as **Normal** or **COPD-affected**. The proposed model combines a pretrained ResNet-18 backbone with Squeeze-and-Excitation (SE) blocks, Convolutional Block Attention Modules (CBAM), and a lightweight self-attention block for enhanced spatial and channel feature selection.

A SimpleCNN baseline model is also included for comparison.

This code accompanies the paper:
> *"A Hybrid Convolutional and Attention-Based Model for COPD Recognition Using Chest Radiographs"*
> Cheemalamarri Vishnupriya, R. Rathna, S. Anubha Pearline — VIT Chennai

---

## Dataset Information

This project uses the **x-ray-lung-diseases-images-9-classes** dataset, reorganized into two classes: Normal and COPD (Doenças Pulmonares Obstrutivas).

- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes
- **Classes used:** `00 Normal Anatomy` and `04 Doenças Pulmonares Obstrutivas`
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Total samples:** ~1,984 images (1,340 Normal + 644 COPD)

> Note: The dataset must be downloaded separately from Kaggle and organized into `train/`, `val/`, and `test/` subfolders before running the code.

---

## Code Information

### Hybrid Attention Model (Proposed)
| File | Description |
|---|---|
| `models_hy.py` | Defines the HybridAttentionModel with SE, CBAM, and Self-Attention blocks |
| `train_hy.py` | Training loop for the hybrid model with early stopping and LR scheduling |
| `evaluate_hy.py` | Evaluation: confusion matrix, classification report, sample predictions |
| `mainhy.py` | End-to-end pipeline runner for the hybrid model |
| `utils_hy.py` | EarlyStopping class and model analysis utilities |

### Baseline Model (SimpleCNN)
| File | Description |
|---|---|
| `train.py` | SimpleCNN architecture definition and training loop |
| `evaluate.py` | Evaluation utilities for the baseline model |
| `main.py` | End-to-end pipeline runner for the SimpleCNN baseline |

---

## Requirements

Python 3.8+ is recommended. Install all dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
tqdm>=4.62.0
Pillow>=9.0.0
```

---

## Usage Instructions

### 1. Prepare the dataset

Download the dataset from Kaggle and organize it as follows:

```
data_processed/
├── train/
│   ├── Normal/
│   └── COPD/
├── val/
│   ├── Normal/
│   └── COPD/
└── test/
    ├── Normal/
    └── COPD/
```

### 2. Run the Hybrid Attention Model (Proposed)

```bash
python mainhy.py
```

### 3. Run the SimpleCNN Baseline

```bash
python main.py
```

### 4. Configure paths

Before running, update the data paths inside `mainhy.py` and `main.py`:

```python
data_root = "/path/to/your/data_processed"
```

Outputs (training curves, confusion matrix, classification report, model checkpoints) will be saved automatically in a timestamped folder under `/outputs/`.

---

## Methodology

1. **Preprocessing:** Images resized to 224×224, converted to RGB, normalized with ImageNet mean/std. Training augmentations include random horizontal flip, rotation (±10°), and color jitter.
2. **Feature Extraction:** Pretrained ResNet-18 backbone extracts hierarchical feature maps (output: 512×7×7).
3. **Attention Refinement:** Features are sequentially refined by SE block (channel recalibration), CBAM (joint channel + spatial attention), and a lightweight self-attention block (long-range context).
4. **Classification:** Global average pooling followed by a two-layer fully connected head with dropout produces Normal/COPD predictions.
5. **Training:** Adam optimizer, ReduceLROnPlateau scheduler, early stopping (patience=7), fixed random seed (42) for reproducibility.
6. **Evaluation:** Accuracy, Precision, Recall, F1-score, confusion matrix on a held-out test set.

---

## Citations

If you use this code, please cite:

> Cheemalamarri Vishnupriya, R. Rathna, S. Anubha Pearline. "A Hybrid Convolutional and Attention-Based Model for COPD Recognition Using Chest Radiographs." PeerJ Computer Science, 2025.

Dataset citation:
> Fernando Feltrin. X-ray Lung Diseases Images (9 classes). Kaggle. https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Contact

For questions regarding this work, please contact:

- Cheemalamarri Vishnupriya (First Author)
- R. Rathna (Corresponding Author): rathna.r@vit.ac.in  
- S. Anubha Pearline: anubhapearline.s@vit.ac.in