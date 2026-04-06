# Plant Disease Classification using Transformer Fusion

## Project Overview

This project focuses on the automated identification of plant diseases to assist in early detection and agricultural management. By leveraging advanced deep learning techniques, the system analyzes leaf images to distinguish between healthy plants and those affected by various pathogens.

The framework is designed to classify **38 different plant species and disease conditions** using a hybrid deep learning approach that combines multiple models, feature engineering, and transformer-based fusion.

---

##  Dataset

The project utilizes the **PlantVillage dataset**, a comprehensive collection of plant leaf images used for training and evaluation.

- **Dataset Link:** [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Total Images:** Approximately 87,000
- **Classes:** 38 (healthy and diseased)
- **Distribution:** Around 1,800 images per class

> **Note:** The dataset is not included in this repository due to size constraints. Download it from the link above and place it in the appropriate directory before running the project. Also note to combine the downloaded folder since it is downloaded as two folders (train,valid).

---

## Key Features

- **Multi-model integration** using:
  - MobileNetV3
  - GoogLeNet
  - DenseNet121
  - ConvNeXt Small
- **Texture and structural analysis** using Gray Level Co-occurrence Matrix (GLCM)
- **Ensemble learning** using CatBoost classifiers
- **Advanced feature fusion** using a Transformer-based approach to model interactions between feature representations

---

## Project Structure

```
Plant_Disease_DAC_Project/
│
├── models/           # Already Trained model weights (all models except for Convnext Small)
├── notebooks/        # Complete Pipeline and Experiments
├── utils/            # Utility Functions
├── README.md
├── requirements.txt
```

> The following components are excluded due to large file size:
> - `dataset/`
> - `features/`
> - Large model files (e.g., ConvNeXt)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Prabhunotfound/Plant_Disease.git
cd Plant_Disease_Transformer
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Execution Pipeline

Navigate to the `notebooks/` directory and execute the notebooks in the following order:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `Preprocessing.ipynb` | Image resizing and noise reduction |
| 2 | `Mobilenetv3.ipynb` | Train MobileNetV3 model |
| 3 | `GoogleNet.ipynb` | Train GoogLeNet model |
| 4 | `Convnext.ipynb` | Train ConvNeXt Small model |
| 5 | `DenseNet121.ipynb` | Train DenseNet121 model |
| 6 | `Feature_Extraction.ipynb` | Generate feature vectors |
| 7 | `CatBoost_Ensemble_v1.ipynb` | First ensemble model |
| 8 | `CatBoost_Ensemble_v2.ipynb` | Second ensemble model |
| 9 | `Transformer_fusion.ipynb` | Final feature fusion and classification |

---

## Repository Notes

- Large files such as datasets, extracted features, and heavy model weights are intentionally excluded
- Ensure the dataset is downloaded and correctly placed before execution
- The pipeline depends on sequential execution of notebooks
