# CNN-XGBoost Stacking Ensemble & XAI for Image Classification

This repository contains a robust, PyTorch-based image classification pipeline (specifically tailored for cucumber image datasets). It leverages a stacking ensemble of state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers, combined with XGBoost and extensive Explainable AI (XAI) techniques.

## 🌟 Key Features

* **State-of-the-Art Models**: Utilizes `timm` to extract features using `CAFormerS18`, `TinyViT`, `DaViTTiny`, and `EfficientNetV2`.
* **Advanced Ensembling**: Compares multiple ensemble strategies:
    * CNN Weighted Soft Voting
    * XGBoost Early Fusion (Feature Concatenation)
    * Custom Meta-Learning (Logistic Regression Stacking)
* **Optimizer Optimization**: Uses the `Lion` optimizer (with a fallback to AdamW) and Automatic Mixed Precision (AMP) for faster, memory-efficient training.
* **Comprehensive XAI Suite**: Visualizes model decision-making using:
    * Grad-CAM & Score-CAM
    * LIME
    * Integrated Gradients (Captum)
    * RISE (Randomized Input Sampling for Explanation)
    * SHAP (TreeExplainer for XGBoost feature importance)

## 🚀 Installation & Setup

1. **Clone the repository:**





2. **Create a virtual environment (optional but recommended):**

   python -m venv myenv
   source myenv/bin/activate  / On Windows use `myenv\Scripts\activate`
 

3. **Install dependencies:**

   pip install -r requirements.txt


## 📂 Dataset Structure

The code expects a local dataset formatted for `torchvision.datasets.ImageFolder`. You will need to update the `merged_dir` variable in the notebook to point to your local dataset path.

```text
/Final_merged
    ├── Class_1/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── Class_2/
    │   ├── img1.jpg
    │   └── ...
```

## 🧠 Saved Artifacts

During execution, the notebook automatically saves trained models, scalers, and evaluation metrics into a `saved_ensemble_models_cucumber/` directory (ignored by git by default). This includes:
* PyTorch `.pth` weights for the CNNs
* XGBoost `.json` models
* Pickled Meta-LR and StandardScalers

## 📊 Results & XAI Output

The notebook outputs a complete dashboard of results ranking the individual models against the ensemble methods. It also generates side-by-side visual comparisons of how different XAI methods interpret the model's focus areas for specific predictions.