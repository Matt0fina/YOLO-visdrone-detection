# Design, Optimization, and Comparative Evaluation of Modern YOLO Models
**Course:** CMPE 401 - Instructor-defined Project 1  
**Author:** Matthew Ofina  

## 📌 Project Overview
This repository contains the complete experimental pipeline for training, analyzing, and optimizing YOLOv26 on the VisDrone-DET dataset. The objective is to evaluate real-time object detection capabilities.

## ⚙️ Setup and Reproduction
The following are instructions on how to clone this repository, install dependencies, and download the VisDrone dataset to reproduce these results.
```bash
git clone [https://github.com/yourusername/CMPE401-YOLO-Project.git](https://github.com/yourusername/CMPE401-YOLO-Project.git)
pip install -r requirements.txt
# run code in ipynb file
```

## 📊 Part I: Baseline Model Evaluation

For this project, the baseline object detection model was established using the YOLOv26 nano architecture, fine-tuned on the VisDrone-DET dataset. The model was trained for 50 epochs to establish baseline performance metrics before attempting architectural or hyperparameter improvements.

### 1. Quantitative Results

The table below summarizes the final evaluation metrics of the baseline model on the validation set at the end of the training run:

| Metric | Score |
| :--- | :--- |
| **mAP@50** | 0.23438 |
| **mAP@50-95** | 0.13159 |
| **Precision** | 0.32891 |
| **Recall** | 0.25416 |

### 2. Training and Validation Curves

The following plots illustrate the model's learning behavior over the 10 training epochs, specifically tracking the convergence of the training loss, validation loss, and mean Average Precision (mAP).

![Baseline Loss and mAP Curves](./results/training_validation_metrics.png)
*(Note: Ensure the generated 3-panel plot showing Box Loss, Class Loss, and mAP is saved in your repository's `/results` folder).*

### 3. Class-Specific Performance

To better understand the model's baseline accuracy across the diverse object classes in the VisDrone dataset, the confusion matrix below highlights the True Positive rates and common misclassifications.

![Confusion Matrix](./results/confusion_matrix.png)
