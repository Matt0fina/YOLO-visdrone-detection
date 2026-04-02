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

Using the trained best.pt model, we can see an improvement in the scores:

| Metric | Score |
| :--- | :--- |
| **mAP@50** | 0.88776 |
| **Precision** | 0.90634 |
| **Recall** | 0.64603 |

### 2. Training and Validation Curves

The following plots illustrate the model's learning behavior over the 10 training epochs, specifically tracking the convergence of the training loss, validation loss, and mean Average Precision (mAP).

<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/adc7729d-8d94-43fb-9d45-0c1be50e09d6" />

*(Note: Ensure the generated 3-panel plot showing Box Loss, Class Loss, and mAP is saved in your repository's `/results` folder).*

### 3. Class-Specific Performance

To better understand the model's baseline accuracy across the diverse object classes in the VisDrone dataset, the confusion matrix below highlights the True Positive rates and common misclassifications.

The normalized training set confusion matrix can be seen as follows:
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/8605bef6-75bb-430e-8480-50209303d965" />

The normalized validation set confusion matrix can be seen as follows:
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/61d43e78-8a07-4a31-b392-134f6417f9f1" />

## 📈 Part II: Loss Curve and Fitting Analysis

Based on the baseline training and validation curves generated over 10 epochs, we can evaluate the model's learning dynamics, specifically regarding convergence and fitting behavior.

### 1. Convergence Behavior
The model was able to demonstrate convergence in its ability to classify objects. As seen in the **Class Loss** graph, both training and validation losses decreased had a sharp decrease and stabilized relatively early (around Epoch 3). This parallel descent indicates that the model was successfully learning the foundational features required to differentiate between the VisDrone classes without simply memorizing the training set. This is further supported by the sharp initial climb and subsequent high plateau in the **mAP** curve. Increasing the Epochs could cause better overall results

### 2. Diagnosis: Mild Overfitting on Box Regression
While the classification performance generalized well, the **Box Loss** graph shows signs of mild overfitting. 
* **Observation:** The training box loss consistently decreases throughout the entire 10 epochs. However, the validation box loss initially drops but begins to flatten out and the training box loss also seems to flatten out at a higher loss rate, indicating no convergence.
* **Underfitting:** There are no severe signs of underfitting, as the training losses successfully minimized and the final mAP achieved a high score of 0.887. The model clearly possessed enough initial learning capability to grasp the task.

### 3. Root Causes: Dataset Size and Model Capacity
The observed overfitting on bounding box coordinates is likely due to the dataset characteristics and the chosen model architecture interacting with each other:
**Dataset Size and Complexity:** The VisDrone dataset is highly complex, including dense scenes with different objects with very different sizes (e.g., tiny pedestrians vs. large vehicles). While the model easily learned *what* the objects were, drawing mathematically precise bounding boxes (which heavily impacts Box Loss) across such a massive, varied dataset is exceedingly difficult.
* **Model Capacity:** The baseline model used was the **nano** version (YOLOv26n). While highly efficient, it has a relatively low parameter count (model capacity). Because its capacity is limited, it lacks the deep representational power needed to perfectly generalize the spatial coordinates of highly dense, small objects in unseen validation images, leading it to over-optimize (overfit) on the spatial coordinates of the training set instead.

## 🧪 Part III: Structured Experimental Design

A controlled experiment was conducted, changing one variable, to evaluate the baseline model's performance for object detections

### 1. Experimental Settings:
* **Experimental Variable:** 640x640
* **Control Group:** YOLOv26n trained for 10 epochs, 8 batch size, and **640x640** resolution.
* **Experimental Group:** YOLOv26n trained for 10 epochs, 8 batch size, and **1024x1024** resolution.
* **Constants:** All other paramters (batch size, learning rate, augmentation, etc.) remained the same to clearly see the affect of the variable.

### 2. Quantitve Results
The following table outlines the perfermance difference between the baseline.

| Metric | Baseline (640px) | Experiment (1024px) | Delta |
| :--- | :--- | :--- | :--- |
| **mAP@50** | 0.23438 | 0.3313 | +0.09692 |
| **mAP@50-95** | 0.13159 | 0.2007 | +0.06911 |
| **Validation Box Loss** | 2.17183 | 1.78897 | -0.38286 |
| **Precision** | 0.32891 | 0.4108 | +0.08189 |
| **Recall** | 0.25416 | 0.34304 | +0.08888 |

<img width="1388" height="491" alt="image" src="https://github.com/user-attachments/assets/62b948fb-058c-4509-adaa-2e56db2e512a" />

### 3. Analysis
Increasing the input image resolution from 640 to 1024 shows an improvement to classification accuracy and precision. Because the VisDrone dataset has images with many small objects cluttered together, where the baseline 640px downscaling likely compressed critical object features into unrecognizable pixel clutters.

With the higher 1024px resolution, the model had much more spatial data to work with. The higher resolution resulted in an increase mAP@50 and a noticeable reduction in Validation Box loss. The precision and recall from the results can also be seen to increase. Overall, providing the model with better quality inputs allowed it to accurately bound small objects without increasing the training epochs.
