# 🔧 PCB Defect Detection

End-to-end Computer Vision system for detecting
manufacturing defects on Printed Circuit Boards (PCBs).

## 🎯 Project Goal

The goal of this project is to automatically detect
PCB manufacturing defects using an object detection model.

The system identifies:

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper

## 🧠 Model

YOLO object detection is used for PCB defect detection.

Input:

PCB image

Output:

- Defect class
- Bounding box
- Confidence score

## 🔬 Pipeline

Kaggle Dataset
→ Data Exploration
→ Data Preparation
→ YOLO Format
→ Data Augmentation
→ Model Training
→ Evaluation
→ Error Analysis
→ Inference
→ Gradio Application
→ Hugging Face Spaces

## 📊 Evaluation Metrics

The model is evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50:95
- Confusion Matrix

## 🌐 Deployment

The trained model is deployed using:

- Hugging Face Spaces
- Gradio

Users can upload a PCB image and receive
an annotated image with detected defects.

## 📁 Project Structure

```text
pcb-defect-detection/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app/
├── outputs/
├── dataset.yaml
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE