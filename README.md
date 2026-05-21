# 🎙️ Voice-Verse
### AI-Powered Speech Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

Voice-Verse is an AI-powered Speech Emotion Recognition (SER) system that analyzes human speech and predicts emotions using deep learning techniques.

The system captures emotional patterns from voice signals such as:
- Tone
- Pitch
- Frequency
- Speech intensity
- Cadence

It helps in understanding the emotional context behind speech and can be used in:
- Virtual assistants
- Mental health monitoring
- Customer support analytics
- Human-computer interaction
- Communication training systems

## ✨ Features

- 🎤 Real-time voice emotion prediction
- 🧠 Deep learning-based emotion classification
- 📊 Audio feature extraction using MFCC
- 🔍 Detects emotions like:
  - Happy
  - Sad
  - Angry
  - Neutral
  - Fear
  - Surprise
- 📁 Supports audio file input
- ⚡ Fast prediction pipeline
- 📈 Model training and evaluation support


## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core Programming |
| TensorFlow / Keras | Deep Learning |
| Librosa | Audio Processing |
| NumPy | Numerical Computation |
| Pandas | Data Handling |
| Matplotlib | Visualization |
| Scikit-learn | Preprocessing & Metrics |



## 🔄 Workflow

1. Audio Input
2. Noise Reduction & Preprocessing
3. Feature Extraction (MFCC)
4. Deep Learning Model Processing
5. Emotion Classification
6. Prediction Output

## 📂 Dataset

The model is trained using publicly available emotional speech datasets such as:
- RAVDESS
- TESS
- CREMA-D

These datasets contain labeled emotional speech samples for training and evaluation.

## 📊 Model Performance

| Metric | Score |
|--------|------|
| Accuracy | 91% |
| Precision | 89% |
| Recall | 90% |
| F1-Score | 89% |

## 📁 Project Structure

voice-verse/
│── dataset/
│── models/
│── notebooks/
│── src/
│── app.py
│── train.py
│── requirements.txt
│── README.md
