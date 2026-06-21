# Clinical-LLM-CDSS

## Overview

Clinical-LLM-CDSS is a multimodal AI-powered Clinical Decision Support System designed to analyze laboratory reports in PDF and image formats. The system automatically extracts clinical information, routes cases to disease-specific prediction models, generates explainable predictions, and provides an interactive chatbot powered by Ollama LLM.

## Key Features

- PDF and image-based laboratory report analysis
- OCR-powered clinical parameter extraction
- Feature normalization and preprocessing
- Intelligent feature-driven routing engine
- Disease-specific prediction models
- Hybrid XGBoost + MLP architecture
- Explainable AI predictions
- Ollama-powered healthcare chatbot
- Multiclass and multilabel disease prediction

## System Architecture

```text
Laboratory Report (PDF/Image)
          ↓
OCR & Data Extraction
          ↓
Feature Mapping & Normalization
          ↓
Router Engine
          ↓
Disease-Specific Model Selection
          ↓
Hybrid XGBoost + MLP Models
          ↓
Prediction Results
          ↓
Explainability Module
          ↓
Ollama Clinical Chatbot
```

The system employs dedicated prediction pipelines for individual diseases. Based on the extracted laboratory features, the router automatically selects the appropriate disease-specific model.

Current implementations include:
- Anemia Prediction
- Liver Disease Prediction

Future extensions:
- Diabetes Prediction
- Cardiac Disease Prediction
- Kidney Disease Prediction

## Technology Stack

- Python
- XGBoost
- TensorFlow / Keras
- MLP Neural Networks
- OCR
- NLP
- Ollama
- Scikit-learn
- Pandas
- Streamlit

## Repository Structure

```text
modules/
models/
artifacts/
app.py
main.py
train.py
