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
OCR & LLM Extraction
          ↓
Feature Mapping & Normalization (Preprocessing)
          ↓
Router Engine (Disease-Specific Model Selection)
          ↓
Hybrid Disease Specific Models (XGBoost + MLP)
          ↓
Prediction Results
          ↓
Severity Module
          ↓
LLM Explainability Module
          ↓
Clinical Decisions
```

The system employs dedicated prediction pipelines for individual diseases. Based on the extracted laboratory features, the router automatically selects the appropriate disease-specific model.

Current implementations include:
- Anemia Prediction
- Liver Disease Prediction

Future extensions:
- Multiple disease Prediction
- Learning Based Router Implementation
- Server Level Implementation

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
app.py
main.py
