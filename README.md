# 📊 Churn Prediction with ANN + Hyperparameter Tuning
## Overview
This project uses an Artificial Neural Network (ANN) to predict customer churn based on demographic and account features. It includes a robust hyperparameter tuning pipeline using GridSearchCV and SciKeras, allowing dynamic optimization of model architecture (neurons, layers, epochs).
## 🧠 Problem Statement
Customer churn is a critical metric in banking and telecom industries. Predicting which customers are likely to leave enables proactive retention strategies. This model classifies customers as likely to churn (Exited = 1) or stay (Exited = 0) based on features such as:
Credit Score
Age
Tenure
Account Balance
Geography
Product Usage
Activity Status
## ⚙️ Technologies Used
Python 3.10
TensorFlow / Keras
SciKeras (KerasClassifier wrapper)
Scikit-learn
Pandas / NumPy
GridSearchCV for hyperparameter tuning
## 🧪 Model Architecture
Input layer: Based on feature count
Hidden layers: Tuned between 1–2 layers
Neurons per layer: Tuned between 16–128
Activation: ReLU
Output layer: Sigmoid (binary classification)
Loss: Binary Crossentropy
Optimizer: Adam
## 🔧 Hyperparameter Tuning

param_grid = {
    'model__neurons': [16, 32, 64, 128],
    'model__layers': [1, 2],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}
Cross-validation: 3-fold
Best model selected based on validation accuracy
Results printed with best score and parameters
## 📈 Results
Best Accuracy: e.g., 86.4%
Best Parameters: {'neurons': 64, 'layers': 2, 'epochs': 100}
Model generalizes well on unseen test data
