# Vendor Invoice Analytics & Prediction System

A comprehensive machine learning system for analyzing vendor invoices, predicting freight costs, and flagging suspicious invoices for review.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Data Schema](#data-schema)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Making Predictions](#making-predictions)
- [Models](#models)
- [Contributing](#contributing)

## Overview

This project provides two main ML-powered capabilities for vendor invoice management:

1. **Freight Cost Prediction**: Predicts freight costs based on invoice dollar amounts
2. **Invoice Flagging**: Identifies potentially suspicious or problematic invoices based on multiple features

The system uses SQLite for data storage and scikit-learn for machine learning models.

## Project Structure

```
new-project/
├── data/
│   └── inventory.db              # SQLite database with vendor and purchase data
├── freight_cost_prediction/
│   ├── data_preprocessing.py     # Data loading and feature preparation
│   ├── model_evaluation.py       # Model training and evaluation functions
│   ├── train.py                  # Training pipeline for freight cost models
│   └── models/
│       └── predict_freight_model.pkl
├── invoice_flagging/
│   ├── data_preprocessing.py     # Data loading with complex SQL joins
│   ├── modeling_evaluation.py    # Random Forest classifier with GridSearch
│   ├── train.py                  # Training pipeline for invoice flagging
│   └── models/
│       ├── predict_flag_invoice.pkl
│       └── scaler.pkl            # StandardScaler for feature normalization
├── inference/
│   ├── predict_freight.py        # Freight cost prediction script
│   └── predict_invoice_flag.py   # Invoice flagging prediction script
└── notebooks/
    ├── predicting_frieght_cost.ipynb
    └── flagging.ipynb
```

## Features

### 1. Freight Cost Prediction
- Predicts freight costs based on invoice dollar amounts
- Compares Linear Regression, Decision Tree, and Random Forest models
- Automatically selects the best model based on Mean Absolute Error (MAE)
- Simple one-feature prediction for ease of use

### 2. Invoice Flagging
- Identifies suspicious invoices using multiple features
- Flags invoices with:
  - Invoice total mismatch with item-level totals (>$5 difference)
  - Abnormally high receiving delays (>10 days)
- Uses Random Forest Classifier with comprehensive GridSearchCV tuning
- Feature scaling with StandardScaler for improved performance

## Data Schema

The system uses an SQLite database (`inventory.db`) with the following key tables:

### `vendor_invoice`
- `VendorNumber`: Vendor identifier
- `VendorName`: Vendor name
- `InvoiceDate`: Date of invoice
- `PONumber`: Purchase order number
- `PODate`: Purchase order date
- `PayDate`: Payment date
- `Quantity`: Invoice quantity
- `Dollars`: Invoice dollar amount
- `Freight`: Freight cost
- `Approval`: Approval status

### `purchases`
- Purchase order line items
- Links to `vendor_invoice` via `PONumber`
- Contains individual item quantities, brands, and dollar amounts

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd new-project
```

2. Install required packages:
```bash
pip install pandas scikit-learn joblib sqlite3
```

3. Ensure the database file exists at:
```
/Volumes/Sachin/ML-Projects/new-project/data/inventory.db
```

## Usage

### Training Models

#### Train Freight Cost Prediction Model
```bash
cd freight_cost_prediction
python train.py
```

This will:
- Load vendor invoice data from the database
- Train Linear Regression, Decision Tree, and Random Forest models
- Evaluate each model on test data
- Save the best performing model to `models/predict_freight_model.pkl`

#### Train Invoice Flagging Model
```bash
cd invoice_flagging
python train.py
```

This will:
- Load and join vendor invoice and purchase data
- Apply labeling rules to identify suspicious invoices
- Perform GridSearchCV to find optimal Random Forest parameters
- Save the best model to `models/predict_flag_invoice.pkl`
- Save the fitted scaler to `models/scaler.pkl`

### Making Predictions

#### Predict Freight Costs
```bash
cd inference
python predict_freight.py
```

Example usage in code:
```python
from predict_freight import predict_freight

sample_data = {
    "Dollars": [214.26, 1500.50, 15527.25, 137483.78, 5000.00]
}

predictions = predict_freight(sample_data)
print(predictions[['Dollars', 'Predicted_Freight']])
```

Output:
```
     Dollars  Predicted_Freight
0     214.26                6.0
1    1500.50               13.0
2   15527.25               83.0
3  137483.78              695.0
4    5000.00               30.0
```

#### Predict Invoice Flags
```bash
cd inference
python predict_invoice_flag.py
```

Example usage in code:
```python
from predict_invoice_flag import predict_invoice_flag

sample_data = {
    "invoice_quantity": [6, 15, 5, 10100, 1935],
    "invoice_dollars": [214.26, 140.55, 106.60, 137483.78, 15527.25],
    "Freight": [3.47, 8.57, 4.61, 2935.20, 429.20],
    "total_item_quantity": [6, 15, 5, 10100, 1935],
    "total_item_dollars": [214.26, 140.55, 106.60, 137483.78, 15527.25]
}

predictions = predict_invoice_flag(sample_data)
print(predictions)
```

Output:
```
   invoice_quantity  invoice_dollars  Freight  total_item_quantity  total_item_dollars  Predicted_Flag
0                 6           214.26     3.47                    6              214.26               1
1                15           140.55     8.57                   15              140.55               1
2                 5           106.60     4.61                    5              106.60               0
3             10100        137483.78  2935.20                10100           137483.78               1
4              1935         15527.25   429.20                 1935            15527.25               0

Flag 0 = Normal Invoice, Flag 1 = Suspicious Invoice
```

## Models

### Freight Cost Prediction Model

**Features:**
- `Dollars`: Invoice dollar amount

**Target:**
- `Freight`: Freight cost

**Algorithms Evaluated:**
- Linear Regression
- Decision Tree Regressor (max_depth=4)
- Random Forest Regressor (max_depth=5)

**Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

**Best Model Selection:**
The model with the lowest MAE is automatically selected and saved.

### Invoice Flagging Model

**Features:**
1. `invoice_quantity`: Quantity on invoice
2. `invoice_dollars`: Dollar amount on invoice
3. `Freight`: Freight cost
4. `total_item_quantity`: Total quantity from purchase order items
5. `total_item_dollars`: Total dollars from purchase order items

**Target:**
- `flag_invoice`: Binary label (0 = Normal, 1 = Suspicious)

**Labeling Rules:**
- Flag = 1 if `|invoice_dollars - total_item_dollars| > 5`
- Flag = 1 if `avg_receiving_delay > 10 days`
- Flag = 0 otherwise

**Algorithm:**
Random Forest Classifier with GridSearchCV

**Hyperparameter Search Space:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [None, 4, 5, 6]
- `min_samples_split`: [2, 3, 5]
- `min_samples_leaf`: [1, 2, 5]
- `criterion`: ['gini', 'entropy']

**Optimization:**
- Scoring: F1 Score
- Cross-validation: 5-fold CV
- Parallel processing enabled (`n_jobs=-1`)

**Preprocessing:**
- StandardScaler applied to all features for normalization

**Metrics:**
- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request



## Made By Sachin


---

**Note:** This system is designed for vendor invoice analysis and fraud detection. Always validate predictions with domain expertise before making business decisions.
