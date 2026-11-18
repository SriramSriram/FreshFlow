# FreshFlow – Predictive Supply Chain Analytics for Perishable Products

## Project Overview

FreshFlow is a data analytics and machine learning project designed to support inventory planning for perishable products. The project analyzes historical demand, product characteristics, inventory-related information and expiry data to identify patterns and provide demand predictions and inventory planning insights.

The project combines Exploratory Data Analysis (EDA), Principal Component Analysis (PCA), Machine Learning and an interactive Streamlit dashboard.

## Objectives

- Analyze historical demand and inventory-related data.
- Identify demand patterns across products and categories.
- Explore dimensionality reduction using PCA.
- Predict demand using a Random Forest Regression model.
- Provide inventory planning insights based on predicted demand, available stock and expiry risk.
- Visualize results using an interactive Streamlit dashboard and Power BI.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- PCA
- Streamlit
- Plotly
- Power BI
- Jupyter Notebook

## Project Structure

```text
FreshFlow/
│
├── app.py
├── .gitignore
├── README.md
│
├── data/
│   └── dashboard.pbix
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 01_eda_Analysis.ipynb
│   ├── 02_PCA_Analysis.ipynb
│   ├── 03_Random_Forest.ipynb
│   ├── 04_Inventory_Recommendations.ipynb
│   ├── models/
│   │   └── random_forest_baseline.pkl
│   └── chat.py
│
└── src/
    └── recommendations.py
```

## Project Workflow

1. Data exploration and understanding
2. Data cleaning and validation
3. Exploratory Data Analysis
4. PCA-based dimensionality exploration
5. Feature preparation
6. Random Forest model training
7. Model evaluation
8. Inventory recommendation logic
9. Interactive dashboard development

## Machine Learning

A Random Forest Regressor is used to predict daily demand.

### Model Evaluation

| Metric | Value |
|---|---:|
| MAE | 19.70 |
| RMSE | 34.51 |
| R² Score | 0.8474 |

The model was evaluated using a chronological train-test split to preserve the temporal order of the records.

## Dashboard

The Streamlit dashboard includes:

- Executive Overview
- Historical Analytics
- Demand Prediction
- Inventory Recommendations
- Expiry and Waste Analysis

The dashboard provides interactive visualizations and inventory planning scenarios.

## Dataset

The project uses a perishable goods management dataset containing 100,000 records and 42 columns.

The dataset is not included in this repository. Users must obtain the dataset separately and place the required CSV files in the appropriate project directories before running the application.

## Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Disclaimer

The inventory recommendations are analytical decision-support suggestions and are not automated purchase orders. Available stock is entered as a scenario input where required, and expiry risk classifications are based on the project's analytical rules.

## Author

M.Tech – Big Data Analytics Project

FreshFlow – Predictive Supply Chain Analytics for Perishable Products
