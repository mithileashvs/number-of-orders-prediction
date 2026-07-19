# 📦 Number of Orders Prediction

## SPARKIIT Machine Learning Internship Project

### Project Topic
Number of Orders Prediction

### Developed By
**MITHILEASH V S**

**Registered Email:** mithileashvs@gmail.com

---

# Project Description

This project predicts the daily number of customer orders using historical Superstore sales data. The dataset is preprocessed, analyzed, and used to train multiple machine learning regression models. The best-performing model is selected based on evaluation metrics.

---

# Features

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Machine Learning Model Training
- Model Comparison
- Order Prediction
- Interactive Streamlit Dashboard

---

# Machine Learning Models Used

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

# Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Streamli
- seabornt

---

# Project Structure

```
NOP/
│
├── app.py
├── main.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── Sample - Superstore.csv
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── monthly_orders.png
│   └── orders_over_time.png
│
└── assets/
```

---

# Installation

Clone or download the project.

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Run the Machine Learning Pipeline

```bash
python main.py
```

---

# Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

# Dataset

Sample - Superstore Dataset

---

# Results

The project compares multiple regression algorithms and selects the best-performing model based on R² Score.

The trained model predicts the daily number of customer orders from historical Superstore sales data.

---

# Future Improvements

- Hyperparameter Tuning
- Additional Machine Learning Models
- Real-time Prediction
- Interactive Visualizations
- Cloud Deployment

---

# Internship

Developed as part of the **SPARKIIT Machine Learning Internship 2026**.