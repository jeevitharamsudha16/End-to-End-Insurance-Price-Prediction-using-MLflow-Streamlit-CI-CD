#  Insurance Price Prediction using Streamlit & CI/CD

This project is an end-to-end Machine Learning solution designed to **predict medical insurance charges** based on user input. It combines robust model training pipelines, automated evaluation, and modern frontend deployment using **Streamlit**. Additionally, it integrates **CI/CD pipelines via GitHub Actions** to ensure smooth deployment and updates.

## 🚀 Live Demo

🌐 [View App on Streamlit Cloud](https://insurance-price-prediction-using-app-and-ci-cd-dmdu9cb7zr5jug5.streamlit.app/)


## 🎯 Problem Statement

Medical insurance costs vary significantly depending on personal factors like age, BMI, region, and smoking habits. Insurance companies want a model that can **accurately estimate charges** for new customers using historical data.


## 📌 Objectives

- Build a predictive model for insurance charges using regression techniques.
- Train and evaluate multiple models and select the best one automatically.
- Develop a user-friendly web interface with **Streamlit**.
- Integrate CI/CD using **GitHub Actions** for automatic deployment on every push.


## 🧪 Dataset Details

- **Size**: 1,300+ rows, 7 columns
- **Features**:
  - `age`: Age of the individual
  - `sex`: Gender (`male`, `female`)
  - `bmi`: Body Mass Index
  - `children`: Number of dependent children
  - `smoker`: Smoking status
  - `region`: Residential region
  - `charges`: Target variable (insurance cost)


## 🔧 Tools & Technologies Used

| Category         | Tools / Libraries                                     |
|------------------|--------------------------------------------------------|
| Programming      | Python                                                 |
| Libraries        | Pandas, NumPy, Scikit-learn, XGBoost, Joblib           |
| Visualization    | Matplotlib, Seaborn                                    |
| UI Framework     | Streamlit                                              |
| Model Tracking   | MLflow (removed in final deployment version)           |
| Automation       | GitHub Actions                                         |



## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**
   - Handle categorical variables (label & one-hot encoding)
   - Normalize features like `age`, `bmi` using MinMaxScaler

2. **Model Training**
   - Models trained:
     - Linear Regression
     - Ridge Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - Support Vector Machine (SVR)
     - K-Nearest Neighbors
     - XGBoost
   - Hyperparameter tuning via `GridSearchCV`
   - Saved each model in `models/` folder

3. **Model Evaluation**
   - Used `R² score` for comparison
   - Visualized predictions vs actuals
   - Selected best-performing model and saved it as `best_model.pkl`


## 📊 Model Performance Summary

| Model               | R² Score (CV) |
|---------------------|---------------|
| Linear Regression   | 0.74          |
| Ridge Regression    | 0.76          |
| Decision Tree       | 0.71          |
| Random Forest       | 0.80          |
| Gradient Boosting ✅| **0.84**      |
| XGBoost             | 0.82          |


> ✅ Best model selected: **Gradient Boosting Regressor**


## 🖥️ Streamlit Web App Features

- User input sliders for all features
- Dynamic prediction based on `best_model.pkl`
- Minimal and responsive UI
- Real-time charge estimation upon clicking **Predict**

## 🔁 CI/CD with GitHub Actions
✅ train_pipeline.yml
Triggers on every push to main
Trains multiple models
Saves the best model to models/best_model.pkl

🔁 streamlit_health.yml
Verifies Streamlit app loads successfully
Prevents broken UI from being deployed





