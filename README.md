# Insurance-charges-prediction-project

# Overview

This project provides a comprehensive demonstration of an end-to-end machine learning pipeline designed for predicting insurance charges. The process includes various stages such as data preparation, model building, evaluation, and deployment. FastAPI was utilized for the deployment of the final model, while MLflow served as a tool for tracking and comparing model performance metrics throughout the development.

# Key Steps

The project begins with data preparation, where the dataset is cleaned and relevant columns are encoded to make it suitable for model training. Five different regression models were built: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Support Vector Regressor (SVR), and XGBoost Regressor. These models were evaluated based on metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). After thorough comparison, the XGBoost model was selected due to its superior performance. This model was then deployed using FastAPI to provide easy API-based interactions.

# Installation

To get started, clone the repository and navigate to the project directory:
git clone https://github.com/your-username/insurance-charges-prediction.git
cd insurance-charges-prediction

# Running the project

Deploy the model using FastAPI:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
The API can be accessed at http://127.0.0.1:8000/.

# License

This project is distributed under the MIT License.

# Contributions

Feedback and contributions to enhance the project are encouraged and appreciated.
