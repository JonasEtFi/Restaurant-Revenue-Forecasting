# Restaurant-Revenue-Forecasting

## Project Overview
This project focuses on forecasting restaurant revenue using time series analysis and machine learning techniques. The goal is to provide better employee planning and resource management for a restaurant in Heidelberg by predicting future revenues based on historical data, weather conditions, and other relevant factors.

## Project Steps

### 1. Data Collection & Feature Selection
- The project began with gathering and selecting relevant features for the analysis. The dataset includes various factors such as weather data (temperature, precipitation, etc.), historical revenue data, and holidays.
- Feature selection was performed to identify the most impactful features contributing to accurate revenue predictions.

### 2. Time Series Analysis with Lagged Features
- Lagged features were introduced to capture the temporal dependencies in the data. For instance, revenue from previous days and rolling averages (3-day and 7-day) were calculated to provide additional context for the model.
  
### 3. Machine Learning Model
- An **XGBoost** model was used to predict future revenue based on the prepared dataset.
- The XGBoost model is a powerful gradient boosting algorithm that works well for tabular data and is particularly effective for handling both linear and non-linear relationships in time series data.

### 4. Model Evaluation
- The model's performance was evaluated using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
  
### 5. Use Case: Employee Planning
- The forecasted revenue is intended to assist in better employee planning for the restaurant in Heidelberg. By accurately predicting revenue, the restaurant can optimize staffing levels and improve operational efficiency.


## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/restaurant-revenue-forecasting.git
   cd restaurant-revenue-forecasting

2. Install requirements
    ```bash
    pip install -r requirements.txt

## Author
Jonas Fischer - jonas.fisch1809@web.de
