# Stock Price Predictor

A full-stack web application that predicts future stock prices using an LSTM neural network.  
The backend is built with Flask and TensorFlow, while the frontend uses HTML, CSS, and JavaScript.

# Features

- User inputs a stock ticker and number of prediction days  
- Historical stock data is fetched using yfinance  
- LSTM model is trained in real time for predictions  
- Prediction results are plotted and displayed in the frontend  
- Clean frontend with separate HTML, CSS, and JS files

# Technologies

- Python  
- Flask  
- TensorFlow / Keras  
- yfinance  
- HTML, CSS, JavaScript  
- Chart.js

# Project Structure

stock-price-predictor  
- backend  
  - app.py  
  - model_utils.py  
- frontend  
  - index.html  
  - style.css  
  - script.js  
- plots  
  - prediction plots will be saved here  
- requirements.txt  
- README.md

# Getting Started

## 1. Clone the repository

git clone https://github.com/yuvaanshkapila/stock-price-predictor.git  
cd stock-price-predictor

## 2. Create and activate a virtual environment

python -m venv venv

On Windows  
venv\Scripts\activate

On macOS/Linux  
source venv/bin/activate

## 3. Install dependencies

pip install -r requirements.txt

## 4. Run the Flask backend

cd backend  
python app.py

Backend will run at: http://127.0.0.1:5000/

## 5. Open the frontend

Open frontend/index.html in your browser  
(You can use Live Server in VS Code for best results)

# How It Works

- The user enters a stock ticker and number of prediction days  
- The frontend sends a POST request to the Flask backend  
- The backend downloads historical data using yfinance, trains an LSTM model, and predicts future prices  
- The predicted values are returned and displayed in a chart using Chart.js

# Notes

- The model is trained dynamically each time based on user input  
- You can add features like model caching or deployment on the cloud  
- Internet access is required to fetch stock data via yfinance

# License

This project is licensed under the MIT License.
