from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import os
import tensorflow as tf

app = Flask(__name__)

SEQ_LENGTH = 60

# Load saved model and scaler once at startup
MODEL_PATH = os.path.join('model', 'lstm_model.h5')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

def create_sequences(data, seq_length):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
    return np.array(X)

@app.route('/')
def serve_frontend():
    return send_file('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    future_days = int(data.get('future_days', 30))

    df = yf.download(ticker, start='2010-01-01')
    if df.empty:
        return jsonify({'error': 'Invalid ticker or no data found'}), 400

    FEATURE = 'Close'
    data_vals = df[[FEATURE]].values.astype('float32')

    # Scale using loaded scaler
    data_scaled = scaler.transform(data_vals)

    if len(data_scaled) < SEQ_LENGTH:
        return jsonify({'error': 'Not enough data to make predictions'}), 400

    # Prepare sequences for test
    X = create_sequences(data_scaled, SEQ_LENGTH)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_test = X[split:]
    y_test = data_scaled[SEQ_LENGTH + split:, 0]

    # Predict on test data
    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    dates_test = df.index[SEQ_LENGTH + split:]

    # Predict future days
    last_sequence = data_scaled[-SEQ_LENGTH:]
    future_scaled = []

    recent_preds = model.predict(X_test[-200:], verbose=0).flatten()
    volatility = np.std(recent_preds) * 0.8

    for _ in range(future_days):
        seq_input = last_sequence.reshape(1, SEQ_LENGTH, 1)
        next_scaled = model.predict(seq_input, verbose=0)[0][0]
        noise = np.random.normal(0, volatility)
        next_scaled_noisy = np.clip(next_scaled + noise, 0, 1)
        future_scaled.append(next_scaled_noisy)
        last_sequence = np.append(last_sequence[1:], [[next_scaled_noisy]], axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d').tolist()

    return jsonify({
        'future_dates': future_dates,
        'future_predictions': future_predictions.tolist(),
        'test_dates': dates_test.strftime('%Y-%m-%d').tolist(),
        'test_actual': actual.flatten().tolist(),
        'test_predicted': predicted.flatten().tolist()
    })

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    app.run()
