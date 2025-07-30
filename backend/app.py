from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import os
import socket
from waitress import serve

app = Flask(__name__, static_folder='../frontend')

SEQ_LENGTH = 60

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    print("Step 1: /predict endpoint hit")

    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    future_days = int(data.get('future_days', 30))

    print(f"Step 2: Received ticker={ticker}, future_days={future_days}")

    df = yf.download(ticker, start='2010-01-01')
    if df.empty:
        print("Error: No data found for ticker")
        return jsonify({'error': 'Invalid ticker or no data found'}), 400

    FEATURE = 'Close'
    data_vals = df[[FEATURE]].values.astype('float32')

    print("Step 3: Raw data downloaded and processed")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_vals)

    if len(data_scaled) < SEQ_LENGTH:
        print("Error: Not enough data to train")
        return jsonify({'error': 'Not enough data to train the model'}), 400

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("Step 4: Data prepared, building model...")

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, 1), activation='relu'))
    model.add(LSTM(units=50, return_sequences=False, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Step 5: Model compiled, starting training...")
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1, verbose=0)
    print("Step 6: Training completed")

    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    dates_test = df.index[SEQ_LENGTH + split:]

    print("Step 7: Model tested on historical data")

    last_sequence = data_scaled[-SEQ_LENGTH:]
    future_scaled = []

    recent_preds = model.predict(X_test[-200:], verbose=0).flatten()
    volatility = np.std(recent_preds) * 0.8

    print("Step 8: Starting future prediction generation...")
    for _ in range(future_days):
        seq_input = last_sequence.reshape(1, SEQ_LENGTH, 1)
        next_scaled = model.predict(seq_input, verbose=0)[0][0]
        noise = np.random.normal(0, volatility)
        next_scaled_noisy = np.clip(next_scaled + noise, 0, 1)
        future_scaled.append(next_scaled_noisy)
        last_sequence = np.append(last_sequence[1:], [[next_scaled_noisy]], axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d').tolist()

    print("Step 9: Future prediction complete, sending response")

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

    use_socket = os.environ.get("USE_SOCKET", "false").lower() == "true"
    if use_socket:
        sock_path = '/home/yuvaansh/yuvaansh.sock'
        if os.path.exists(sock_path):
            os.remove(sock_path)
        print(f"Serving Flask via Unix socket at {sock_path}")
        serve(app, unix_socket=sock_path)
    else:
        print("Serving Flask on 0.0.0.0:41251")
        app.run(host='0.0.0.0', port=41251, debug=True)
