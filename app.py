from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

app = Flask(__name__, static_folder='templates')

SEQ_LENGTH = 60

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
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
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        future_days = int(data.get('future_days', 30))

        df = yf.download(ticker, start='2010-01-01')
        if df.empty:
            return jsonify({'error': 'Invalid ticker or no data found'}), 400

        FEATURE = 'Close'
        data_vals = df[[FEATURE]].values.astype('float32')

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data_vals)

        if len(data_scaled) < SEQ_LENGTH:
            return jsonify({'error': 'Not enough data to train the model'}), 400

        X, y = create_sequences(data_scaled, SEQ_LENGTH)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, 1), activation='relu'))
        model.add(LSTM(units=50, return_sequences=False, activation='relu'))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        dates_test = df.index[SEQ_LENGTH + split:]

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

    except Exception as e:
        return jsonify({'error': str(e)}), 500


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


