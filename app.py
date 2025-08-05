from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import tensorflow as tf

app = Flask(__name__, static_folder='templates')
SEQ_LENGTH = 60
MODEL = None
SCALER = None
DATES_TEST = None
ACTUAL = None
PREDICTED = None

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_model(ticker):
    global MODEL, SCALER, DATES_TEST, ACTUAL, PREDICTED

    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start='2010-01-01', auto_adjust=False)
    if df.empty:
        raise ValueError("Failed to download stock data")

    FEATURE = 'Close'
    data_vals = df[[FEATURE]].values.astype('float32')

    SCALER = MinMaxScaler(feature_range=(0, 1))
    data_scaled = SCALER.fit_transform(data_vals)

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"Training shape: X={X_train.shape}, y={y_train.shape}")

    MODEL = Sequential([
        Input(shape=(SEQ_LENGTH, 1)),
        LSTM(units=50, return_sequences=True, activation='relu'),
        LSTM(units=50, activation='relu'),
        Dense(units=1)
    ])
    MODEL.compile(optimizer='adam', loss='mean_squared_error')
    MODEL.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    predicted_scaled = MODEL.predict(X_test)
    PREDICTED = SCALER.inverse_transform(predicted_scaled)
    ACTUAL = SCALER.inverse_transform(y_test.reshape(-1, 1))
    DATES_TEST = df.index[SEQ_LENGTH + split:]

    print(f"Model trained and ready. Test size: {len(DATES_TEST)}")

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if MODEL is None or SCALER is None or ACTUAL is None or PREDICTED is None:
            return jsonify({'error': 'Model not trained yet'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON payload'}), 400

        future_days = int(data.get('future_days', 30))

        if len(ACTUAL) < SEQ_LENGTH:
            return jsonify({'error': 'Not enough historical data for prediction'}), 400

        last_sequence = SCALER.transform(ACTUAL)[-SEQ_LENGTH:]
        future_scaled = []

        recent_preds = SCALER.transform(PREDICTED[-200:]).flatten()
        volatility = np.std(recent_preds) * 0.8

        for _ in range(future_days):
            seq_input = last_sequence.reshape(1, SEQ_LENGTH, 1)
            next_scaled = MODEL.predict(seq_input, verbose=0)[0][0]
            noise = np.random.normal(0, volatility)
            next_scaled_noisy = np.clip(next_scaled + noise, 0, 1)
            future_scaled.append(next_scaled_noisy)
            last_sequence = np.append(last_sequence[1:], [[next_scaled_noisy]], axis=0)

        future_predictions = SCALER.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=DATES_TEST[-1] + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d').tolist()

        return jsonify({
            'future_dates': future_dates,
            'future_predictions': future_predictions.tolist(),
            'test_dates': DATES_TEST.strftime('%Y-%m-%d').tolist(),
            'test_actual': ACTUAL.flatten().tolist(),
            'test_predicted': PREDICTED.flatten().tolist()
        })

    except Exception as e:
        print("Error during /predict:", e)
        return jsonify({'error': str(e)}), 500

# Train model on startup with default ticker
try:
    train_model("AAPL")
except Exception as e:
    print("Startup training failed:", e)

# Optional: prevent GPU crash warnings
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass

if __name__ == '__main__':
    app.run(debug=True)
