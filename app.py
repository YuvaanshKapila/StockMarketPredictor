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
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def calculate_macd(series, short_span=12, long_span=26, signal_span=9):
    exp1 = series.ewm(span=short_span, adjust=False).mean()
    exp2 = series.ewm(span=long_span, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

@app.route('/')
def serve_frontend():
    return send_from_directory('templates', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = (data.get('ticker') or '').upper()
    future_days = int(data.get('future_days', 30))

    if not ticker:
        return jsonify({'error': "Ticker symbol is required"}), 400

    # Download historical data (keep auto_adjust=False to preserve 'Close')
    df = yf.download(ticker, start='2010-01-01', auto_adjust=False)

    if df.empty:
        return jsonify({'error': f'Invalid ticker or no data found for {ticker}'}), 400

    # Flatten MultiIndex columns like ('Close','AAPL') -> 'Close_AAPL'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map('_'.join)

    close_col = f'Close_{ticker}'
    if close_col not in df.columns:
        return jsonify({'error': f"Error: 'Close' price data not available for ticker {ticker}"}), 400

    # Drop NaNs and ensure we have enough
    df = df.dropna(subset=[close_col])
    if len(df) < SEQ_LENGTH + 10:
        return jsonify({'error': 'Not enough historical data to train and test'}), 400

    # Compute latest close and support/resistance (30-day)
    series_close = df[close_col]
    latest_close = round(float(series_close.iloc[-1]), 2)
    last_30 = series_close.tail(30)
    support_level = round(float(last_30.min()), 2)
    resistance_level = round(float(last_30.max()), 2)

    # Prepare data for LSTM
    data_vals = series_close.values.astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_vals)

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build LSTM model with Input layer style to avoid warning
    model = Sequential()
    model.add(tf.keras.Input(shape=(SEQ_LENGTH, 1)))
    model.add(LSTM(50, return_sequences=True, activation='relu'))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    # Backtest prediction
    predicted_scaled = model.predict(X_test, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    dates_test = df.index[SEQ_LENGTH + split:]

    # Future forecasting with volatility noise
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

    # Indicators
    macd_line, signal_line, histogram = calculate_macd(series_close)
    rsi = calculate_rsi(series_close)

    return jsonify({
        'symbol': ticker,
        'latest_close': latest_close,
        'support_30d': support_level,
        'resistance_30d': resistance_level,
        'future_dates': future_dates,
        'future_predictions': future_predictions.tolist(),
        'test_dates': dates_test.strftime('%Y-%m-%d').tolist(),
        'test_actual': actual.flatten().tolist(),
        'test_predicted': predicted.flatten().tolist(),
        'macd': {
            'macd': macd_line.fillna(0).tolist(),
            'signal': signal_line.fillna(0).tolist(),
            'histogram': histogram.fillna(0).tolist()
        },
        'rsi': rsi.tolist()
    })


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    app.run(debug=True)
