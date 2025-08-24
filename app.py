from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import requests

app = Flask(__name__, static_folder='templates')
SEQ_LENGTH = 60

def create_sequences(data, seq_length, future_days):
    X, y = [], []
    for i in range(seq_length, len(data) - future_days):
        X.append(data[i - seq_length:i])
        y.append(data[i:i + future_days, 0])  
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

@app.route("/api/yahoo-quote")
def yahoo_quote():
    symbol = request.args.get("symbol", "AAPL")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.info
        selected_data = {
            "symbol": symbol,
            "longName": data.get("longName"),
            "currentPrice": data.get("currentPrice"),
            "marketCap": data.get("marketCap"),
            "previousClose": data.get("previousClose"),
            "open": data.get("open"),
            "dayHigh": data.get("dayHigh"),
            "dayLow": data.get("dayLow"),
            "fiftyTwoWeekHigh": data.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": data.get("fiftyTwoWeekLow"),
            "volume": data.get("volume"),
            "averageVolume": data.get("averageVolume"),
            "currency": data.get("currency"),
        }

        return jsonify(selected_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

    df = yf.download(ticker, start='2010-01-01', auto_adjust=False)
    if df.empty:
        return jsonify({'error': f'Invalid ticker or no data found for {ticker}'}), 400

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map('_'.join)

    close_col = f'Close_{ticker}'
    if close_col not in df.columns:
        return jsonify({'error': f"Error: 'Close' price data not available for ticker {ticker}"}), 400

    df = df.dropna(subset=[close_col])
    if len(df) < SEQ_LENGTH + future_days + 10:
        return jsonify({'error': 'Not enough historical data to train and test'}), 400

    series_close = df[close_col]
    latest_close = round(float(series_close.iloc[-1]), 2)
    last_30 = series_close.tail(30)
    support_level = round(float(last_30.min()), 2)
    resistance_level = round(float(last_30.max()), 2)

    macd_line, signal_line, histogram = calculate_macd(series_close)
    rsi = calculate_rsi(series_close)

    features = pd.DataFrame({
        'log_close': np.log(series_close),
        'macd': macd_line,
        'rsi': rsi
    }).dropna()

    scaler_log_close = MinMaxScaler()
    scaler_macd = MinMaxScaler()
    scaler_rsi = MinMaxScaler()

    scaled_log_close = scaler_log_close.fit_transform(features[['log_close']])
    scaled_macd = scaler_macd.fit_transform(features[['macd']])
    scaled_rsi = scaler_rsi.fit_transform(features[['rsi']])

    scaled = np.hstack([scaled_log_close, scaled_macd, scaled_rsi])

    X, y = create_sequences(scaled, SEQ_LENGTH, future_days)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential()
    model.add(tf.keras.Input(shape=(SEQ_LENGTH, X.shape[2])))
    model.add(LSTM(50, return_sequences=True, activation='relu'))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(future_days))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    predicted_scaled = model.predict(X_test, verbose=0)
    predicted_log = scaler_log_close.inverse_transform(predicted_scaled)
    actual_log = scaler_log_close.inverse_transform(y_test)
    predicted = np.exp(predicted_log[:, 0])
    actual = np.exp(actual_log[:, 0])
    dates_test = features.index[SEQ_LENGTH + split:SEQ_LENGTH + split + len(actual)]

    last_sequence = scaled[-SEQ_LENGTH:]
    future_scaled = model.predict(last_sequence.reshape(1, SEQ_LENGTH, X.shape[2]), verbose=0)[0]
    future_log = scaler_log_close.inverse_transform(future_scaled.reshape(-1, 1)).flatten()
    future_predictions = np.exp(future_log)

    min_price = latest_close * 0.5
    max_price = latest_close * 1.5
    future_predictions = np.clip(future_predictions, min_price, max_price)

    future_dates = pd.date_range(start=features.index[-1] + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d').tolist()

    return jsonify({
        'symbol': ticker,
        'latest_close': latest_close,
        'support_30d': support_level,
        'resistance_30d': resistance_level,
        'future_dates': future_dates,
        'future_predictions': [round(float(p), 2) for p in future_predictions],
        'test_dates': dates_test.strftime('%Y-%m-%d').tolist(),
        'test_actual': [round(float(a), 2) for a in actual],
        'test_predicted': [round(float(p), 2) for p in predicted],
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
