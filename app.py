import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit  # Usamos TimeSeriesSplit para validación temporal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import tensorflow as tf
from datetime import datetime, timedelta

from flask import Flask, jsonify
app = Flask(__name__)

# Establecer semillas aleatorias para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)


# Función para obtener datos de Binance sin caché
def get_binance_data(api_key, symbol="BTCUSDT", interval="1h", limit=2000):
    endpoint = f"https://api.binance.com/api/v3/klines"
    url = f"{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"
    
    headers = {
        'X-MBX-APIKEY': api_key,
        'Cache-Control': 'no-cache'  # Agrega esta línea para desactivar la caché
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

# Función para crear conjuntos de datos
def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size])
    return np.array(X), np.array(Y)


# Clave API de Binance (reemplazar por tu clave real)
api_key = ""

# Obtener datos de Binance
df = get_binance_data(api_key)

# Convertir el precio de cierre a valores flotantes y escalarlo
price = df['close'].astype(float).values
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(price.reshape(-1, 1))

# Tamaño de la ventana de datos
window_size = 500

# Crear conjuntos de datos
X, Y = create_dataset(price_scaled, window_size)

# Usar TimeSeriesSplit para validación temporal
tscv = TimeSeriesSplit(n_splits=5)

# Función para predecir el precio de cierre de la próxima hora
def predict_next_hour_price(model, recent_data, scaler):
    if len(recent_data) != window_size:
        raise ValueError(f"Expected input of length {window_size}. Got {len(recent_data)}.")
    recent_data_normalized = scaler.transform(np.array(recent_data).reshape(-1, 1))
    input_data = np.reshape(recent_data_normalized, (1, window_size, 1))
    predicted_normalized = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_normalized)
    return predicted_price[0][0]


@app.route("/")
def home():
    return "API acknowledgement!"

@app.route("/train")
def train(): 
    # Crear un modelo de red neuronal recurrente (GRU)
    model = Sequential([
        GRU(50, input_shape=(window_size, 1), return_sequences=True),
        GRU(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrenamiento utilizando validación temporal
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        history = model.fit(X_train, Y_train, epochs=42, validation_data=(X_test, Y_test), batch_size=64, verbose=0)  # Oculta los detalles del entrenamiento
    
    # Evaluar el modelo en el conjunto de prueba
    loss = model.evaluate(X_test, Y_test)
    print(f"Loss on test set: {loss}")
    
    # Obtener el precio de cierre de la última hora completa
    last_complete_hour_prices = price[-window_size-1:-1]  # Tomamos hasta el penúltimo dato
    last_complete_hour_timestamps = df['timestamp'][-window_size-1:-1].values
    
    # Convertir el último timestamp a tipo datetime
    last_timestamp = pd.to_datetime(last_complete_hour_timestamps[-1])
    
    # Predecir el precio para la próxima hora
    predicted_price_next_hour = predict_next_hour_price(model, last_complete_hour_prices, scaler)
    print(f"Timestamp de la última hora completa (GMT-5): {last_timestamp - timedelta(hours=5)}")
    print(f"Precio de cierre de la última hora completa: {last_complete_hour_prices[-1]}")
    print(f"Predicción del precio para la próxima hora: {predicted_price_next_hour:.2f}")
    first = f"{last_timestamp - timedelta(hours=5)}"
    second = f"{last_complete_hour_prices[-1]}"
    third = f"{predicted_price_next_hour:.2f}"
    response = {"Timestamp de la última hora completa (GMT-5)": first,
                "Precio de cierre de la última hora completa": second,
                "Predicción del precio para la próxima hora": third         
                                                                 }
    
    # Guardar el modelo en formato HDF5
    #model.save("my_model.h5")
    return jsonify(response)

if __name__ == '__main__':
    app.run()
