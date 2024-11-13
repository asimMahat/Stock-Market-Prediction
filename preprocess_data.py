import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

file_path = "dataset/ibm.csv"
data = pd.read_csv(file_path)


def preprocess_data(data):
    data_filtered = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    data_filtered["Date"] = pd.to_datetime(data_filtered["Date"])
    data_filtered = data_filtered.sort_values(by="Date")
    data_cleaned = data_filtered.dropna()
    data_close = data_cleaned["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_close)
    joblib.dump(scaler, "scaler.pkl")
    return data_scaled, scaler


def create_sequences(data, time_step):
    X = np.array([data[i : i + time_step, 0] for i in range(len(data) - time_step - 1)])
    y = np.array([data[i + time_step, 0] for i in range(len(data) - time_step - 1)])
    return X, y


# unpacking values
data_scaled, scaler = preprocess_data(data)

print(data_scaled)

time_step = 60
X, y = create_sequences(data_scaled, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


print(X_train.shape)
print(y_train.shape)
