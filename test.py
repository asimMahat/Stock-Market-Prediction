
# getting modules from sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from data_preprocessing import X_test, y_test
from train import load_trained_model

import joblib
import os

scaler = joblib.load('scaler.pkl')
model = load_trained_model('models/lstm_model.keras')
print(model)
predicted_stock_price = model.predict(X_test)

predicted_stock_price_original = scaler.inverse_transform(predicted_stock_price)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# Mean Squared Error (MSE)
mse = mean_squared_error(y_test_actual, predicted_stock_price_original)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_actual, predicted_stock_price_original)

# Results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")