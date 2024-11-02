# Importing necessary modules
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from data_preprocessing import X_test, y_test
from train import load_trained_model
import joblib


scaler = joblib.load('scaler.pkl')

model_choice = input("Do you want to test on RNN , GRU , Single Layer LSTM or Multi Layer LSTM? (Type 'RNN','GRU' ,'S_LSTM', 'M_LSTM' ): ").strip().upper()

if model_choice == 'RNN':
    model_path = 'models/rnn_model.keras'
elif model_choice == 'GRU':
    model_path = "models/gru_model.keras"
elif model_choice == 'S_LSTM':
    model_path = 'models/lstm_model_single_layer_withoutValidation.keras'
elif model_choice == 'M_LSTM':
    model_path = "models/lstm_model_multi_layer.keras"
else:
    raise ValueError("Invalid input! Please type 'RNN','GRU','S_LSTM' or 'M_LSTM'.")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist. Please train the model first.")

model = load_trained_model(model_path)
print(f"Testing on model: {model_choice}")

predicted_stock_price = model.predict(X_test)


predicted_stock_price_original = scaler.inverse_transform(predicted_stock_price)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the errors
mse = mean_squared_error(y_test_actual, predicted_stock_price_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, predicted_stock_price_original)


print(f"Results for {model_choice} Model:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")