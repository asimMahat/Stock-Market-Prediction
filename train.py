
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from data_preprocessing import X_train, y_train,time_step
# from tensorflow.keras.models import save_model

# Define the training function
def train_lstm_model(X_train, y_train, time_step, batch_size=32, epochs=30, save_model_path='models/lstm_model.keras'):
    # LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2)) 
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    checkpoint = ModelCheckpoint(save_model_path, monitor='loss', save_best_only=True, mode='min', verbose=1)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])

    # Save the final model after training
    model.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    return model

# Function to load the saved model
def load_trained_model(model_path='models/lstm_model.keras'):
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"The model file {model_path} does not exist.")


train_lstm_model(X_train,y_train,time_step)


