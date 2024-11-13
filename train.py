import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from preprocess_data import X_train, y_train, time_step


def train_lstm_single_layer(
    X_train,
    y_train,
    time_step,
    batch_size=32,
    epochs=30,
    save_model_path="models/lstm_model_single_layer_withoutValidation.keras",
):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    checkpoint = ModelCheckpoint(
        save_model_path, monitor="loss", save_best_only=True, mode="min", verbose=1
    )
    model.fit(
        X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint]
    )

    # Save the final model after training
    model.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    return model


def train_lstm_multi_layer(
    X_train,
    y_train,
    time_step,
    batch_size=32,
    epochs=30,
    save_model_path="models/lstm_model_multi_layer.keras",
):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    checkpoint = ModelCheckpoint(
        save_model_path, monitor="loss", save_best_only=True, mode="min", verbose=1
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
    )

    model.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    return model


def train_rnn_model(
    X_train,
    y_train,
    time_step,
    batch_size=32,
    epochs=30,
    save_model_path="models/rnn_model.keras",
):
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    checkpoint = ModelCheckpoint(
        save_model_path, monitor="loss", save_best_only=True, mode="min", verbose=1
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
    )

    model.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    return model


def train_gru_model(
    X_train,
    y_train,
    time_step,
    batch_size=32,
    epochs=30,
    save_model_path="models/gru_model.keras",
):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    checkpoint = ModelCheckpoint(
        save_model_path, monitor="loss", save_best_only=True, mode="min", verbose=1
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
    )

    model.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    return model


def load_trained_model(model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file {model_path} does not exist.")


if __name__ == "__main__":

    model_choice = input("'RNN','GRU','S_LSTM' or 'M_LSTM': ").strip().upper()

    if model_choice == "RNN":
        train_rnn_model(X_train, y_train, time_step)
    elif model_choice == "S_LSTM":
        train_lstm_single_layer(X_train, y_train, time_step)
    elif model_choice == "M_LSTM":
        train_lstm_multi_layer(X_train, y_train, time_step)
    elif model_choice == "GRU":
        train_gru_model(X_train, y_train, time_step)
    else:
        print("Provide valid input")
