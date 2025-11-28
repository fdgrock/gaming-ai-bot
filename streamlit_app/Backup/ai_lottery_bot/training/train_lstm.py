# train_lstm.py
def train_lstm(sequence_data, labels, epochs: int = 5):
    """Train an LSTM model when keras/tensorflow is available, otherwise fallback to sklearn.

    sequence_data: np.array shape (n_samples, timesteps, features)
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_data.shape[1], sequence_data.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(sequence_data, labels, epochs=epochs, batch_size=32, verbose=0)
        return model
    except Exception:
        # fallback: flatten sequences and train a simple sklearn model
        try:
            import numpy as _np
            X = sequence_data.reshape((sequence_data.shape[0], -1))
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=50)
            m.fit(X, labels)
            return m
        except Exception:
            return None
