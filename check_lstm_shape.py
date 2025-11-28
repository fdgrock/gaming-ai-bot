import tensorflow as tf
from pathlib import Path

# Find the latest LSTM model
lstm_dir = Path("models/lstm")
lstm_files = sorted(lstm_dir.glob("lstm_lotto_max_*.keras"))

if lstm_files:
    model_path = lstm_files[-1]
    print(f"Loading LSTM model: {model_path.name}")
    
    model = tf.keras.models.load_model(str(model_path))
    print(f"Input shape: {model.input_shape}")
    print(f"First layer: {model.layers[0]}")
    
    # Check if model has Flatten layer and what comes before it
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} - {type(layer).__name__} - input_shape={layer.input_shape}, output_shape={layer.output_shape}")
else:
    print("No LSTM model found")
