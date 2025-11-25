from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop

def build_lstm_model(
    vocab_size: int,
    input_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    num_lstm_layers: int = 1,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    optimizer_name: str = "Adam",
) -> Sequential:
    """
    Builds and compiles an LSTM model with specified hyperparameters.
    """
    model = Sequential()
    
    # Embedding Layer
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=input_length,
        )
    )
    
    # LSTM Layers
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1) # True for all but the last layer
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        if return_sequences:
             model.add(Dropout(dropout_rate))

    # Final Dropout
    model.add(Dropout(dropout_rate))
    
    # Dense Output Layer
    model.add(Dense(1, activation="sigmoid"))
    
    # Optimizer
    if optimizer_name.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
