from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, Layer, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, l2_reg=0.0):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, kernel_regularizer=l2(l2_reg))
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu", kernel_regularizer=l2(l2_reg)), 
             Dense(embed_dim, kernel_regularizer=l2(l2_reg)),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def build_lstm_model(
    vocab_size: int,
    input_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    num_lstm_layers: int = 1,
    dropout_rate: float = 0.3,
    use_bidirectional: bool = False,
    dense_units: int = 0,
    l2_reg: float = 0.0,
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
        
        lstm_layer = LSTM(
            lstm_units, 
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
        )
        
        if use_bidirectional:
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)
            
        if return_sequences:
             model.add(Dropout(dropout_rate))

    # Final Dropout
    model.add(Dropout(dropout_rate))
    
    # Optional Dense Layer
    if dense_units > 0:
        model.add(Dense(
            dense_units, 
            activation='relu',
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
        ))
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

def build_transformer_model(
    vocab_size: int,
    input_length: int,
    embedding_dim: int = 64,
    num_heads: int = 2,
    ff_dim: int = 64,
    num_transformer_blocks: int = 1,
    dropout_rate: float = 0.1,
    dense_units: int = 32,
    l2_reg: float = 0.0,
    learning_rate: float = 0.001,
    optimizer_name: str = "Adam",
) -> Model:
    """
    Builds and compiles a Transformer model.
    """
    inputs = Input(shape=(input_length,))
    embedding_layer = TokenAndPositionEmbedding(input_length, vocab_size, embedding_dim)
    x = embedding_layer(inputs)
    
    for _ in range(num_transformer_blocks):
        transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim, rate=dropout_rate, l2_reg=l2_reg)
        x = transformer_block(x)
        
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    
    if dense_units > 0:
        x = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
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
