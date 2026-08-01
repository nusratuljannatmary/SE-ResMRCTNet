"""
Model architecture: CNN + Squeeze-and-Excitation (SE) blocks + Transformer encoder.

This is the hybrid network used for EEG stress classification in both the
STEW and EEGMAT experiments. Keeping it in one file means train.py,
evaluate.py and the EEGMAT pipeline all import the *same* architecture,
so results stay reproducible.
"""

from tensorflow.keras.layers import (
    Input, Dense, concatenate, Conv1D, GlobalAveragePooling1D,
    Dropout, Reshape, Multiply, Add,
)
from tensorflow.keras.models import Model
from keras_nlp.layers import TransformerEncoder


def se_block(input_tensor, reduction_ratio=4):
    """Squeeze-and-Excitation block: re-weights channels by importance."""
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Reshape((1, channels))(se)
    se = Dense(channels // reduction_ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])


def combined_cnn_se_block(input_tensor):
    """Two parallel Conv1D branches (different kernel sizes) + SE, concatenated."""
    conv1 = Conv1D(32, kernel_size=16, strides=2, activation='relu', padding='same')(input_tensor)
    conv2 = Conv1D(32, kernel_size=32, strides=2, activation='relu', padding='same')(input_tensor)
    se1 = se_block(conv1)
    se2 = se_block(conv2)
    return concatenate([se1, se2], axis=2)


def build_transformer_model(seq_len, input_dim, intermediate_dim=32, num_heads=5,
                            num_layers=1, dropout_rate=0.1, num_classes=2):
    """
    Full model:
      input -> CNN+SE -> Transformer encoder(s) (residual) ->
      global pooling -> residual MLP head -> softmax.

    Args:
        seq_len:   number of time samples per epoch
        input_dim: number of EEG channels
    """
    inputs = Input(shape=(seq_len, input_dim))
    x = combined_cnn_se_block(inputs)

    for _ in range(num_layers):
        transformer_out = TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
        )(x)
        x = Add()([x, transformer_out])

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    shortcut = x
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(shortcut.shape[-1], activation='linear')(x)
    x = Add()([shortcut, x])
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
