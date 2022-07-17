"""
This module just creates the model
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(size: int):
    """
    Build the autoencoder model which reduce the dimension one time
    """
    padding = "same"
    activation = "relu"
    half_size = size / 2
    kernel_size = (6, 6)
    pool_size = (2, 2)
    inputs = layers.Input(shape=(size, size, 1))
    # Encoder
    conv1_1 = layers.Conv2D(
        size, kernel_size=kernel_size, activation=activation, padding=padding
    )(inputs)
    pool1 = layers.MaxPooling2D(pool_size=pool_size, padding=padding)(conv1_1)
    conv1_2 = layers.Conv2D(
        half_size, kernel_size=kernel_size, activation=activation, padding=padding
    )(pool1)
    pool2 = layers.MaxPooling2D(pool_size=pool_size, padding=padding)(conv1_2)
    conv1_3 = layers.Conv2D(
        half_size, kernel_size=kernel_size, activation=activation, padding=padding
    )(pool2)
    latent_space_representation = layers.MaxPooling2D(
        pool_size=pool_size, padding=padding
    )(conv1_3)

    # Decoder
    conv2_1 = layers.Conv2D(
        half_size, kernel_size=kernel_size, activation=activation, padding=padding
    )(latent_space_representation)
    up1 = layers.UpSampling2D(pool_size=pool_size)(conv2_1)
    conv2_2 = layers.Conv2D(
        half_size, kernel_size=kernel_size, activation=activation, padding=padding
    )(up1)
    up2 = layers.UpSampling2D(pool_size=pool_size)(conv2_2)
    conv2_3 = layers.Conv2D(
        size, kernel_size=kernel_size, activation=activation, padding=padding
    )(up2)
    up3 = layers.UpSampling2D(pool_size=pool_size)(conv2_3)
    outputs = layers.Conv2D(
        1, kernel_size=kernel_size, activation="linear", padding=padding
    )(up3)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.summary()
    return autoencoder


if __name__ == "__main__":
    model = build_autoencoder(64)
