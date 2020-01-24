import tensorflow as tf
from .blocks import *


def ESNet(input_shape=(1024, 512, 3), output_channels=19):
    inp = tf.keras.layers.Input(input_shape)
    ##### Encoder #####
    # Block 1
    x = DownsamplingBlock(inp, 3, 16)
    x = FCU(x, 16, K=3)
    x = FCU(x, 16, K=3)
    x = FCU(x, 16, K=3)
    # Block 2
    x = DownsamplingBlock(x, 16, 64)
    x = FCU(x, 64, K=5)
    x = FCU(x, 64, K=5)
    # Block 3
    x = DownsamplingBlock(x, 64, 128)
    x = PFCU(x, 128)
    x = PFCU(x, 128)
    x = PFCU(x, 128)
    ##### Decoder #####
    # Block 4
    x = UpsamplingBlock(x, 64)
    x = FCU(x, 64, K=5, dropout_prob=0.0)
    x = FCU(x, 64, K=5, dropout_prob=0.0)
    # Block 5
    x = UpsamplingBlock(x, 16)
    x = FCU(x, 16, K=3, dropout_prob=0.0)
    x = FCU(x, 16, K=3, dropout_prob=0.0)
    output = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, padding='same',
        strides=(2, 2), use_bias=True
    )(x)
    return tf.keras.models.Model(inp, output)