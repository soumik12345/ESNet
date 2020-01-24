import tensorflow as tf


def DownsamplingBlock(input_tensor, output_channels):
    '''Downsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        output_channels -> Number of output channels
    '''
    x1 = tf.keras.layers.Conv2D(
        output_channels - 3, (3, 3),
        strides=(2, 2), use_bias=True, padding='same'
    )(input_tensor)
    x2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(input_tensor)
    x = tf.keras.layers.Concatenate(axis = 3)([x1, x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x