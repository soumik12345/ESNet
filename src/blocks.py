import tensorflow as tf


def DownsamplingBlock(input_tensor, input_channels, output_channels):
    '''Downsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        input_channels  -> Number of channels in the input tensor
        output_channels -> Number of output channels
    '''
    x1 = tf.keras.layers.Conv2D(
        output_channels - input_channels, (3, 3),
        strides=(2, 2), use_bias=True, padding='same'
    )(input_tensor)
    x2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(input_tensor)
    x = tf.keras.layers.Concatenate(axis = 3)([x1, x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def FCU(input_tensor, output_channels, K=3, dropout_prob=0.03):
    '''Factorized Convolutional Unit
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor -> Input Tensor
        K -> Size of Kernel
    '''
    x = tf.keras.layers.Conv2D(
        output_channels, (K, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (1, K),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (K, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (1, K),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([input_tensor, x])
    x = tf.keras.layers.Dropout(dropout_prob)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def PFCU(input_tensor, output_channels):
    '''Parallel Factorized Convolutional Unit
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor -> Input Tensor
        output_channels -> Number of output channels
    '''
    x = tf.keras.layers.Conv2D(
        output_channels, (3, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (1, 3),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Branch 1
    branch_1 = tf.keras.layers.Conv2D(
        output_channels, (3, 1), dilation_rate = (2, 2),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_1 = tf.keras.layers.ReLU()(branch_1)
    branch_1 = tf.keras.layers.Conv2D(
        output_channels, (1, 3), dilation_rate = (2, 2),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_1)
    branch_1 = tf.keras.layers.BatchNormalization()(branch_1)
    # Branch 2
    branch_2 = tf.keras.layers.Conv2D(
        output_channels, (3, 1), dilation_rate = (5, 5),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_2 = tf.keras.layers.ReLU()(branch_2)
    branch_2 = tf.keras.layers.Conv2D(
        output_channels, (1, 3), dilation_rate = (5, 5),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_2)
    branch_2 = tf.keras.layers.BatchNormalization()(branch_2)
    # Branch 3
    branch_3 = tf.keras.layers.Conv2D(
        output_channels, (3, 1), dilation_rate = (9, 9),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_3 = tf.keras.layers.ReLU()(branch_3)
    branch_3 = tf.keras.layers.Conv2D(
        output_channels, (1, 3), dilation_rate = (9, 9),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_3)
    branch_3 = tf.keras.layers.BatchNormalization()(branch_3)
    x = tf.keras.layers.Add()([input_tensor, branch_1, branch_2, branch_3])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def UpsamplingBlock(input_tensor, output_channels):
    '''Upsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        output_channels -> Number of output channels
    '''
    x = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, padding='same',
        strides=(2, 2), use_bias=True
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x