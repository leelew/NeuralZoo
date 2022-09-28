# ==============================================================================
# Keras hybrid domain attention layers
#
# (1) CBAM: Convolutional Block Attention Module, that sequentially infers
#   attention maps along two separate dimensions, channel and spatial, then
#   the attention maps are multiplied to the input feature map for adaptive
#   feature refinement. CBAM learns what and where to emphasize or suppress
#   and refines intermediate feature effectively.
# (2) DANet:
# (3) axial attention: used for adaptive extract features, which is proposed by
#   Li et al., 2021.
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Conv2D, Dense, multiply


class CBAM(layers.Layer):
    """Convolutional block attention module.

    Notes:
      Compared to SE, CBAM simply changes the channel-wise attention by
      adding max-pool feature rather than only use average-pool features. 
      Woo et al. 2018 argue that max-pool feature could encode the degree
      of the most salient part which can compensate the average-pooled 
      features simultaneously. A shared network was used to merge both
      max and average-pool features.

      Woo et al. 2018 compared two spatial attention: pooling and 1x1 
      convolution, and pointed that explicitly modeled pooling lead to finer 
      attention inference compared to learnable weighted channel pooling.

    Args:
      units (int): The dimensionality of output space. Equal to input channels.
      reduction_ratio (int, optional): The same setting as 16 with SE block. 
      kernel_size (int, optional): The kernel size of convolutional operation 
        in spatial aggregation. Woo et al. 2018 find that adopting a larger 
        kernel size generates better accuracy, which implies a larger 
        receptive field. Thus setting defaults to 7.
      data_format ([type], optional): [description]. Defaults to None.
      trainable (bool, optional): Defaults to True.
      name (str, optional): Defaults to 'CBAM'.
      dtype ([type], optional): Defaults to None.
      dynamic (bool, optional): Defaults to False.

    Call arguments:
      inputs: A 4D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.

    Input shape:
      - 4D: (batch_size, height, weight, channels) &

    output shape:
      - channel attentioned inputs: the same dims with inputs
    
    References:
      - [Woo et al., 2018](https://arxiv.org/abs/1807.06521)
    """
    def __init__(self,
                 units: int,
                 reduction_ratio: int = 16,
                 kernel_size: int = 7,
                 data_format=None,
                 trainable: bool = True,
                 name: str = 'CBAM',
                 dtype=None,
                 dynamic: bool = False,
                 **kwargs):
        self.units = units
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.data_format = data_format

        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        self.channel = input_shape[channel_axis]

        # channel attention
        self.dense1 = Dense(self.channel // self.reduction_ratio,
                            activation='relu')
        self.dense2 = Dense(self.channel, activation='relu')
        self.act1 = Activation('sigmoid')

        # spatial attention
        self.conv = Conv2D(filters=1,
                           kernel_size=self.kernel_size,
                           strides=[1, 1],
                           activation='sigmoid',
                           padding='same')

        self.built = True

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. global average pooling & max pooling for spatial
                   1. share dense layer for both pooling
                   2. concat
                   3. spread dimensions

                   0. global average pooling & max pooling for channel
                   1. concat
                   2. convolutional and activate
                   3. spread dimensions
        """
        # channel
        avg_pool = tf.math.reduce_mean(inputs, axis=[-2, -3], keep_dims=True)
        max_pool = tf.math.reduce_max(inputs, axis=[-2, -3], keep_dims=True)
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)
        scale = self.act1(avg_pool + max_pool)
        inputs = multiply([inputs, scale])

        # spatial
        avg_pool = tf.math.reduce_mean(inputs, axis=[-1], keep_dims=True)
        max_pool = tf.math.reduce_max(inputs, axis=[-1], keep_dims=True)
        concat = tf.concat([avg_pool, max_pool], -1)
        scale = self.conv(concat)
        outputs = multiply([inputs, scale])

        return outputs


class DANet(layers.Layer):
    pass


class AxialAttention(layers.Layer):
    pass
