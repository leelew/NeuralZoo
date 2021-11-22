# ==============================================================================
# Keras channel domain attention layers
#
# (1) SE: Squeeze-and-Excitation block, that adaptively recalibrates
#   channel-wise feature responses by explicitly modelling interdependencies
#   between channels. Original code is available at
#   https://github.com/hujie-frank/SENet.
# (2) SK:
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================

import tensorflow as tf
from tensorflow import multiply
from tensorflow.keras import layers
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Reshape)


class SE(layers.Layer):
    """Squeeze-and-Excitation block.

    Args:
      units (int): the dimensionality of the output space (i.e., the number 
        of output units in fc layer). Equal to the number of inputs channels.s
      reduction_ratio (int, optional): A hyperparameter which allows us to 
        vary the capacity and computational cost of the SE blocks in the 
        network. Hu et al., (2019) says setting r = 16 achieves a good 
        balance between accuracy and complexity. 
      squeeze_operator (str, optional): Squeeze operator, which could be
        average pooling and max pooling. Hu et al. 2019 points that the 
        performance of SE blocks is fairly robust to the choice of specific 
        aggregation operator. Defaults to 'avg'.
      excitation_operator (str, optional): Excitation operator, which could 
        be 'ReLU', 'tanh' and 'sigmoid'. Defaults to 'sigmoid'. The study 
        points that exchanging the sigmoid for tanh slightly worsens 
        performance, while using ReLU is dramatically worse and drop below 
        that baseline. 
      trainable (bool, optional): Defaults to True.
      name (str, optional): Defaults to 'SE-block'.
      dtype ([type], optional): Defaults to None.
      dynamic (bool, optional): Defaults to False.
    Call arguments:
      inputs: A 4D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    Input shape:
      - If data_format='channels_first'
        4D tensor with shape: `(samples, time, channels, rows)` 
      - If data_format='channels_last'
        4D tensor with shape: `(samples, time, rows, channels)`
    Output shape:
      - 4D tensor with shape: `(samples, time, units, rows)` 
    
    Raises:
      ValueError: in case of invalid constructor arguments.
    
    References:
      - [Hu et al., 2019](https://arxiv.org/abs/1709.01507)
    """
    def __init__(self,
                 units: int,
                 reduction_ratio: int = 16,
                 squeeze_operator: str = 'avg',
                 excitation_operator: str = 'sigmoid',
                 data_format=None,
                 trainable: bool = True,
                 name: str = 'SE-block',
                 dtype=None,
                 dynamic: bool = False,
                 **kwargs):
        if squeeze_operator not in ['avg', 'max']:
            raise ValueError("Squeeze operator must be global average \
                            pooling or global max pooling, use 'avg' \
                            and 'max' instead.")
        if excitation_operator not in ['sigmoid', 'relu', 'tanh']:
            raise ValueError("Excitation operator must be sigmoid, relu and \
                            tanh. Hu et al. 2019 recommend sigmoid for ResNet \
                            according to several experiements.")
        self.units = units
        self.squeeze_operator = squeeze_operator
        self.excitation_operator = squeeze_operator
        self.reduction_ratio = reduction_ratio
        self.data_format = data_format

        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first': channel_axis = 1
        else: channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs (last axis) \
                            should be defined. Found None. Full input shape \
                            received: input_shape={input_shape}')

        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(4)

        # squeeze block
        if self.squeeze_operator == 'avg':
            self.squeeze = GlobalAveragePooling2D()
        if self.squeeze_operator == 'max': self.squeeze = GlobalMaxPooling2D()
        # excitation
        self.dense1 = Dense(self.units // self.reduction_ratio,
                            activation='relu')
        self.dense2 = Dense(self.units, activation=self.excitation_operator)

        self.built = True

    def call(self, inputs, **kwargs):
        # squeeze
        squeeze = self.squeeze(inputs)
        # excitation
        excitation = self.dense1(squeeze)
        excitation = self.dense2(excitation)
        excitation = Reshape((1, 1, self.units))(excitation)

        return multiply([inputs, excitation])

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'reduction_ratio': self.reduction_rato,
            'squeeze_operator': self.squeeze_operator,
            'excitation_operator': self.excitation_operator,
            'data_format': self.data_format
        })
        return config


class se_5d(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = Dense(input_shape[-1] // 8)
        #activation='relu',
        #kernel_initializer='he_normal',
        #use_bias=True,
        #bias_initializer='zeros')
        self.dense2 = Dense(input_shape[-1])
        #activation='relu',
        #kernel_initializer='he_normal',
        #use_bias=True,
        #bias_initializer='zeros')

        self.gap = GlobalAveragePooling2D()  #(keepdims=True)

    def call(self, inputs, input_shape):
        t, lat, lon, c = input_shape
        print((t, lat, lon, c))
        a = Reshape((lat, lon, c * t))(inputs)
        print(a.shape)
        se = self.gap(a)  # tf version > 2.6.0
        se = Reshape((1, 1, c * t))(se)
        print(se.shape)
        se = self.dense1(se)
        print(se.shape)
        se = self.dense2(se)
        print(se.shape)
        a = multiply([a, se])
        print(a.shape)
        inputs = Reshape((t, lat, lon, c))(a)

        return inputs


class SK():
    pass
