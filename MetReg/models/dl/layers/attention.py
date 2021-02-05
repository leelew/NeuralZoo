import tensorflow as tf



class FeatureAttention(tf.keras.layers.Layer):
    """Channel Attention layers.

    Parameters
    ----------
    activation: str, optional (default 'sigmoid')
    dense_ratio: float, optional (default 1)

    input:
        could be applied for both input shape
        - 4D: (batch_size, height, weight, channels) &
        - 5D: (batch_size, timesteps, height, weight, channels)

    output:
        - channel attentioned inputs: the same dims with inputs
        - channel attention weight:
            - 4D: (batch_size, 1, 1, channels) &
            - 5D: (batch_size, timesteps, 1, 1, channels)

    References:
        - Contains the implementation of Squeeze-and-Excitation(SE) block.
        As described in https://arxiv.org/abs/1709.01507.
    """

    def __init__(self,
                 dense_ratio=1,
                 activation='sigmoid'):
        super().__init__()

        self.dense_ratio = dense_ratio
        self.activation = activation

    def build(self, input_shape):

        self.dense_avg_pool_1 = tf.keras.layers.Dense(
            units=int(input_shape[-1]) //
            self.dense_ratio,
            activation='relu',
            name='MLP_AVG_POOL_1')
        self.dense_avg_pool_2 = tf.keras.layers.Dense(
            units=int(input_shape[-1]),
            name='MLP_AVG_POOL_2')
        self.activation = tf.keras.layers.Activation(
            activation='sigmoid',
            name='CHANNEL_ATTENTION_SIGMOID')

        super().build(input_shape)

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. global average pooling & max pooling
                   1. two same dense layers for both pooling
                   2. concat
        """
        # avg pooling
        avg_pool = tf.compat.v1.reduce_mean(
            inputs, axis=[-3, -2], keep_dims=True)
        x_avg_pool = self.dense_avg_pool_1(avg_pool)
        x_avg_pool = self.dense_avg_pool_2(x_avg_pool)
        # max pooling
        max_pool = tf.compat.v1.reduce_max(
            inputs, axis=[-3, -2], keep_dims=True)
        x_max_pool = self.dense_avg_pool_1(max_pool)
        x_max_pool = self.dense_avg_pool_2(x_max_pool)
        # concat
        scale = self.activation(x_avg_pool + x_max_pool)
        return inputs*scale


class SpatialAttention(tf.keras.layers.Layer):
    """Spatial Attention layers.

    Parameters
    ----------
    kernel_size:

    input:
        - 4D: (batch_size, height, weight, channels) &
        - 5D: (batch_size, timesteps, height, weight, channels)

    output shape:
        - channel attentioned inputs: the same dims with inputs
        - channel attention weight:
            - 4D: (batch_size, 1, 1, channels) &
            - 5D: (batch_size, timesteps, 1, 1, channels)

    References:
        - Contains the implementation of Convolutional Block Attention Module
        (CBAM) block. As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        self.kernel_size = kernel_size

    def build(self, input_shape):

        self.dims = len(input_shape)
        self.Conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=[1, 1],
            padding='same',
            name='CONV')
        self.activation = tf.keras.layers.Activation(
            activation='sigmoid',
            name='SPATIAL_ATTENTION_SIGMOID')

        super().build(input_shape)

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. global average pooling & max pooling for channel
                   1. concat
                   2. convolutional and activate
                   3. spread dimensions
        """
        # dense channel dims
        avg_pool = tf.compat.v1.reduce_mean(
            inputs, axis=[-1], keep_dims=True)
        max_pool = tf.compat.v1.reduce_max(
            inputs, axis=[-1], keep_dims=True)
        # concat
        concat = tf.concat([avg_pool, max_pool], -1)

        # reshape
        if self.dims == 5:
            _, T, H, W, C = concat.get_shape().as_list()
            concat = tf.reshape(concat, [-1, H, W, C])
        else:
            _, H, W, C = concat.get_shape().as_list()

        # contract info
        x = self.Conv(concat)
        # activate
        scale = self.activation(x)

        # reshape
        if self.dims == 5:
            scale = tf.reshape(scale, [-1, T, H, W, 1])
        else:
            scale = tf.reshape(scale, [-1, H, W, 1])

        return inputs*scale


class SelfAttention(tf.keras.layers.Layer):
    """Self Attention layers.

    Parameters
    ----------
    kernel_size:

    input shape:
        could be applied for both input shape
        - 4D: (batch_size, height, weight, channels) &

    output shape:
        - channel attentioned inputs: the same dims with inputs
        - channel attention weight:
            - 4D: (batch_size, 1, 1, channels) &
            - 5D: (batch_size, timesteps, 1, 1, channels)

    References:
        - Attention is all you need.
    """

    def __init__(self, dropout_rate=0.1):
        super().__init__()

        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        self.dims = len(input_shape)

        self.H, self.W, self.C = \
            int(input_shape[-3]), \
            int(input_shape[-2]), \
            int(input_shape[-1])

        if self.dims == 5:
            self.T = int(input_shape[-4])

        self.q = tf.keras.layers.Conv2D(
            filters=int(input_shape[-1]), kernel_size=1)
        self.k = tf.keras.layers.Conv2D(
            filters=int(input_shape[-1]), kernel_size=1)
        self.v = tf.keras.layers.Conv2D(
            filters=int(input_shape[-1]), kernel_size=1)

        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.bn = tf.keras.layers.BatchNormalization()

        super().build(input_shape)

    def call(self, inputs):
        """Exec"""
        if self.dims == 5:
            inputs = tf.reshape(inputs, [-1, self.H, self.W, self.C])

        # get q,k,v
        Q = self.q(inputs)
        K = self.k(inputs)
        V = self.v(inputs)
        # get shape
        _, H, W, C = Q.get_shape().as_list()
        # reshape 4D to 3D
        Q_ = tf.reshape(Q, [-1, H*W, C])
        K_ = tf.reshape(K, [-1, H*W, C])
        V_ = tf.reshape(V, [-1, H*W, C])
        # multiply
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # scale
        outputs = outputs / (C ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)
        # Dropouts
        outputs = self.dropout_1(outputs)
        # weighted sum
        outputs = tf.matmul(outputs, V_)
        # reshape 3D to 4D
        outputs = tf.reshape(outputs, [-1, H, W, C])
        # residual connection
        outputs += inputs
        # normalization
        outputs = self.bn(outputs)

        if self.dims == 5:
            outputs = tf.reshape(outputs, [-1, self.T, self.H, self.W, self.C])

        return outputs
