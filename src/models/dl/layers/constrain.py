import tensorflow as tf



class Constrain(tf.keras.layers.Layer):
    """Constrain layer Type 1

        combine center cropping & space to channel & average pooling
        to constrain the info from space into channel.

        input:
            - 5D: (batch_size, timesteps, height, width, channels)

        output:
            - 5D: (batch_size, timesteps, height/2, width/2, 2*channels)
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        # average pooling
        self.avg_pool_1 = tf.keras.layers.AveragePooling2D()
        # center cropping
        self.crop_1 = tf.keras.layers.Cropping2D(
            cropping=((int(input_shape[-3]) // 4, int(input_shape[-3]) // 4),
                      (int(input_shape[-2]) // 4, int(input_shape[-2]) // 4)))
        # space to channel (4 parts)
        self.space2channel_1 = tf.keras.layers.Cropping2D(
            cropping=((0, int(input_shape[-3]) // 2),
                      (0, int(input_shape[-2]) // 2)))
        self.space2channel_2 = tf.keras.layers.Cropping2D(
            cropping=((0, int(input_shape[-3]) // 2),
                      (int(input_shape[-2]) // 2, 0)))
        self.space2channel_3 = tf.keras.layers.Cropping2D(
            cropping=((int(input_shape[-3]) // 2, 0),
                      (0, int(input_shape[-2]) // 2)))
        self.space2channel_4 = tf.keras.layers.Cropping2D(
            cropping=((int(input_shape[-3]) // 2, 0),
                      (int(input_shape[-2]) // 2, 0)))

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. merge dimension sample and timestep
                   1. global average pool for spatial
                   2. cropping for spatial
                   3. space to channel
                   4. concat
                   5. spread dimensions
        """
        # get shape 5D
        _, T, H, W, C = inputs.get_shape().as_list()
        # 5D to 4D
        x = tf.reshape(inputs, [-1, H, W, C])
        # avg
        x_avg = self.avg_pool_1(x)
        # crop
        x_crop = self.crop_1(x)
        # space to channel
        x_space2channel = tf.concat((self.space2channel_1(x),
                                     self.space2channel_2(x),
                                     self.space2channel_3(x),
                                     self.space2channel_4(x)), axis=-1)
        # concat
        x = tf.concat([x_avg, x_crop], -1)
        # reshape
        x = tf.reshape(x, [-1, T, H // 2, W // 2, 2*C])  # 6*C
        return x


class Downsampling(tf.keras.layers.Layer):
    """Constrain layer Type 2.

        construct by convolutional & pooling layers to exact info
        between each channels.

    Input shape:
        - 5D: (batch_size, timesteps, height, width, channels)

     Output shape:
        - 5D: (batch_size, timesteps, height/2, width/2, channels)
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=2*int(input_shape[-1]),
            kernel_size=(3, 3),
            padding='same')
        self.pool_1 = tf.keras.layers.MaxPooling2D()

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=2*int(input_shape[-1]),
            kernel_size=(3, 3),
            padding='same')
        self.pool_2 = tf.keras.layers.MaxPooling2D(strides=(2, 2))

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. merge dimension sample and timestep
                   1. convolutional layer
                   2. pooling
                   5. spread dimensions
        """
        # get shape 5D
        _, T, H, W, C = inputs.get_shape().as_list()
        # 5D to 4D
        x = tf.reshape(inputs, [-1, H, W, C])
        #
        x = self.conv_1(x)
        x = self.pool_1(x)
        #
        x = tf.reshape(x, [-1, T, H // 2, W // 2, 2 * C])
        return x
