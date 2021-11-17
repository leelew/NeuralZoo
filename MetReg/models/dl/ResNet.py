import tensorflow as tf


def ResNet(self):
    def resid_block(self, num_channels, num_residuals, first_block=False):

        model = tf.keras.models.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                model.add(resid(num_channels, pointwise=True, strides=2))
            else:
                model.add(resid(num_channels, strides=1))
        return model

    """https: // arxiv.org/pdf/1512.03385.pdf
    """
    resnet = tf.keras.models.Sequential()
    resnet.add(
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'))
    resnet.add(tf.keras.layers.BatchNormalization())
    resnet.add(tf.keras.layers.Activation('relu'))
    resnet.add(
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

    resnet.add(self.resid_block(64, 2, first_block=True))
    resnet.add(self.resid_block(128, 2))
    resnet.add(self.resid_block(256, 2))
    resnet.add(self.resid_block(512, 2))

    resnet.add(tf.keras.layers.GlobalAveragePooling2D())
    resnet.add(tf.keras.layers.Dense(10))

    resnet.build([None, 128, 128, 1])

    return resnet
