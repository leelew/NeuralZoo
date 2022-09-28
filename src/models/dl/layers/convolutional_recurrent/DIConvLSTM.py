import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.python.keras.layers.convolutional_recurrent import \
    ConvLSTM2DCell


class DIConvLSTM(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        super().__init__()

    def build(self, input_shape):
        self.convlstm = []
        self.t = input_shape[1]

        for i in range(self.t):
            self.convlstm.append(ConvLSTM2DCell(filters=self.filters, 
                                                kernel_size=self.kernel_size, 
                                                strides=self.strides, 
                                                padding=self.padding))
        self.dense = layers.Dense(1)

        
        self.built = True

    def call(self, inputs):
        # must inputs[:,0] don't have nan

        h0 = tf.zeros_like(inputs[:,0])
        h = tf.tile(h0, [1, 1, 1, self.filters])
        c = tf.tile(h0, [1, 1, 1, self.filters])

        h_all = []
        for i in range(self.t):   
            x = inputs[:, i]
            if i > 0:   
                #FIXME: Don't know how to replace NaN with predict 
                #       value in tensorflow. Thus use mean value for 
                #       obs and predict images.
                #m = tf.stack([x, out], axis=0)
                #x = tf.experimental.numpy.nanmean(m, axis=0)

                a = tf.convert_to_tensor(x)
                b = tf.convert_to_tensor(out)
                x = tf.where(tf.math.is_nan(a), b, a)
                #mask = tf.where(tf.math.is_nan(x))
                #mask = x == x
                #print(mask)
                #x = tf.tensor_scatter_nd_update(x, mask, out[mask])
                #x[mask] = a[mask]

            out, [h, c] = self.convlstm[i](x, [h, c])
            out = self.dense(out)
            h_all.append(h)
        
        return tf.stack(h_all, axis=1)


if __name__ == '__main__':

    # inputs 
    in_l3 = Input(shape=(7, 112, 112, 1))
    in_l4 = Input(shape=(7, 112, 112, 8))
    
    # preprocess l3
    out_l3 = DIConvLSTM(filters=8, kernel_size=5)(in_l3)
    out_l3 = tf.keras.layers.BatchNormalization()(out_l3)
    out_l3 = tf.keras.layers.ReLU()(out_l3)
    print(out_l3.shape)

    # preprocess l4
    out_l4 = tf.keras.layers.ConvLSTM2D(8, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(in_l4)
    out_l4 = tf.keras.layers.BatchNormalization()(out_l4)
    out_l4 = tf.keras.layers.ReLU()(out_l4)
    print(out_l4.shape)

    out = tf.keras.layers.Add()([out_l3, out_l4, in_l4])
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([in_l3, in_l4], out)
    mdl.summary()
