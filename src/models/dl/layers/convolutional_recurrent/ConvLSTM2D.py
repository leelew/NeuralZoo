import tensorflow as tf
from tensorflow.keras import Input, Model, layers


class ConvLSTMCell(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.padding = padding
        self.strides = strides
        self.activation = activation

    def build(self, input_shape):
        # get shape
        _, self.height, self.width, _ = input_shape

        # horiziontal forward, i.e., standard ConvLSTM
        self._ifo_x = layers.Conv2D(filters=3 * self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation=self.activation)
        self._ifo_h = layers.Conv2D(filters=3 * self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation=self.activation)
        self._a_x = layers.Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  padding=self.padding,
                                  activation=self.activation)
        self._a_h = layers.Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  padding=self.padding,
                                  activation=self.activation)

        self.built =True

    def call(self, inputs, states, training=None):
        """forward pass of horizontal pass."""
        # previous hidden, cell state
        h_prev, c_prev = states[0], states[1]

        ifo = tf.sigmoid(self._ifo_x(inputs) + self._ifo_h(h_prev))
        i, f, o = tf.split(ifo, 3, axis=-1)
        a = tf.math.tanh(self._a_x(inputs) + self._a_h(h_prev))
        c = tf.math.multiply(i, a) + tf.math.multiply(f, c_prev)
        h = tf.math.multiply(o, tf.math.tanh(c))
        return h, [h,c]




class ConvLSTM2D(layers.Layer): #TODO:Add iterative/teacher forcing/schedule sample forecast methods
 
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=True,
                 return_state=True,
                 **kwargs):
        super().__init__()

        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.padding=padding
        self.strides = strides
        self.return_sequences = return_sequences
        self.return_state = return_state

    def build(self, input_shape):
        _, self.t, self.height, self.width, _ = input_shape

        self.clstm = []
        for i in range(self.t):
            self.clstm.append(ConvLSTMCell(filters=self.filters, 
                                                 kernel_size=self.kernel_size, 
                                                 strides=self.strides,
                                                 padding=self.padding))
        self.dense = layers.Dense(1, activation=None)

    def call(self, inputs, initial_state=None):
        """Causality LSTM"""
        if initial_state is None: states = self.initial_tree_states(inputs)
        else: states = initial_state

        out = []
        for i in range(self.t): 
            states = self.clstm[i](inputs[:, i], states)
            out.append(self.dense(states[0]))  # output = tf.stack(output, axis=1)

        if self.return_sequences: out = tf.stack(out, axis=1)
        if self.return_state: return out, states
        else: return out

    def initial_tree_states(self, inputs):
        """initial tree state using default LSTM."""
        init = tf.zeros_like(inputs[:, 0, :, :, :1])
        state = tf.tile(init, [1, 1, 1, self.filters])
        return [state, state]


if __name__ == '__main__':
    encoder_inputs = Input((1, 112, 112, 8))
    decoder_inputs = Input((1, 112, 112, 5))

    outputs, states = ConvLSTM2D(filters=16,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         return_sequences=True,
                         return_state=True)(encoder_inputs)
    print(outputs.shape)
    mdl = Model([encoder_inputs, decoder_inputs], outputs)
    mdl.summary()
