import tensorflow as tf
from tensorflow.keras import Input, Model, layers


class CausalConvLSTMNodeCell(layers.Layer):
    """Cell class for each node of Causality-structured ConvLSTM layer.

    Args:
      num_children ([type]): [description]
      filters ([type]): [description]
      kernel_size ([type]): [description]
      strides (int, optional): [description]. Defaults to 1.
      padding (str, optional): [description]. Defaults to 'valid'.
      data_format ([type], optional): [description]. Defaults to None.
      dilation_rate (int, optional): [description]. Defaults to 1.
      activation (str, optional): [description]. Defaults to 'tanh'.
      recurrent_activation (str, optional): [description]. Defaults to 'hard_sigmoid'.
      use_bias (bool, optional): [description]. Defaults to True.
      kernel_initializer (str, optional): [description]. Defaults to 'glorot_uniform'.
      recurrent_initializer (str, optional): [description]. Defaults to 'orthogonal'.
      bias_initializer (str, optional): [description]. Defaults to 'zeros'.
      unit_forget_bias (bool, optional): [description]. Defaults to True.
      kernel_regularizer ([type], optional): [description]. Defaults to None.
      recurrent_regularizer ([type], optional): [description]. Defaults to None.
      bias_regularizer ([type], optional): [description]. Defaults to None.
      kernel_constraint ([type], optional): [description]. Defaults to None.
      recurrent_constraint ([type], optional): [description]. Defaults to None.
      bias_constraint ([type], optional): [description]. Defaults to None.
      dropout (float, optional): [description]. Defaults to 0.0.
      recurrent_dropout (float, optional): [description]. Defaults to 0.0.
    Call arguments:
      inputs: A 4D tensor. 
      states: List of state tensors corresponding to the previous timestep.
      child_n: A 5D tensor. Causality states from children nodes.
    Input shape:
      - 4D tensor with shape: `(samples, height, width, channels)`
    Output shape:
      - causality (n), hidden (h) and cell (c) states with shape: `(samples, height, width, filters)`

    Reference:
      - [Li et al. 2022]: see equation (1-10) in paper.
    """
    def __init__(self,
                 num_children,
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
        self.num_children = num_children
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

        # vertical forward
        if self.num_children == 0:
            pass  # print()#'This is a leaf node')
        else:
            self._r_child_x = layers.Conv2D(filters=self.num_children * self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation)
            self._r_child_h = layers.Conv2D(filters=self.num_children * self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation)
        self._n_1_x = layers.Conv2D(filters=self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation=self.activation)
        self._n_1_h = layers.Conv2D(filters=self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation=self.activation)
        self._n_2_x = layers.Conv2D(filters=self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation=self.activation)
        self._n_2_h = layers.Conv2D(filters=self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation=self.activation)
        self.built =True

    def _horizontal_forward(self, inputs, states, training=None):
        """forward pass of horizontal pass."""
        # previous hidden, cell state
        h_prev, c_prev = states[0], states[1]

        ifo = tf.sigmoid(self._ifo_x(inputs) + self._ifo_h(h_prev))
        i, f, o = tf.split(ifo, 3, axis=-1)
        a = tf.math.tanh(self._a_x(inputs) + self._a_h(h_prev))
        c = tf.math.multiply(i, a) + tf.math.multiply(f, c_prev)
        h = tf.math.multiply(o, tf.math.tanh(c))
        return h, [h,c]

    def _vertical_forward(self, inputs, h_prev, h, child_n=None):
        """forward pass of vertical pass."""
        # generate intermediate variable for neighborhood
        if self.num_children == 0: r = 0
        else:
            if child_n is None: raise KeyError('Give causality states of children nodes')
            # -> (None, lat, lon, num_hiddens * num_child)
            child_r = tf.sigmoid(self._r_child_x(inputs) + self._r_child_h(h_prev))
            # -> (num_child, None, lat, lon, num_hiddens)
            child_r = tf.reshape(child_r, \
                [self.num_children, -1, self.height, self.width, self.filters])
            # -> (num_child, None, lat, lon, num_hiddens)
            child_r_n = tf.math.multiply(child_r, child_n)
            # -> (None, lat, lon, num_hiddens)
            r = tf.reduce_sum(child_r_n, axis=0)
        # generate weight for neighborhood and hidden state
        n_1 = tf.sigmoid(self._n_1_x(inputs) + self._n_1_h(h_prev))
        n_2 = tf.sigmoid(self._n_2_x(inputs) + self._n_2_h(h_prev))
        # generate current neighborhood state
        n = tf.math.multiply(n_1, r) + tf.math.multiply(n_2, h)
        return n

    def call(self, inputs, states, child_n=None):
        _, [h, c] = self._horizontal_forward(inputs, states)
        n = self._vertical_forward(inputs, states[0], h, child_n)
        return [n, h, c]


class CausalConvLSTMCell(layers.Layer):
    """Cell class of Causality-structured ConvLSTM layer.

    Args:
      num_children ([type]): [description]
      filters ([type]): [description]
      kernel_size ([type]): [description]
      strides (int, optional): [description]. Defaults to 1.
      padding (str, optional): [description]. Defaults to 'valid'.
      data_format ([type], optional): [description]. Defaults to None.
      dilation_rate (int, optional): [description]. Defaults to 1.
      activation (str, optional): [description]. Defaults to 'tanh'.
      recurrent_activation (str, optional): [description]. Defaults to 'hard_sigmoid'.
      use_bias (bool, optional): [description]. Defaults to True.
      kernel_initializer (str, optional): [description]. Defaults to 'glorot_uniform'.
      recurrent_initializer (str, optional): [description]. Defaults to 'orthogonal'.
      bias_initializer (str, optional): [description]. Defaults to 'zeros'.
      unit_forget_bias (bool, optional): [description]. Defaults to True.
      kernel_regularizer ([type], optional): [description]. Defaults to None.
      recurrent_regularizer ([type], optional): [description]. Defaults to None.
      bias_regularizer ([type], optional): [description]. Defaults to None.
      kernel_constraint ([type], optional): [description]. Defaults to None.
      recurrent_constraint ([type], optional): [description]. Defaults to None.
      bias_constraint ([type], optional): [description]. Defaults to None.
      dropout (float, optional): [description]. Defaults to 0.0.
      recurrent_dropout (float, optional): [description]. Defaults to 0.0.
    Call arguments:
      inputs: A 4D tensor. 
      states: A 5D tensor with shape: `(num_nodes, samples, height, width, filters)`. 
        List of state tensors corresponding to the previous timestep. 
    Input shape:
      - 4D tensor with shape: `(samples, height, width, channels)`
    Output shape:
      - causality (n), hidden (h) and cell (c) states with shape: `(num_nodes, samples, height, width, filters)`

    Reference:
      - [Li et al. 2022]
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 child: dict,
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

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding=padding
        self.strides = strides
        self.child_num = child['child_num']
        self.child_input_idx = child['child_input_idx']
        self.child_state_idx = child['child_state_idx']
        self.num_nodes = len(self.child_num)

    def build(self, input_shape):
        # get shape
        _, self.height, self.width, _ = input_shape

        self.lstm = []
        for i in range(self.num_nodes):
            self.lstm.append(CausalConvLSTMNodeCell(num_children=self.child_num[i], 
                                                    filters=self.filters, 
                                                    kernel_size=self.kernel_size, 
                                                    strides=self.strides, 
                                                    padding=self.padding))
        self.built =True

    def call(self, inputs, states):
        # get states
        n, c, h = states[0], states[1], states[2]

        # forward for each nodes in causality structure
        for i in range(self.num_nodes):
            _in_x = tf.stack([inputs[:, :, :, k] for k in self.child_input_idx[i]], axis=-1)
            _h, _c = h[i], c[i]

            # get states for each node
            if self.child_num[i] == 0: 
                n_new, h_new, c_new = self.lstm[i](_in_x, [_h, _c], child_n=None)
            else:
                child_n = tf.stack([n[j] for j in self.child_state_idx[i]], axis=0)
                n_new, h_new, c_new = self.lstm[i](_in_x, [_h, _c], child_n)

            # update states for each node
            n = self.update_state(n, n_new[tf.newaxis], i)
            h = self.update_state(h, h_new[tf.newaxis], i)
            c = self.update_state(c, c_new[tf.newaxis], i)
        return [n, h, c]

    def update_state(self, state, state_new, state_idx):
        """update states in tensorflow manner."""
        if state_idx == 0: state = tf.concat([state_new, state[state_idx + 1:]], axis=0)
        elif state_idx == self.num_nodes: state = tf.concat([state[:state_idx], state_new], axis=0)
        else: state = tf.concat([state[:state_idx], state_new, state[state_idx + 1:]],axis=0)
        return state


class CausalConvLSTM2D(layers.Layer):
    """Causality-structured ConvLSTM layer.

    Args:
      num_children ([type]): [description]
      filters ([type]): [description]
      kernel_size ([type]): [description]
      strides (int, optional): [description]. Defaults to 1.
      padding (str, optional): [description]. Defaults to 'valid'.
      data_format ([type], optional): [description]. Defaults to None.
      dilation_rate (int, optional): [description]. Defaults to 1.
      activation (str, optional): [description]. Defaults to 'tanh'.
      recurrent_activation (str, optional): [description]. Defaults to 'hard_sigmoid'.
      use_bias (bool, optional): [description]. Defaults to True.
      kernel_initializer (str, optional): [description]. Defaults to 'glorot_uniform'.
      recurrent_initializer (str, optional): [description]. Defaults to 'orthogonal'.
      bias_initializer (str, optional): [description]. Defaults to 'zeros'.
      unit_forget_bias (bool, optional): [description]. Defaults to True.
      kernel_regularizer ([type], optional): [description]. Defaults to None.
      recurrent_regularizer ([type], optional): [description]. Defaults to None.
      bias_regularizer ([type], optional): [description]. Defaults to None.
      kernel_constraint ([type], optional): [description]. Defaults to None.
      recurrent_constraint ([type], optional): [description]. Defaults to None.
      bias_constraint ([type], optional): [description]. Defaults to None.
      dropout (float, optional): [description]. Defaults to 0.0.
      recurrent_dropout (float, optional): [description]. Defaults to 0.0.
    Call arguments:
      inputs: A 4D tensor. 
      states: A 5D tensor with shape: `(num_nodes, samples, height, width, filters)`. 
        List of state tensors corresponding to the previous timestep. 
    Input shape:
      - 4D tensor with shape: `(samples, height, width, channels)`
    Output shape:
      - causality (n), hidden (h) and cell (c) states with shape: `(num_nodes, samples, height, width, filters)`

    Reference:
      - [Li et al. 2022]
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 child: dict,
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

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding=padding
        self.strides = strides
        self.child = child
        self.num_nodes = len(child['child_num'])
        self.return_sequences = return_sequences
        self.return_state = return_state

    def build(self, input_shape):
        _, self.t, self.height, self.width, _ = input_shape

        self.clstm = []
        for i in range(self.t):
            self.clstm.append(CausalConvLSTMCell(filters=self.filters, 
                                                 kernel_size=self.kernel_size, 
                                                 child=self.child, 
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
            out.append(self.dense(states[0][-1]))  # output = tf.stack(output, axis=1)

        if self.return_sequences: out = tf.stack(out, axis=1)
        if self.return_state: return out, states
        else: return out

    def initial_tree_states(self, inputs):
        """initial tree state using default LSTM."""
        init = tf.zeros_like(inputs[tf.newaxis, :, 0, :, :, :1])
        state = tf.tile(init, [self.num_nodes, 1, 1, 1, self.filters])
        return [state, state, state]


if __name__ == '__main__':
    child = {
        'child_num': [0, 0, 2],
        'child_input_idx': [[0], [1], [2]],
        'child_state_idx': [[], [], [0, 1]]
    }
    encoder_inputs = Input((7, 112, 112, 8))
    decoder_inputs = Input((5, 112, 112, 5))

    outputs, states = CausalConvLSTM2D(filters=16,
                         kernel_size=3,
                         child=child,
                         strides=1,
                         padding='same',
                         return_sequences=True,
                         return_state=True)(encoder_inputs)
    print(outputs.shape)
    outputs, states = CausalConvLSTM2D(filters=16,
                         kernel_size=3,
                         child=child,
                         strides=1,
                         padding='same',
                         return_sequences=True,
                         return_state=True)(decoder_inputs, initial_state=states)
    
    mdl = Model([encoder_inputs, decoder_inputs], outputs)
    mdl.summary()
