import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from model.tree_causality import CausalTree


class CausalLSTMNodeCell(layers.Layer):
    """node calculation of causality-structured LSTM, this class implemented the graph calculation of single node in causality-structure for a single cell.

    Call arguments:
    inputs: A 3D tensor with shape [batch_size, 1, num_nodes*num_grids]
    h: hidden state from last timestep, a 3D tensor with shape [batch_size, 1, num_hiddens]
    c: cell state from last timestep, a 3D tensor with shape [batch_size, 1, num_hiddens]
    n: causality state from last timestep, a 3D tensor with shape [batch_size, 1, num_hiddens]
    """
    def __init__(self, num_hiddens=16, num_children=3):
        super().__init__()

        self.num_hiddens = num_hiddens
        self.num_children = num_children

        # horiziontal forward, i.e., standard LSTM
        self._ifo_x = layers.Dense(3 * num_hiddens, activation=None)
        self._ifo_h = layers.Dense(3 * num_hiddens, activation=None)
        self._a_x = layers.Dense(num_hiddens, activation=None)
        self._a_h = layers.Dense(num_hiddens, activation=None)

        # vertical forward
        if num_children == 0:
            pass  # print()#'This is a leaf node')
        else:
            self._r_child_x = layers.Dense(num_children * num_hiddens,
                                           activation=None)
            self._r_child_h = layers.Dense(num_children * num_hiddens,
                                           activation=None)

        self._n_1_x = layers.Dense(num_hiddens, activation=None)
        self._n_1_h = layers.Dense(num_hiddens, activation=None)
        self._n_2_x = layers.Dense(num_hiddens, activation=None)
        self._n_2_h = layers.Dense(num_hiddens, activation=None)

    def _horizontal_forward(self, inputs, h_prev, c_prev):
        """forward pass of horizontal pass."""
        # generate input, forget, output gates
        ifo = tf.sigmoid(self._ifo_x(inputs) + self._ifo_h(h_prev))
        i, f, o = tf.split(ifo, 3, axis=-1)

        # generate current information state
        a = tf.math.tanh(self._a_x(inputs) + self._a_h(h_prev))

        # generate current cell state
        c = tf.math.multiply(i, a) + tf.math.multiply(f, c_prev)

        # generate current hidden state
        h = tf.math.multiply(o, tf.math.tanh(c))

        return h, c

    def _vertical_forward(self, inputs, h_prev, h, child_n=None):
        """forward pass of vertical pass"""
        if self.num_children == 0:
            r = 0
        else:
            # generate intermediate variable for neighborhood
            child_r = tf.sigmoid(
                self._r_child_x(inputs) +
                self._r_child_h(h_prev))  #[b, 1, num_hiddens*num_childs]
            child_r = tf.reshape(child_r,
                                 [-1, 1, self.num_hiddens, self.num_children])

            # (num_child, None, num_hiddens)
            child_r_n = tf.math.multiply(
                child_r, child_n)  # [b, 1, units, num_children]
            r = tf.reduce_sum(child_r_n, axis=-1)  # (None, num_hiddens)

        # generate weight for neighborhood and hidden state
        n_1 = tf.sigmoid(self._n_1_x(inputs) +
                         self._n_1_h(h_prev))  #[b, 1, num_hidden]
        n_2 = tf.sigmoid(self._n_2_x(inputs) + self._n_2_h(h_prev))

        # generate current neighborhood state
        n = tf.math.multiply(n_1, r) + tf.math.multiply(n_2, h)

        return n  # [b, 1, num_hidden]

    def call(self, inputs, h_prev, c_prev, child_n=None):
        h, c = self._horizontal_forward(inputs, h_prev, c_prev)
        n = self._vertical_forward(inputs, h_prev, h, child_n)
        return n, h, c


class CausalLSTMCell(layers.Layer):
    """Cell for Causality-structured LSTM, this class implemented the graph 
    calculation of all nodes in causality-structure for a single cell.

    Args:

    Call arguments:
        inputs: A 3D tensor with shape [batch_size, 1, num_grid*num_feature]
        h: hidden state from last timestep, a 4D tensor with shape [batch_size, 1, num_hiddens, num_nodes]
        c: cell state from last timestep, a 4D tensor with shape [batch_size, 1, num_hiddens, num_nodes]
        n: causality state from last timestep, a 4D tensor with shape [batch_size, 1, num_hiddens, num_nodes]
    """
    def __init__(self,
                 num_hiddens=16,
                 num_nodes=6,
                 len_outputs=1,
                 children=None,
                 child_input_idx=None,
                 child_state_idx=None):
        super().__init__()

        self.num_hiddens = num_hiddens
        self.num_nodes = num_nodes
        self.child_input_idx = child_input_idx  # list [[1,2,3],[4,5,6]]
        self.child_state_idx = child_state_idx
        self.children = children

        self.clstm_node_layers = []
        for i in range(num_nodes):
            self.clstm_node_layers.append(
                CausalLSTMNodeCell(num_hiddens, children[i]))

        self.out_linear_layer = layers.Dense(len_outputs, activation=None)

    def _update_state(self, state, state_new, state_idx):
        """update all nodes state by new state for each nodes

        Args:
            state: A 4D tensor with shape [batch_size, 1, num_hiddens, num_nodes]
            state_new: A 4D tensor with shape [batch_size, 1, num_hiddens, 1]
            state_idx (int): The index of node
        """
        if state_idx == 0:
            state = tf.concat([state_new, state[:, :, :, state_idx + 1:]],
                              axis=-1)
        elif state_idx == self.num_nodes:
            state = tf.concat([state[:, :, :, :state_idx], state_new], axis=-1)
        else:
            state = tf.concat([
                state[:, :, :, :state_idx], state_new, state[:, :, :,
                                                             state_idx + 1:]
            ],
                              axis=-1)
        return state

    def call(self, inputs, h, c, n):

        #TODO: Select feature for each node over all grids. If this cause poor
        #      performance, the alternative way is using mean of all features on
        #      each node.
        for i in range(self.num_nodes):

            # prepare inputs of each nodes
            _in_x = tf.concat(
                [inputs[:, :, ::k] for k in self.child_input_idx[i]],
                axis=-1)  # [b, 1, f]
            _h, _c = h[:, :, :, i], c[:, :, :, i]  # [b, 1, units]

            # this node have no parents
            if self.children[i] == 0:
                n_new, h_new, c_new = self.clstm_node_layers[i](_in_x,
                                                                _h,
                                                                _c,
                                                                child_n=None)

            # have parents
            else:
                # make parents causality states
                child_n = tf.stack(
                    [n[:, :, :, j] for j in self.child_state_idx[i]],
                    axis=-1)  # (b, 1, units, num_nodes)
                n_new, h_new, c_new = self.clstm_node_layers[i](
                    _in_x, _h, _c, child_n)  # (b, 1, units)
            n = self._update_state(n, n_new[:, :, :, tf.newaxis], i)
            h = self._update_state(h, h_new[:, :, :, tf.newaxis], i)
            c = self._update_state(c, c_new[:, :, :, tf.newaxis], i)

        out = self.out_linear_layer(n[:, :, :, -1])

        return out, (h, c, n)


class CausalLSTM(Model):
    """causality-structured LSTM (CLSTM)

    Args:
        num_nodes (int): The number of nodes in graph calculation.
        num_hiddens (int): The number of cells in hidden layers. 
        children (nest list): A nest list contain the parent nodes of nodes in causality-structure.
        child_input_idx ([type]): [description]
        child_state_idx ([type]): [description]
        return_sequences (bool, optional): Whether to return the last output. 
            in the output sequence, or the full sequence. Defaults to True.

    Call arguments:
        inputs: A 3D tensor with shape [batch_size, timesteps, num_grid*num_feature]
    """
    def __init__(self,
                 len_inputs,
                 num_nodes,
                 num_hiddens,
                 len_outputs,
                 children,
                 child_input_idx,
                 child_state_idx,
                 return_sequences=True,
                 **kwargs):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.len_inputs = len_inputs
        self.return_sequences = return_sequences

        #TODO: Make sure that should CLSTM share weight of different CLSTM
        #      cell? Now we set different weight.
        self.clstm_layers = []
        for _ in range(self.len_inputs):
            self.clstm_layers.append(
                CausalLSTMCell(num_hiddens, num_nodes, len_outputs, children,
                               child_input_idx, child_state_idx))

    def call(self, inputs, h0=None, c0=None, n0=None):
        batch_size, timestep, nx = inputs.shape

        #TODO: Decide a properly initializers, now set to zeros.
        # Maybe should be setting to the same initializers with tensorflow.LSTM
        if h0 is None:
            h = tf.zeros(
                [tf.shape(inputs)[0], 1, self.num_hiddens, self.num_nodes])
        if c0 is None:
            c = tf.zeros(
                [tf.shape(inputs)[0], 1, self.num_hiddens, self.num_nodes])
        if n0 is None:
            n = tf.zeros(
                [tf.shape(inputs)[0], 1, self.num_hiddens, self.num_nodes])

        successive_outputs = []
        for i in range(self.len_inputs):
            out, (h, c, n) = self.clstm_layers[i](inputs[:, i:i + 1, :], h, c,
                                                  n)
            successive_outputs.append(out)

        if self.return_sequences is True:
            outputs = tf.concat(successive_outputs, axis=1)
        else:
            outputs = successive_outputs[-1]

        return outputs


def clstm_v1(X, y, len_inputs, len_outputs, num_hiddens, num_nodes,
             feature_params, corr_threshold, mic_threshold, flag, depth):
    """default version (v1.0.0) of causality-structured LSTM

    This default edition is proposed by Li et al. 2021, including two steps: 
    1) generate causality-structure by two correlation and two granger causality 
       test, which is implemented by `CausalTree`.
    2) training CLSTM according to causality-structure. CLSTM invovles a new 
       state named causality state to propagate causality information vertically 
       through causality-structure.

    Args:
        X (nd.array): A 4D nd.array with shape as [sample, timestep, lat, lon, nx]
        y (nd.array): A 4D nd.array with shape as [sample, timestep, lat, lon, 1]
        nx (int): The number of input features
        len_inputs ([type]): [description]
        len_outputs ([type]): [description]
        num_hiddens ([type]): [description]
        num_nodes ([type]): [description]
        feature_params ([type]): [description]
        corr_threshold ([type]): [description]
        mic_threshold ([type]): [description]
        flag ([type]): [description]
        depth ([type]): [description]
    """
    _, Nlat, Nlon, Nf = X.shape
    ct = CausalTree(
        num_features=len(feature_params),
        name_features=feature_params,
        #FIXME: Change the name of parameters
        corr_thresold=corr_threshold,
        mic_thresold=mic_threshold,
        flag=flag,
        depth=depth)
    children, child_input_idx, child_state_idx = ct(np.nanmean(X, axis=(1, 2)))

    inputs = tf.keras.layers.Input(shape=(len_inputs, Nlat * Nlon * Nf))

    x = CausalLSTM(num_nodes=len(children),
                   num_hiddens=num_hiddens,
                   len_inputs=len_inputs,
                   len_outputs=len_outputs,
                   children=children,
                   child_input_idx=child_input_idx,
                   child_state_idx=child_state_idx)(inputs)

    # build
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    # summary
    model.summary()

    return model
