# ==============================================================================
# Keras FConvLSTM layers 
#
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python.keras import activations
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.layers.recurrent import Recurrent
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops


class ConvRecurrent2D(Recurrent):
  """Abstract base class for convolutional recurrent layers.
  Do not use in a model -- it's not a functional layer!
  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
          dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
          specifying the strides of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, time, ..., channels)`
          while `channels_first` corresponds to
          inputs with shape `(batch, time, channels, ...)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      go_backwards: Boolean (default False).
          If True, rocess the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
  Input shape:
      5D tensor with shape `(num_samples, timesteps, channels, rows, cols)`.
  Output shape:
      - if `return_sequences`: 5D tensor with shape
          `(num_samples, timesteps, channels, rows, cols)`.
      - else, 4D tensor with shape `(num_samples, channels, rows, cols)`.
  # Masking
      This layer supports masking for input data with a variable number
      of timesteps. To introduce masks to your data,
      use an `Embedding` layer with the `mask_zero` parameter
      set to `True`.
      **Note:** for the time being, masking is only supported with Theano.
  # Note on using statefulness in RNNs
      You can set RNN layers to be 'stateful', which means that the states
      computed for the samples in one batch will be reused as initial states
      for the samples in the next batch.
      This assumes a one-to-one mapping between
      samples in different successive batches.
      To enable statefulness:
          - specify `stateful=True` in the layer constructor.
          - specify a fixed batch size for your model, by passing
              a `batch_input_size=(...)` to the first layer in your model.
              This is the expected shape of your inputs *including the batch
              size*.
              It should be a tuple of integers, e.g. `(32, 10, 100)`.
      To reset the states of your model, call `.reset_states()` on either
      a specific layer, or on your entire model.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               return_sequences=False,
               go_backwards=False,
               stateful=False,
               **kwargs):
    super(ConvRecurrent2D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                    'dilation_rate')
    self.return_sequences = return_sequences
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.input_spec = InputSpec(ndim=5)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[3]
      cols = input_shape[4]
    elif self.data_format == 'channels_last':
      rows = input_shape[2]
      cols = input_shape[3]
    rows = conv_utils.conv_output_length(
        rows,
        self.kernel_size[0],
        padding=self.padding,
        stride=self.strides[0],
        dilation=self.dilation_rate[0])
    cols = conv_utils.conv_output_length(
        cols,
        self.kernel_size[1],
        padding=self.padding,
        stride=self.strides[1],
        dilation=self.dilation_rate[1])
    if self.return_sequences:
      if self.data_format == 'channels_first':
        return tensor_shape.TensorShape(
            [input_shape[0], input_shape[1], self.filters, rows, cols])
      elif self.data_format == 'channels_last':
        return tensor_shape.TensorShape(
            [input_shape[0], input_shape[1], rows, cols, self.filters])
    else:
      if self.data_format == 'channels_first':
        return tensor_shape.TensorShape(
            [input_shape[0], self.filters, rows, cols])
      elif self.data_format == 'channels_last':
        return tensor_shape.TensorShape(
            [input_shape[0], rows, cols, self.filters])

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'return_sequences': self.return_sequences,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful
    }
    base_config = super(ConvRecurrent2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class GatedConvLSTM2D(ConvRecurrent2D):
  """Gated Convolutional LSTM.
  It is similar to an LSTM layer, but the input transformations
  and recurrent transformations are both convolutional.
  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
          dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
          specifying the strides of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, time, ..., channels)`
          while `channels_first` corresponds to
          inputs with shape `(batch, time, channels, ...)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state..
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      go_backwards: Boolean (default False).
          If True, rocess the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
  Input shape:
      - if data_format='channels_first'
          5D tensor with shape:
          `(samples,time, channels, rows, cols)`
      - if data_format='channels_last'
          5D tensor with shape:
          `(samples,time, rows, cols, channels)`
   Output shape:
      - if `return_sequences`
           - if data_format='channels_first'
              5D tensor with shape:
              `(samples, time, filters, output_row, output_col)`
           - if data_format='channels_last'
              5D tensor with shape:
              `(samples, time, output_row, output_col, filters)`
      - else
          - if data_format ='channels_first'
              4D tensor with shape:
              `(samples, filters, output_row, output_col)`
          - if data_format='channels_last'
              4D tensor with shape:
              `(samples, output_row, output_col, filters)`
          where o_row and o_col depend on the shape of the filter and
          the padding
  Raises:
      ValueError: in case of invalid constructor arguments.
  References:
      - [Convolutional LSTM Network: A Machine Learning Approach for
      Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
      The current implementation does not include the feedback loop on the
      cells output
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               return_sequences=False,
               go_backwards=False,
               stateful=False,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    super(GatedConvLSTM2D, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        return_sequences=return_sequences,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # TODO(fchollet): better handling of input spec
    self.input_spec = InputSpec(shape=input_shape)

    if self.stateful:
      self.reset_states()
    else:
      # initial states: 2 all-zero tensor of shape (filters)
      self.states = [None, None]

    if self.data_format == 'channels_first':
      channel_axis = 2
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    depthwise_kernel_shape = self.kernel_size + (input_dim, 1)
    pointwise_kernel_shape = (1, 1, input_dim, self.filters)
    self.depthwise_kernel_shape = depthwise_kernel_shape
    self.pointwise_kernel_shape = pointwise_kernel_shape
    recurrent_depthwise_kernel_shape = self.kernel_size + (self.filters, 1)
    recurrent_pointwise_kernel_shape = (1, 1, self.filters, self.filters)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.kernel_initializer,
        name='depthwise_kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.pointwise_kernel = self.add_weight(
        shape=pointwise_kernel_shape,
        initializer=self.kernel_initializer,
        name='pointwise_kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_depthwise_kernel = self.add_weight(
        shape=recurrent_depthwise_kernel_shape,
        initializer=self.recurrent_initializer,
        name='recurrent_depthwise_kernel',
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    self.recurrent_pointwise_kernel = self.add_weight(
        shape=recurrent_pointwise_kernel_shape,
        initializer=self.recurrent_initializer,
        name='recurrent_pointwise_kernel',
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.filters,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None

    gate_kernel_shape = (1, 1, input_dim, self.filters*3)
    recurrent_gate_kernel_shape = (1, 1, self.filters, self.filters*3)
    self.gate_kernel = self.add_weight(
        shape=gate_kernel_shape,
        initializer=initializers.constant(value=1.0/self.filters),
        name='gate_kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_gate_kernel = self.add_weight(
        shape=recurrent_gate_kernel_shape,
        initializer=initializers.constant(value=1.0/self.filters),
        name='recurrent_gate_kernel',
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    self.gate_bias = self.add_weight(
          shape=(self.filters*3,),
          initializer=self.bias_initializer,
          name='gate_bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)

    self.kernel_i = self.gate_kernel[:, :, :, :self.filters]
    self.recurrent_kernel_i = self.recurrent_gate_kernel[:, :, :, :self.filters]
    self.kernel_f = self.gate_kernel[:, :, :, self.filters:self.filters * 2]
    self.recurrent_kernel_f = self.recurrent_gate_kernel[:, :, :, self.filters:
                                                    self.filters * 2]
    self.kernel_o = self.gate_kernel[:, :, :, self.filters * 2:]
    self.recurrent_kernel_o = self.recurrent_gate_kernel[:, :, :, self.filters * 2:]

    if self.use_bias:
      self.bias_i = self.gate_bias[:self.filters]
      self.bias_f = self.gate_bias[self.filters:self.filters * 2]
      self.bias_o = self.gate_bias[self.filters * 2:]
      self.bias_c = self.bias
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_o = None
    self.built = True

  def get_initial_states(self, inputs):
    # (samples, timesteps, rows, cols, filters)
    initial_state = K.zeros_like(inputs)
    # (samples, rows, cols, filters)
    initial_state = K.sum(initial_state, axis=1)
    depthwise_shape = list(self.depthwise_kernel_shape)
    pointwise_shape = list(self.pointwise_kernel_shape)
    initial_state = self.input_conv(
        initial_state, K.zeros(tuple(depthwise_shape)), 
        K.zeros(tuple(pointwise_shape)), padding=self.padding)

    initial_states = [initial_state for _ in range(2)]
    return initial_states

  def reset_states(self):
    if not self.stateful:
      raise RuntimeError('Layer must be stateful.')
    input_shape = self.input_spec.shape
    output_shape = self._compute_output_shape(input_shape)
    if not input_shape[0]:
      raise ValueError('If a RNN is stateful, a complete '
                       'input_shape must be provided '
                       '(including batch size). '
                       'Got input shape: ' + str(input_shape))

    if self.return_sequences:
      out_row, out_col, out_filter = output_shape[2:]
    else:
      out_row, out_col, out_filter = output_shape[1:]

    if hasattr(self, 'states'):
      K.set_value(self.states[0],
                  np.zeros((input_shape[0], out_row, out_col, out_filter)))
      K.set_value(self.states[1],
                  np.zeros((input_shape[0], out_row, out_col, out_filter)))
    else:
      self.states = [
          K.zeros((input_shape[0], out_row, out_col, out_filter)), K.zeros(
              (input_shape[0], out_row, out_col, out_filter))
      ]

  def get_constants(self, inputs, training=None):
    constants = []
    if self.implementation == 0 and 0 < self.dropout < 1:
      ones = K.zeros_like(inputs)
      ones = K.sum(ones, axis=1)
      ones += 1

      def dropped_inputs():
        return K.dropout(ones, self.dropout)

      dp_mask = [
          K.in_train_phase(dropped_inputs, ones, training=training)
          for _ in range(4)
      ]
      constants.append(dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(4)])

    if 0 < self.recurrent_dropout < 1:
      depthwise_shape = list(self.depthwise_kernel_shape)
      pointwise_shape = list(self.pointwise_kernel_shape)
      ones = K.zeros_like(inputs)
      ones = K.sum(ones, axis=1)
      ones = self.input_conv(ones, K.zeros(depthwise_shape), 
             K.zeros(pointwise_shape), padding=self.padding)
      ones += 1.

      def dropped_inputs():  # pylint: disable=function-redefined
        return K.dropout(ones, self.recurrent_dropout)

      rec_dp_mask = [
          K.in_train_phase(dropped_inputs, ones, training=training)
          for _ in range(4)
      ]
      constants.append(rec_dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(4)])
    return constants

  def context_gating(self, x, w, rx, rw, b=None, padding='valid'):
    input_shape = x.get_shape().as_list()
    if self.data_format == 'channels_first':
      x = K.pool2d(x, (input_shape[2], input_shape[3]), pool_mode='avg')
      rx = K.pool2d(rx, (input_shape[2], input_shape[3]), pool_mode='avg')
    elif self.data_format == 'channels_last':
      x = K.pool2d(x, (input_shape[1], input_shape[2]), pool_mode='avg')
      rx = K.pool2d(rx, (input_shape[1], input_shape[2]), pool_mode='avg')
    conv_out1 = K.conv2d(
        x,
        w,
        strides=self.strides,
        padding=padding,
        data_format=self.data_format)
    conv_out2 = K.conv2d(
        rx,
        rw,
        strides=self.strides,
        padding=padding,
        data_format=self.data_format)
    conv_out = conv_out1 + conv_out2
    if b is not None:
      conv_out = K.bias_add(conv_out, b, data_format=self.data_format)
    return conv_out

  def input_conv(self, x, dw, pw, b=None, padding='valid'):
    conv_out = K.separable_conv2d(
        x,
        dw,
        pw,
        strides=self.strides,
        padding=padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)
    if b is not None:
      conv_out = K.bias_add(conv_out, b, data_format=self.data_format)
    return conv_out

  def recurrent_conv(self, x, dw, pw):
    conv_out = K.separable_conv2d(
        x, dw, pw, strides=(1, 1), padding='same', data_format=self.data_format)
    return conv_out

  def step(self, inputs, states):
    assert len(states) == 4
    h_tm1 = states[0]
    c_tm1 = states[1]
    dp_mask = states[2]
    rec_dp_mask = states[3]

    x_i = self.context_gating(
        inputs, self.kernel_i, h_tm1, self.recurrent_kernel_i, self.bias_i)
    x_f = self.context_gating(
        inputs, self.kernel_f, h_tm1, self.recurrent_kernel_f, self.bias_f)
    x_o = self.context_gating(
        inputs, self.kernel_o, h_tm1, self.recurrent_kernel_o, self.bias_o)

    x_c = self.input_conv(
        inputs * dp_mask[2], self.depthwise_kernel, self.pointwise_kernel, self.bias_c, padding=self.padding)
    h_c = self.recurrent_conv(h_tm1 * rec_dp_mask[2], self.recurrent_depthwise_kernel, self.recurrent_pointwise_kernel)

    i = self.recurrent_activation(x_i)
    f = self.recurrent_activation(x_f)
    o = self.recurrent_activation(x_o)
    c = f * c_tm1 + i * self.activation(x_c + h_c)
    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(GatedConvLSTM2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

