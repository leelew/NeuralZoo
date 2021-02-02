
class Spatitemporal_LSTM(Model):

    def __init__(self, num_hidden, width, filter_size, stride, layer_norm, channel):
        """cell class for spatiotemporal LSTM.


        Ref: https://arxiv.org/pdf/1804.06300.pdf.

        Args:
            num_hidden ([type]): [description]
            width ([type]): [description]
            filter_size ([type]): [description]
            stride ([type]): [description]
            layer_norm ([type]): [description]
            channel ([type]): [description]
        """
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size//2
        self._forget_bias = 1.0
        self.conv_x = Sequential(
            Conv2D(num_hidden, num_hidden * 7, 
            kernel_size=filter_size, stride=stride, padding=self.padding),
            LayerNormalization())
        self.conv_h = Sequential(
            Conv2D(num_hidden, num_hidden * 4, 
            kernel_size=filter_size, stride=stride, padding=self.padding),
            LayerNormalization()
        )
        self.conv_m = Sequential(
            Conv2D(num_hidden, num_hidden * 3, 
            kernel_size=filter_size, stride=stride, padding=self.padding),
            LayerNormalization()
        )
        self.conv_o = Sequential(
            Conv2D(num_hidden*2, num_hidden, 
            kernel_size=filter_size, stride=stride, padding=self.padding),
            LayerNormalization()
        )
        self.conv_last = Conv2D(num_hidden * 2, num_hidden, kernel_size=1, 
        stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = tf.split(x_concat, num_or_size_splits=7, dim=-1)
        i_h, f_h, g_h, o_h = tf.split(h_concat, num_or_size_splits=7, dim=-1)
        i_m, f_m, g_m = tf.split(m_concat, num_or_size_splits=7, dim=-1)


        i_t = sigmoid(i_x+i_h)
        f_t = sigmoid(f_x+f_h+self._forget_bias)
        g_t = sigmoid(g_x+g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = sigmoid(i_x_prime + i_m)
        f_t_prime = sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = concatenate([c_new, m_new], axis=-1)
        o_t = sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * tanh(self.conv_last(mem))


        return h_new, c_new, m_new


class predRNN(Model):

    def __init__(self, num_layers, num_hidden, patch_size, width,channel):
        """[summary]

        Args:
            num_layers ([type]): [description]
            num_hidden ([type]): [description]
            patch_size ([type]): [description]
            width ([type]): [description]
            channel ([type]): [description]
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden

        cell_list = []

        for i in range(num_layers):
            cell_list.append(
                Spatitemporal_LSTM(num_hidden=num_hidden[i],
                                    num_feature = num_feature)
            )
        
        self.conv_last = Conv2D(num_hidden[num_layers-1])

    def call(self):

        Nbatch, Nlength, Nfeature, Nlat, Nlon=inputs.shape

        # init
        hidden_state = []
        cell_state = []
        predict = []

        # init
        for i in range(self.num_layers):
            zeros = tf.zeros([Nbatch, self.num_hidden[i], Nlat, Nlon])
            hidden_state.append(zeros)
            cell_state.append(zeros)

        memory = tf.zeros([Nbatch, self.num_hidden[0], Nlat, Nlon])

        
"""
		for t in range(Nlength):
			for i in range(self.num_layers):
			
				if i==0:
					cell_state[0], memory_state, hidden_state[0] = self.cell_list[0](
						inputs[:, t, :,:,:], hidden_state[0], cell_state[0], memory_state)
				else:
					cell_state[i], memory_state, hidden_state[i] = self.cell_list[i](
						hidden_state[i-1], hidden_state[i], cell_state[i], memory_state)
		
			
			predict.append(self.conv_last(hidden_state[self.num_layers-1]))
		
        predict = tf.stack(predict, dim=0)
"""