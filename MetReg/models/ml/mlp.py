from MetReg.base.base_model import BaseModel
from sklearn import neural_network
import numpy as np
np.random.seed(1)

class MLPRegressor(BaseModel):

    def __init__(self,
                 hidden_layers_sizes=(16,),
                 activation='logistic',
                 alpha=0.0001,
                 learning_rate_init=0.001, early_stopping=False, solver='adam', batch_size='auto',
                 n_iter_no_change=10, tol=0.0001, shuffle=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                 validation_fraction=None, random_state=None, verbose=0):

        self.hidden_layers_sizes = hidden_layers_sizes
        self.activation = activation
        self.regressor = None
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.beta_1 = beta_1

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.regressor = neural_network.MLPRegressor(
            hidden_layer_sizes=self.hidden_layers_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            shuffle=self.shuffle,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=True,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            beta_1=self.beta_2,
            beta_2=self.beta_1,
            epsilon=self.epsilon,
        )

        self.regressor.fit(X, y)
        return self
