from sklearn import neighbors
from MetReg.base.base_model import BaseModel
import numpy as np
np.random.seed(1)

class KNNRegressor(BaseModel):

    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 p=2,
                 random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state
        self.regressor = None

    def fit(self, X, y):

        self.regressor = neighbors.KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p
        )

        self.regressor.fit(X, y)

        return self
