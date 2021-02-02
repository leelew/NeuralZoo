from MetReg.base.base_model import BaseModel
from sklearn import gaussian_process


class GaussianProcessRegressor(BaseModel):
    """[summary]"""

    def __init__(self,
                 alpha=1e-10,
                 thetaL=1e-5,
                 thetaU=1e5,
                 random_state=None,
                 ):
        self.alpha = float(alpha)
        self.thetaL = float(thetaL)
        self.thetaU = float(thetaU)
        self.random_state = random_state
        self.regressor = None
        self.scaler = None

    def fit(self, X, y=None):
        n_features = X.shape[-1]
        kernel = gaussian_process.kernels.RBF(
            length_scale=[1.0]*n_features,
            length_scale_bounds=[(self.thetaL, self.thetaU)]*n_features)

        self.regressor = gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer='fmin_l_bfgs_b',
            alpha=self.alpha,
            copy_X_train=True,  # copy training data. care memory.
            random_state=self.random_state,
            normalize_y=True,
        )

        self.regressor.fit(X, y)
        return self

    @staticmethod
    def get_hyperparameter_search_space():
        pass

    def __repr__(self):
        return {'short name': 'GP',
                'name': 'Gaussian Process Regression'}
