
from sklearn import svm
from MetReg.base.base_model import BaseModel
import numpy as np
np.random.seed(1)


class LinearSVRegressor(BaseModel):

    def __init__(self,
                 epsilon=0.1,
                 loss='epsilon_insensitive',
                 dual=True,
                 tol=0.0001,
                 C=10,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 random_state=None):
        self.epsilon = epsilon
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.regressor = None

    def fit(self, X, y):
        self.regressor = svm.LinearSVR(
            epsilon=self.epsilon,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            random_state=self.random_state)

        self.regressor.fit(X, y)
        return self


class SVRegressor(BaseModel):

    def __init__(self,
                 kernel='rbf',
                 C=0.6,
                 gamma=0.1,
                 epsilon=0.1,
                 tol=0.001,
                 shrinking=True,
                 degree=3,
                 coef0=0.0,
                 verbose=False,
                 max_iter=-1,
                 random_state=None
                 ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.shrinking = shrinking
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):

        self.regressor = svm.SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            tol=self.tol,
            shrinking=self.shrinking,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            verbose=self.verbose,
            max_iter=self.max_iter)

        self.regressor.fit(X, y)
        return self
