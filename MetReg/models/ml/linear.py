
import warnings

import numpy as np
from sklearn import linear_model

warnings.filterwarnings("ignore")


class LR():
    """implementation of base linear regression.

    Ordinary least squares Linear Regression.LinearRegression fits a linear 
    model with coefficients w = (w1, …, wp) to minimize the residual sum of 
    squares between the observed targets in the dataset, and the targets 
    predicted by the linear approximation.

    Args:
        fit_intercept (bool, optional): 
            Whether to calculate the intercept for this model. If set to False, 
            no intercept will be used in calculations (i.e. data is expected 
            to be centered). Defaults to True.
        normalize (bool, optional): 
            This parameter is ignored when `fit_intercept` is set to False. If 
            True, the regressors X will be normalized before regression by 
            subtracting the mean and dividing by the l2-norm. Defaults to False.

    Attributes:
        coef_:
            Estimated coefficients for the linear regression problem. If 
            multiple targets are passed during the fit (y 2D), this is a 2D 
            array of shape (n_targets, n_features), while if only one target 
            is passed, this is a 1D array of length n_features.
    """

    def __init__(self,
                 fit_intercept=True,
                 normalize=True,
                 **kwargs):
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        self.regressor = None

    def fit(self, X, y):
        self.regressor = linear_model.LinearRegression(
            fit_intercept=self.fit_intercept,
            normalize=self.normalize)


        self.regressor.fit(X,y)
        return self


class Ridge():
    """Linear least squares with l2 regularization.

    Minimizes the objective function::
    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

    cv: TODO(lewlee): implement of `grid_search` edition. 
        Ridge regression with built-in cross-validation.

    Args:
        cv (bool, optional): [description]. Defaults to True.
        cv_alphas (ndarray, optional): 
            Array of alpha values to try.
        cv_num_folds (int, optional):
            cross-validation generator or an iterable, Determines the 
            cross-validation splitting strategy.

            Possible inputs for cv are:
            - None, to use the efficient Leave-One-Out cross-validation
            - integer, to specify the number of folds.
        alpha (float, optional): 
            Regularization strength; must be a positive float. Regularization
            improves the conditioning of the problem and reduces the variance 
            of the estimates. Larger values specify stronger regularization.
            Alpha corresponds to ``1 / (2C)`` in other linear models. If an 
            array is passed, penalties are assumed to be specific to the 
            targets. Hence they must correspond in number. Defaults to 1.0.
        max_iter ([type], optional): 
            Maximum number of iterations for conjugate gradient solver.     
            For 'sparse_cg' and 'lsqr' solvers, the default value is determined
            by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
            Defaults to None.
        tol ([type], optional): 
            Precision of the solution. Defaults to 1e-3.
        solver (str, optional): 
            - 'auto' chooses the solver automatically based on the type of data.
            - 'svd' uses a Singular Value Decomposition of X to compute the 
                Ridge coefficients. More stable for singular matrices than 
                'cholesky'.
            - 'cholesky' uses the standard scipy.linalg.solve function to
                obtain a closed-form solution.. Defaults to 'auto'.
            - 'lsqr' uses the dedicated regularized least-squares routine
                scipy.sparse.linalg.lsqr. It is the fastest and uses an 
                iterative procedure.
        fit_intercept (bool, optional): see class `LR`. Defaults to True.
        normalize (bool, optional): see class `LR`. Defaults to False.

    Attributes:
        coef_ : 
            ndarray of shape (n_features,) or (n_targets, n_features) 
            Weight vector(s).
        intercept_ : 
            float or ndarray of shape (n_targets,) Independent term in 
            decision function.
        n_iter_ : None or ndarray of shape (n_targets,)
            Actual number of iterations for each target. Available only for
            lsqr solvers. Other solvers will return None.
    """

    def __init__(self,
                 max_iter=1000,
                 tol=1e-4,
                 fit_intercept=True,
                 normalize=True,
                 alpha=1.0,
                 solver='auto',
                 cv=False,
                 cv_alphas=[0.1,1.0,10.0],
                 cv_num_folds=5,
                 **kwargs):
        # common setting
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        # default setting
        self.alpha = alpha
        self.solver = solver

        # cross-validation setting
        self.cv = cv
        self.cv_alphas = cv_alphas
        self.cv_num_folds = cv_num_folds

        self.regressor = None

    def fit(self, X, y):
        if self.cv:
            self.regressor = linear_model.RidgeCV(
                alphas=self.cv_alphas,
                cv=self.cv_num_folds,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,)
        else:
            self.regressor = linear_model.Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver)

        self.regressor.fit(X,y)
        return self


class Lasso():
    """Linear Model trained with L1 prior as regularizer.

    The optimization objective for Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    The Lasso is a linear model that estimates sparse coefficients. It is 
    useful in some contexts due to its tendency to prefer solutions with fewer 
    non-zero coefficients, effectively reducing the number of features upon 
    which the given solution is dependent. 
    """

    def __init__(self,
                 max_iter=1000,
                 tol=1e-4,
                 fit_intercept=True,
                 normalize=True,
                 alpha=0.01,
                 cv=False,
                 cv_num_folds=5,
                 **kwargs):
        # common setting
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        # default setting
        self.alpha = alpha

        # cross-validation setting
        self.cv = cv
        self.cv_num_folds = cv_num_folds
        self.regressor = None

    def fit(self, X, y):
        if self.cv:
            self.regressor = linear_model.LassoCV(
                cv=self.cv_num_folds,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            self.regressor = linear_model.Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        self.regressor.fit(X,y)
        return self


class ElasticNet():
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::
            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    Notes:: l1_ratio parmameters means the coefficient of L1 regularization, 
            the coefficient of L2 regularization is (1-l1_ratio).
    """

    def __init__(self,
                 max_iter=1000,
                 tol=1e-4,
                 fit_intercept=True,
                 normalize=True,
                 alpha=0.01,
                 l1_ratio=0.2,
                 cv=False,
                 cv_num_folds=5,
                 **kwargs):
        # common setting
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept,
        self.normalize = normalize

        # default setting
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        # cross-validation setting
        self.cv = cv
        self.cv_num_folds = cv_num_folds
        self.regressor = None

    def fit(self,X, y):
        if self.cv:
            self.regressor = linear_model.ElasticNetCV(
                l1_ratio=self.l1_ratio,
                cv=self.cv_num_folds,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            self.regressor = linear_model.ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        self.regressor.fit(X,y)
        return self


class expand_linear_model():
    """expand edition of linear model from sklearn library.
    """

    def estimate(self, algorithm=None):
        """
        1. automatic relevance determination, 
        Compared to the OLS (ordinary least squares) estimator, the coefficient 
        weights are slightly shifted toward zeros, which stabilises them. 
        2. stochastic gradient descent method
        NOTE: could used for different penalty
        3. online passive-aggressive
        They are similar to the Perceptron in that they do not require a 
        learning rate. However, contrary to the Perceptron, they include a 
        regularization parameter C.
        """
        if algorithm == 1:
            reg = linear_model.ARDRegression()
        elif algorithm == 2:
            reg = linear_model.SGDRegressor()
        elif algorithm == 3:
            reg = linear_model.PassiveAggressiveRegressor()
        return reg

    def robust(self, algorithm=None):
        """
        Robust regression aims to fit a regression model in the presence of 
        corrupt data: either outliers, or error in the model.
        1. RANSAC
        2. Theil-Sen
        3. Huber
        """
        if algorithm == 1:
            reg = linear_model.RANSACRegressor()
        elif algorithm == 2:
            reg = linear_model.TheilSenRegressor()
        elif algorithm == 3:
            reg = linear_model.HuberRegressor()
        return reg



if __name__ == "__main__":

    pass
