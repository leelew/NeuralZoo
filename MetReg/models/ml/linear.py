
from sklearn import linear_model
import numpy as np
import warnings

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
                 config,
                 **kwargs):
        self.fit_intercept = config.fit_intercept
        self.normalize = config.normalize

    def __call__(self):
        mdl = linear_model.LinearRegression(
            fit_intercept=self.fit_intercept,
            normalize=self.normalize)
        return mdl


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
                 config,
                 **kwargs):
        # common setting
        self.max_iter = config.max_iter
        self.tol = config.tol
        self.fit_intercept = config.fit_intercept,
        self.normalize = config.normalize

        # default setting
        self.alpha = config.alpha_ridge
        self.solver = config.solver

        # cross-validation setting
        self.cv = config.cv_ridge
        self.cv_alphas = config.cv_alphas
        self.cv_num_folds = config.cv_num_folds

    def __call__(self):
        if self.cv:
            mdl = linear_model.RidgeCV(
                alphas=self.cv_alphas,
                cv=self.cv_num_folds,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,)
        else:
            mdl = linear_model.Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver)
        return mdl


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
                 config,
                 **kwargs):
        # common setting
        self.max_iter = config.max_iter
        self.tol = config.tol
        self.fit_intercept = config.fit_intercept,
        self.normalize = config.normalize

        # default setting
        self.alpha = config.alpha_lasso

        # cross-validation setting
        self.cv = config.cv_lasso
        self.cv_num_folds = config.cv_num_folds

    def __call__(self):
        if self.cv:
            mdl = linear_model.LassoCV(
                cv=self.cv_num_folds,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            mdl = linear_model.Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        return mdl


class ElasticNet():
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::
            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    """

    def __init__(self,
                 config,
                 **kwargs):
        # common setting
        self.max_iter = config.max_iter
        self.tol = config.tol
        self.fit_intercept = config.fit_intercept,
        self.normalize = config.normalize

        # default setting
        self.alpha = config.alpha_elasticnet
        self.l1_ratio = config.l1_ratio

        # cross-validation setting
        self.cv = config.cv_elasticnet
        self.cv_num_folds = config.cv_num_folds

    def __call__(self):
        if self.cv:
            mdl = linear_model.ElasticNetCV(
                l1_ratio=self.l1_ratio,
                cv=self.cv_num_folds,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            mdl = linear_model.ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        return mdl


if __name__ == "__main__":

    X = np.array([[1, 2], [3, 4], [3, 2], [1, 3], [3, 4]])
    y = np.array([4, 5, 6, 7, 9])

    mdl = get_lr_mdl(mdl_name='LR')
    mdl.fit(X, y)
    print(mdl.coef_)

    mdl = get_lr_mdl(mdl_name='Ridge')
    mdl.fit(X, y)
    print(mdl.coef_)

    mdl = get_lr_mdl(mdl_name='Lasso')
    mdl.fit(X, y)
    print(mdl.coef_)

    mdl = get_lr_mdl(mdl_name='ElasticNet')
    mdl.fit(X, y)
    print(mdl.coef_)
