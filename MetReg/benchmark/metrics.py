# -----------------------------------------------------------------------------
#                                Evaluate module                              # 
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo is for model performance based on all ILAMB metrics (https://doi. #
# org/10.1029/2018MS001354), including sklearn based metrics, unbiased metric #
# -----------------------------------------------------------------------------


import numpy as np
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_gamma_deviance,
                             mean_poisson_deviance, mean_squared_error,
                             mean_squared_log_error, mean_tweedie_deviance,
                             median_absolute_error, r2_score)

from sklearn.preprocessing import MinMaxScaler

def mae(y_true, y_pred):
    output_errors = np.average(np.abs(y_pred - y_true), axis=0)
    return np.average(output_errors)

def mse(y_true, y_pred):
    output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    return np.average(output_errors)

def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (NSE)."""
    nse_ = 1-(
        np.sum((y_true-y_pred)**2, axis=0)/np.sum((y_true-np.mean(y_true))**2))
    return nse_

def rmse(y_true, y_pred):
    rmse_ = np.sqrt(np.mean((y_true-y_pred)**2, axis=0))
    return rmse_


def inverse(pred, true):
    """Inverse prediction array with true array"""
    # scaler
    scaler = MinMaxScaler()
    # inverse
    for i in range(true.shape[1]):
        for j in range(true.shape[2]):
            if (np.isnan(true[:, i, j, :]).any()) or (np.isnan(pred[:, i, j, :]).any()):
                print('This pixel ')
            else:
                scaler.fit_transform(true[:, i, j, :])
                pred[:, i, j, :] = scaler.inverse_transform(pred[:, i, j, :])
    return pred






class Metrics():

    def __init__(self, validate, forecast, metrics:list=None):

        if metrics is None:
            self.metrics = True
        else:
            self.metrics = metrics

        self._get_data_attr(validate)

    def _get_data_attr(self, validate):
        self.T, self.H, self.W = validate.shape

    def get_sklearn_metrics(self, validate,forecast):

        """get regression metrics from sklearn API.

        Args:
            forecast (np.array):
                shape of (timestep, height, width)
            validate (np.array):
                shape of (timestep, height, width) 

        Returns:
            metrics (dict):
        """
        metrics = {}
        
        return metrics
    
    @staticmethod
    def _print_avg_metrics(metrics):

        for metric_name, metric in metrics.items():
            print('{} is {}'.format(metric_name, np.nanmean(metric)))


    @staticmethod
    def get_unbiased_metrics(forcast, validate):
        pass
    

    def get_criterion_metrics(self):
        """aic, aicc, bic, mallows cp
        """
        pass

    def get_importance_metrics(self, reg,x_train):
        """
        """
        # generate importance array
        try:
            importance = reg.coef_ * x_train.std(axis=0)
        except AttributeError:  # for tree regression
            try:  # only for RANSAC, cuz different code of using coef_
                importance = reg.estimator_.coef_ * x_train.std(axis=0)
            except AttributeError:
                try:
                    importance = reg.feature_importances_
                except:
                    importance = False
                    print('This is not a linear regression or tree regression')

        return importance
