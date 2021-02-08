# -----------------------------------------------------------------------------
#                                Evaluate module                              #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo is for model performance based on all ILAMB metrics (https://doi. #
# org/10.1029/2018MS001354), including sklearn based metrics, unbiased metric #
# -----------------------------------------------------------------------------

import math

import numpy as np
from MetReg.base.base_score import BaseScore
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_gamma_deviance,
                             mean_poisson_deviance, mean_squared_error,
                             mean_squared_log_error, mean_tweedie_deviance,
                             median_absolute_error, r2_score)
from sklearn.preprocessing import MinMaxScaler


class Bias(BaseScore):

    def __init__(self):
        super().__init__()

    def _cal_bias(self, y_true, y_pred):
        """Bias of mean temporal state."""
        return np.mean(y_pred) - np.mean(y_true)

    def _cal_crms(self, X):
        """Centralized RMS."""
        return np.sqrt(np.sum((X-np.mean(X))**2, axis=0))

    def _cal_score_bias(self, y_true, y_pred):
        """bias score on 1 grid."""
        relative_bias = np.abs(self._cal_bias(
            y_true, y_pred))/self._cal_crms(y_true)
        score_bias_ = math.exp(-relative_bias)
        return score_bias_

    def cal_bias(self, y_true, y_pred):
        """Mean values over space of bias score.

        Notes:: Only calculate bias on given spatial location leads to
                consequence that in area where given variable has a small
                magnitude, simple noise can lead to large relative errors.

                ILAMB give a concept, mass weighting, i.e., when performing
                the spatial integral to obtain a scaler score, we weight the
                integral by the period mean value of the true value (also could
                be reference value for lat-lon grids).
        """
        return self.score(func=self._cal_score_bias,
                          y_true=y_true,
                          y_pred=y_pred,
                          mass_weight=True)

    def __repr__(self):
        return {
            'short name': 'bias',
            'long name': 'bias',
        }


class RMSE(Bias):

    def __init__(self):
        super().__init__()

    def _cal_rmse(self, y_true, y_pred):
        """RMSE."""
        return np.sqrt(np.mean((y_true-y_pred)**2, axis=0))

    def _cal_crmse(self, y_true, y_pred):
        """Centralized RMSE."""
        return np.sqrt(np.mean(((y_pred-np.mean(y_pred))-(y_true-np.mean(y_true)))**2))

    def _cal_score_rmse(self, y_true, y_pred):
        relative_rmse = self._cal_crmse(y_true, y_pred)/self._cal_crms(y_true)
        score_rmse_ = math.exp(-relative_rmse)
        return score_rmse_

    def cal_rmse(self, y_true, y_pred):
        """Mean value over space of RMSE score.

        Notes:: score the centralized RMSE to decouple the bias score
                from the RMSE score, allowing the RMSE score to focus
                on an orthogonal aspect of model performance.
        """
        return self.score(func=self._cal_score_rmse,
                          y_true=y_true,
                          y_pred=y_pred,
                          mass_weight=True)

    def __repr__(self):
        return {
            'short name': 'rmse',
            'long name': 'relatively mean squared error',
        }


class InterannualVariablity(Bias):

    def __init__(self):
        super().__init__()

    def _cal_annual_cycle(self, X):
        return 1  # Todo: cal ac

    def _cal_score_iv(self, y_true, y_pred):
        relative_iv = (self._cal_annual_cycle(
            y_pred) - self._cal_annual_cycle(y_true)) / self._cal_annual_cycle(y_true)
        score_iv_ = math.exp(-relative_iv)
        return score_iv_

    def cal_iv(self, y_true, y_pred):
        return self.score(func=self._cal_score_iv,
                          y_true=y_true,
                          y_pred=y_pred,
                          mass_weight=True)

    def __repr__(self):
        return {
            'short name': 'iv',
            'long name': 'interannual variability',
        }


class SpatialDist(Bias):

    def __init__(self):
        super().__init__()

    def _cal_score_dist(self, y_true, y_pred):
        2*(1+self._cal_spatial_correlation())


class PhaseShift(Bias):

    def __init__(self):
        pass


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


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

    def __init__(self, validate, forecast, metrics: list = None):

        if metrics is None:
            self.metrics = True
        else:
            self.metrics = metrics

        self._get_data_attr(validate)

    def _get_data_attr(self, validate):
        self.T, self.H, self.W = validate.shape

    def get_sklearn_metrics(self, validate, forecast):
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

    def get_importance_metrics(self, reg, x_train):
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
