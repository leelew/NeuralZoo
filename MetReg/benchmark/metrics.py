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
            y_pred) - self._cal_annual_cycle(y_true)) / \
            self._cal_annual_cycle(y_true)
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

    def _cal_std(self, X):
        return np.std(X)

    def _cal_normalized_std(self, y_true, y_pred):
        return self._cal_std(y_pred)/self._cal_std(y_true)

    def _cal_scorr(self, y_true, y_pred):
        """Calculate spatial correlation.

        Args:
            y_true, y_pred (nd.array): shape of (lat,lon)
        """
        return np.corrcoef(y_true.reshape(-1,), y_pred.reshape(-1,))[0, 1]

    def cal_dist(self, y_true, y_pred):
        v_ref = np.mean(y_true, axis=0)
        v_mod = np.mean(y_pred, axis=0)
        std_ = self._cal_normalized_std(v_ref, v_mod)
        scorr_ = self._cal_scorr(v_ref, v_mod)
        score_dist = 2 * (1 + scorr_) / ((std_ + 1 / std_) ** 2)
        return score_dist


class PhaseShift(Bias):

    def __init__(self):
        pass


class DefaultRegressionScore:
    """All regression scores includes bias, rmse, r2, mae, mse, nse. 

    Notes:: Now only support bias & rmse for ilamb edition, which represents
            bias and variance. r2 is a non-dimensionalized metrics which 
            already represent score. mse is nearly the same with rmse. nse
            need further thinking.
    """

    def __init__(self):
        pass

    def _cal_r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def _cal_mae(self, y_true, y_pred):
        output_errors = np.average(np.abs(y_pred - y_true), axis=0)
        return np.average(output_errors)

    def _cal_mse(self, y_true, y_pred):
        output_errors = np.average((y_true - y_pred) ** 2, axis=0)
        return np.average(output_errors)

    def _cal_nse(self, y_true, y_pred):
        """Nash-Sutcliffe Efficiency (NSE)."""
        nse_ = 1-(
            np.sum((y_true-y_pred)**2, axis=0)/np.sum((y_true-np.mean(y_true))**2))
        return nse_
    
    def _cal_rmse(self, y_true, y_pred):
        return RMSE()._cal_rmse(y_true, y_pred)

    def _cal_bias(self, y_true, y_pred):
        return Bias()._cal_bias(y_true, y_pred)

    def cal(self, y_true, y_pred):
        
        bias = self._cal_bias(y_true, y_pred)
        rmse = self._cal_rmse(y_true, y_pred)
        nse = self._cal_nse(y_true, y_pred)
        r2 = self._cal_r2(y_true, y_pred)

        return [bias, rmse, nse, r2]

class CriterionScore(Bias):
    """support aic, aicc, bic, mallows cp indexs."""
    pass


def TreeImportanceArray(reg):
    """Get important array if `reg` is tree regression."""
    return reg.feature_importances_


class OverallScore():
    """Get overall score for benchmarking using method from ILAMB.
    """

    def __init__(self):
        self.bias = Bias()
        self.rmse = RMSE()
        self.iv = InterannualVariablity()
        self.dist = SpatialDist()

    def score_3d(self, y_true, y_pred):
        bias = self.bias.cal_bias(y_true, y_pred)
        rmse = self.rmse.cal_rmse(y_true, y_pred)
        iv = self.iv.cal_iv(y_true, y_pred)
        dist = self.dist.cal_dist(y_true, y_pred)

        return (bias + 2 * rmse + iv + dist) / (1 + 2 + 1 + 1)

    def score_1d(self, y_true, y_pred):
        bias = self.bias._cal_score_bias(y_true, y_pred)
        rmse = self.rmse._cal_score_rmse(y_true, y_pred)
        iv = self.iv._cal_score_iv(y_true, y_pred)

        return (bias + 2 * rmse + iv) / (1 + 2 + 1 + 1)
