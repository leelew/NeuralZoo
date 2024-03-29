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
        return np.nanstd(X)

    def _cal_normalized_std(self, y_true, y_pred):
        return self._cal_std(y_pred)/self._cal_std(y_true)

    def _cal_scorr(self, y_true, y_pred):
        """Calculate spatial correlation.

        Args:
            y_true, y_pred (nd.array): shape of (lat,lon)
        """
        y_true = y_true.reshape(-1,)
        y_pred = y_pred.reshape(-1,)
        y_true = y_true[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        return np.corrcoef(y_true.reshape(-1,), y_pred.reshape(-1,))[0, 1]

    def cal_dist(self, y_true, y_pred):
        v_ref = np.nanmean(y_true, axis=0)
        v_mod = np.nanmean(y_pred, axis=0)
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

    def _cal_wi(self, y_true, y_pred):
        mean = np.mean(y_true)
        wi_ = 1 - np.sum((y_true-y_pred)**2, axis=0) / \
            np.sum(abs(y_pred-mean)+abs(y_true-mean)**2)
        return wi_

    def _cal_r(self, y_true, y_pred):
        m1, m2 = np.mean(y_true), np.mean(y_pred)
        t1 = np.sqrt(np.sum((y_true-m1)**2))
        r = np.sum((np.abs(y_true - m1) * np.abs(y_pred - m2))) / \
            (np.sqrt(np.sum((y_true - m1) ** 2))
             * np.sqrt(np.sum((y_pred - m2) ** 2)))
        return r

    def _cal_kge(self, y_true, y_pred):
        r = self._cal_r(y_true, y_pred)
        beta = np.mean(y_pred)/np.mean(y_true)
        gamma = (np.std(y_pred)/np.mean(y_pred)) / \
            (np.std(y_true)/np.mean(y_true))
        return 1-np.sqrt((r-1)**2+(beta-1)**2+(gamma-1)**2)

    def _cal_mean(self, y_true, y_pred):
        m1, m2 = np.mean(y_true), np.mean(y_pred)
        return m1, m2

    def _cal_std(self, y_true, y_pred):
        std1, std2 = np.nanstd(y_true), np.nanstd(y_pred)
        return std1, std2

    def _cal_score(self, y_true, y_pred):
        nse = self._cal_nse(y_true, y_pred)
        wi = self._cal_wi(y_true, y_pred)
        r = self._cal_r(y_true, y_pred)
        rmse = RMSE()._cal_score_rmse(y_true, y_pred)
        kge = self._cal_kge(y_true, y_pred)

        return (r+wi+kge+rmse+2*nse)/(1+1+1+1+2)

    def cal(self, y_true, y_pred):

        bias = self._cal_bias(y_true, y_pred)
        rmse = self._cal_rmse(y_true, y_pred)
        nse = self._cal_nse(y_true, y_pred)
        r2 = self._cal_r2(y_true, y_pred)
        wi = self._cal_wi(y_true, y_pred)
        m1, m2 = self._cal_mean(y_true, y_pred)
        kge = self._cal_kge(y_true, y_pred)
        r = self._cal_r(y_true, y_pred)
        mae = self._cal_mae(y_true, y_pred)
        mse = self._cal_mse(y_true, y_pred)
        score = self._cal_score(y_true, y_pred)
        std1, std2 = self._cal_std(y_true, y_pred)

        return [bias, rmse, nse, r2, wi, kge, r, m1, m2, mae, mse, score, std1, std2]


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



def SWDI(historical, inputs, predict=None):
    """Soil moisture deficit index(SMDI) from [Narasimhan and Srinivasan et al. 2005].
        
    Reference:
    [Narasimhan and Srinivasan et al. 2005]: Development and evaluation of 
                                             Soil Moisture Deficit Index(SMDI) and Evapotranspiration Deficit Index (ETDI) for agricultural drought monitoring.
    
    [Yidraw et al. 2008]: GRACE satellite observations of terrestrial moisture 
                          changes for drought characterization in the Canadian Prairie.
    """
    if predict is not None:
        s, t, h, w, _ = predict.shape

        swdi_regions = np.full((int(s/7), h, w), np.nan)
        for i in range(h):
            for j in range(w):
                print(i,j)
                # 
                percentile95 = np.nanpercentile(historical[:, 0, i, j], 95)
                percentile05 = np.nanpercentile(historical[:, 0, i, j], 5)

                # 
                swdi_year = []
                
                for week in range(int(s/7)):
                    # see eq.8 in Narasimhan and Srinivasan et al. 2005
                    sw = np.nanmean(predict[:,:,:,:,0], axis=1)
                    sw = sw[week*7, i, j]
                    
                    swdi = (sw-percentile95)/(percentile95-percentile05)*10
                    swdi_year.append(swdi)

                swdi_regions[:, i, j] = np.array(swdi_year)

    else:
        t, _, h, w = inputs.shape

        swdi_regions = np.full((int(t/7), h, w), np.nan)
        for i in range(h):
            for j in range(w):
                print(i,j)
                # 
                percentile95 = np.nanpercentile(historical[:, 0, i, j], 95)
                percentile05 = np.nanpercentile(historical[:, 0, i, j], 5)

                # 
                swdi_year = []
                for week in range(int(t/7)):
                    # see eq.8 in Narasimhan and Srinivasan et al. 2005
                    sw = np.nanmean(inputs[week*7:week*7+7, 0, i, j])
                    
                    swdi = (sw-percentile95)/(percentile95-percentile05)*10
                    swdi_year.append(swdi)

                swdi_regions[:, i, j] = np.array(swdi_year)
    
    return swdi_regions



def SMDI(historical, inputs, predict=None):
    """Soil moisture deficit index(SMDI) from [Narasimhan and Srinivasan et al. 2005]."""
    t, _, h, w = inputs.shape

    smdi_regions = np.full((int(t/7), h, w), np.nan)
    for i in range(h):
        for j in range(w):
            print(i,j)
            # generate median, max, min for each grid
            max = np.nanmax(historical[:, 0, i, j])
            min = np.nanmin(historical[:, 0, i, j])
            median = np.nanmedian(historical[:, 0, i, j])

            # 
            smdi_year = []
            for week in range(int(t/7)):
                # see eq.8 in Narasimhan and Srinivasan et al. 2005
                sw = np.nanmean(inputs[week*7:week*7+7, 0, i, j])
                
                if sw <= median: 
                    sd = 100*(sw - median)/(median - min)
                elif sw > median:
                    sd = 100*(sw - median)/(max - median)

                if week == 0:
                    smdi = sd/50
                else:
                    smdi += sd/50
                print(smdi)
                smdi_year.append(smdi)

            smdi_regions[:, i, j] = np.array(smdi_year)

    return smdi_regions