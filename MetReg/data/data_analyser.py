
import numpy as np
from sklearn import linear_model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller


class Data_analyser_ts():
    """analysis data for time series.

    A class contain timeseries analysis functions, including quality, 
    information, deeper information, etc.
    """
    def __init__(self):
        pass

    def calc(self):
        pass

    def _calc_adf(self, timeseries):
        adf = adfuller(timeseries)
        if adf[1] < 0.05: 
            adf_index = 1
        return adf, adf_index

    def _calc_acf(self, timeseries, nlags:int=40):
        pass

    def _calc_avg(self, timeseries,):
        return np.nanmean(timeseries)
        
    def _calc_trend(self, timeseries):
        mdl = linear_model.LinearRegression()
        return mdl.fit(np.arange(len(timeseries)), timeseries).coef_[0]
    
    def _calc_seasonality(self, timeseries):
        pass



class Data_analyser_sts(Data_analyser_ts):

    def __init__(self):
        pass

    def region_calc(self):
        """Perform a calculation for selected regions."""
        pass
