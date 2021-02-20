#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:19:44 2019

@author: lewlee
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime as dt
import pandas as pd
import statsmodels.tsa.stattools as ts


class Detrend():

    """
    Only for daily data.
    1. remove trend and periodicty of time series.

    methods
    [1]: two-step method
        (1) First, using timestep to fit time seires,like y = at+b, 
            get the detrend time series, i.e., b.
        (2) Second, get climatology average of each feature, and remove it
            from detrend time series, i.e., b = x + db.
    [2]: fourier method
        (1) First, construct cos and sin terms of periodicity(e.g., annual,
        seasonal) to represent periodicity of variables. record as X
        (2) Second, X fit Y using linear models(e.g., GLM) and nonlinear models
        (e.g., RF) to remove linearity and nonlinearity of periodictiy

    2. stability test using Augmented Dickey–Fuller test
    ***TODO: dataframe interface need to be constructed

    paramters
    _________

    target_data: array-like, (n_length, n_features)

    begin_date: begin date of target_data, ***TODO: auto define method need 
                type as "YYYY-MM-DD"

    end_date: end date of target_data

    attributes
    __________

    _detrend_seasonal_t: time series remove seasonality

    _detrend_t: time series remove trending

    _adf_value: result of ADF test     

    """
    
    def __init__(self,
                 target_data,
                 begin_date='2019-05-30',
                 end_date='2019-06-04'):

        self.target_data = target_data
        self.begin_date = begin_date
        self.end_date = end_date  
    
    def _detrend(self,target_data, 
                      begin_date, 
                      end_date):
        
        """
        only for daily data. 
        1. default method. [1]
        2. fourier method used in [2]

        [1] Papagiannopoulou, C., Gonzalez Miralles, D., Decubber, S.,
            Demuzere, M., Verhoest, N., Dorigo, W. A., & Waegeman, W. (2017). 
            A non-linear Granger-causality framework to investigate 
            climate-vegetation dynamics. 
            Geoscientific Model Development, 10(5), 1945-1960.
            
        [2] Tuttle, S., & Salvucci, G. (2016). 
            Empirical evidence of contrasting soil moisture–precipitation 
            feedbacks across the United States. 
            Science, 352(6287), 825-828.
        """
        # [1] 
        
        # detrend timesteps
        _detrend_t, _trend_t = self._detrend_timestep(target_data)
        # remove seasonality
        _detrend_seasonal_t, _seasonal_avg = self._detrend_seasonality \
                                             (_detrend_t,begin_date,end_date)
                                             
        # plot module
#        plt.figure()
#        plt.subplot(2,2,1)
#        plt.plot(target_data[:,1],label='raw data')
#        plt.plot(_trend_t[:,1],label='trend_t')        
#        plt.legend(loc='best')
#        
#        plt.subplot(2,2,2)
#        plt.plot(_detrend_t[:,1],label='detrend_t')
#        plt.legend(loc='best')
#
#        plt.subplot(2,2,3)
#        plt.plot(_seasonal_avg[:,1],label='seasonality')
#        plt.legend(loc='best')
#
#        plt.subplot(2,2,4)
#        plt.plot(_detrend_seasonal_t[:,1],label='de_seasonality')
#        plt.legend(loc='best')

        
        # adf test
        _adf_value = self._adf_test(_detrend_seasonal_t)
        # if pass adf test after remove trend and seasonality, return it
        # if not, neet to be different
        if sum(np.array(_adf_value)-1) == 0:
            print()#return _detrend_seasonal_t
        else:
            print("Time series aren't stable! \nNeed to be differented")
        return _detrend_seasonal_t

        """
        TODO: [2] need to be developed
        """
        
    def _detrend_timestep(self, 
                          target_data,
                          default=None):
        
        """
        remove the trend from the time series. 
        default uses linear regression.
        """
        
        if default is None:               
            # set timestamp t used as a predictor variables.
            _t = np.array(range(self.T)).reshape(-1,1)
            # fit raw time series using t and remove trending 
            _detrend_t = target_data - \
                LinearRegression().fit(_t, target_data).predict(_t)
            _trend_t = LinearRegression().fit(_t, target_data).predict(_t)
            # plt.plot(_detrend_t[:,0])
            return _detrend_t, _trend_t
        else:
            """
            TODO: nonlinear method need improved
            """
            print('Sorry, lilu love panjinjing ')

    def _detrend_seasonality(self, 
                             target_data,
                             begin_date,
                             end_date,                            
                             type_reg=None, 
                             year_length=366):
        
        """
        remove the seasonality from the time series.
        
        parameters:
        __________
        
        target_data: matrix of time series which need to remove seasonality
                     as T x N matrix
                     
        begin_date: begin date of time series. as '2019-05-30'
            
        end_date: end date of time series. as '2019-05-30'
        
        type_reg: regression type. could be linear and nonlinear
                  default set as linear method shown in _detrend[1] above
                  TODO: nonlinear method need improved
                  
        year_length: length of year used to get average seasonality.
                     default consider leap year set as 366.
                     
        Attributes:
        ___________
        
        _detrend_seasonal: matrix of time series after remove seasonality
                           shape is the same as target_data. as T x N matrix
                           
        _seasonal_avg: matrix of seasonality. 
                       shape is year_length x N. 
            
        """
        
        # create DatetimeIndex array as '2019-05-30'...'2019-10-04'
        dates = self._date_array(begin_date, end_date)
        # caculate corresponding day in each year of the time series
        jd = self._date_jd(dates)
        # shape of target data
        _T,_N = np.shape(target_data)
        # Initialize the array for contain de-periodicty time series
        _detrend_seasonal = np.zeros((_T, _N))
        _seasonal_avg = np.zeros((year_length,_N))
        # main loop for 1,2,3,...,366
        for j in range(1,year_length+1):
            # set list of index of 1,2,3,...,366 day over dataframe
            _jd_list = [i for i, value in enumerate(jd) if value == j]
            # get the average value on 1,2,3...,366 day
            _detrend_seasonal[_jd_list,:] = target_data[_jd_list,:] - \
                                   np.mean(target_data[_jd_list,:],0)
            # seasonal average time series for plot
            _seasonal_avg[j-1,:] = np.mean(target_data[_jd_list,:],0)

        return _detrend_seasonal, _seasonal_avg                        

    def _date_array(self, begin_date, end_date):
        
        """
        create DatatimeIndex array as '2019-05-30','2019-05-31',...
        """
        
        # Initialize the list from begin_date to end_date
        dates = []
        # Initialize the timeindex for append in dates array.
        _dates = dt.datetime.strptime(begin_date, "%Y-%m-%d")
        # initialized the timeindex for decide whether break loop
        _date = begin_date[:]
        # main loop
        while _date <= end_date:
            # pass date in the array
            dates.append(_dates)
            # refresh date by step 1
            _dates = _dates + dt.timedelta(1)
            # changed condition by step 1
            _date = _dates.strftime("%Y-%m-%d")
        return dates
    
    def _date_jd(self, dates):
        
        """
        create list for corresponding day in each year of the time series
        """
        
        # changed DatatimeIndex array in time format
        dates = pd.to_datetime(dates, format='%Y%m%d')  
        # create DatatimeIndex array of 1st day of each year
        new_year_day = pd.to_datetime([pd.Timestamp(year=i, month=1, day=1) 
                                        for i in dates.year])
        # caculate corresponding day in each year of the time series
        jd = list((dates-new_year_day).days + 1)
        return jd   
                
    def _adf_test(self, 
                  data, 
                  significant_value=0.05):
        
        """
        Augmented Dickey–Fuller test
        """
        _T,_N = np.shape(data)
        _adf_value = np.zeros((_N,1))
        # significant margin value 5%
        if significant_value == 0.05:
            for i in range(_N):
                # AD-fuller test, return tuple as
                # (t value
                #  p value
                #  lagged order
                #  freedom degree
                #  1% 5%,10% significant value)
                _adf_result = ts.adfuller(data[:,i]) 
                if _adf_result[0] < _adf_result[4]['5%']:
                    _adf_value[i] = 1
                else:
                    _adf_value[i] = 0
            return _adf_value
        else:
            print('Sorry, lilu love lifuguang')
