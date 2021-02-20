import datetime as dt
import glob
import math
import os
import re
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import pywt
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def gen_region_index(series, target_range):
    # region index

    return np.argwhere(
        [(series > target_range[0]) & (series < target_range[1])])[:, 1]


def gen_region_data(data, lat_range, lon_range, name):
    data = data[:, lat_range[0]:lat_range[-1], lon_range[0]:lon_range[-1]]
    print(name)
    np.save(os.getcwd() + '/' + name + '_usa.npy', data)


class data_read():

    def __init__(self, path, params=['P_F', 'PA_F'],
                 time_params=['TIMESTAMP']):

        self.path = path
        self.params = params

        self.time_params = time_params

    # NOTE: only for csv file, other files need to give by hand
    def gen_begin_end_date(self):

        self.time = pd.read_csv(self.path, usecols=self.time_params)

        self.begin_date = parse(
            str(self.time.values[0][0])).strftime('%Y-%m-%d')
        self.end_date = parse(
            str(self.time.values[-1][0])).strftime('%Y-%m-%d')

        return self.begin_date, self.end_date

    def parse_csv(self):

        self.data = pd.read_csv(self.path, usecols=self.params)
        self.data = self.data[self.params]
        return self.data  # df type

    def parse_nc(self):

        self.data = []
        # reading variable in each file
        for index in range(len(self.path)):
            print('processing the {} files of {} path'.format(
                index, self.path[index]))
            obj = nc.Dataset(self.path[index])
            self.data.append(obj[self.params[0]][:])

        self.data = np.array(self.data)

        return self.data

    def parse_hdf(self):

        self.data = []
        for index in range(len(self.path)):
            print('processing the {} files of {} path'.format(
                index, self.path[index]))
            f = h5py.File(self.path[index])

            # Geophysical for only SMAP
            self.data.append(f['Geophysical_Data'][self.params[0]])

        self.data = np.array(self.data)

        return self.data

    def parse_npy(self):
        pass


class data_prep(data_read):
    # NOTE: MUST BE PANDAS DATAFRAME TYPE  [1st: OUTPUT, 2nd: QC control, 3rd: timestamp, 4th-end: INPUT]

    def __init__(self, df,
                 missing_value=-9999,
                 fill_missing_data=True, fillna_method='interp',
                 normalization=True, norm_method='minmax',
                 denoise=False, wavelet_threshold=0.04, denoise_method='db8',
                 split_size=0.2, split_method='default',
                 window_range=range(1, 8, 1),
                 begin_date='2019-05-03', end_date='2020-05-03',
                 temporal_resolution='DD',
                 num_annual=6, num_seasonal=5,
                 qc=True, qc_thresold=0.5,
                 input_len=50, output_len=1, window_size=1, batch=False):

        # handle input dataframe
        self.df = df
        self.df_name = df.columns.values
        self.timestep, self.nums_feature = self.df.shape[0], self.df.shape[1] - 2
        self.y = self.df.iloc[:, 0]
        self.x = self.df.iloc[:, 2:]

        self.qc = qc
        self.input_len = input_len
        self.output_len = output_len
        self.window_size = window_size
        self.missing_value = missing_value
        self.qc_thresold = qc_thresold
        self.fill_missing_data = fill_missing_data
        self.fillna_method = fillna_method
        self.denoise = denoise
        self.wavelet_threshold = wavelet_threshold
        self.denoise_method = denoise_method
        self.begin_date = begin_date
        self.end_date = end_date
        self.temporal_resolution = temporal_resolution
        self.split_size = split_size
        self.split_method = split_method
        self.normalization = normalization
        self.norm_method = norm_method

        # generate
        if temporal_resolution == 'DD':
            self.window_range = range(1, 8, 1)  # range(7, 15, 1)
        elif temporal_resolution == 'HH':
            # range(7 * 48, 15 * 48, 48)
            self.window_range = range(1 * 48, 8 * 48, 48)
        elif temporal_resolution == '3HH':
            # range(7 * 8, 15 * 8, 8)
            self.window_range = range(1 * 8, 8 * 8, 8)

        self.num_annual = num_annual
        self.num_seasonal = num_seasonal

        self.gen_2D_df()

        self.batch = batch
        if self.batch:
            self.gen_3D_df()

    def _fix_nan(self):
        # turn default missing value to nan
        self.df[self.df == self.missing_value] = None

        return self.df

    def _qc_control(self):

        # qc control NOTE: only control y (swc in this flx case)
        self.df[self.qc < self.qc_thresold] = None

    def _idx_nonan(self):
        # aim to remove the nan data in the begin and end of series
        self.idx = np.where(~np.isnan(self.df.iloc[:, 0]))[0]
        # rm the continuious missing data at begin/end
        self.df = self.df[self.idx[0]:self.idx[-1]]

    def _fill_nan(self):

        if self.fillna_method == 'interp':
            self.df = self.df.interpolate(method='linear', axis=0)
            # NOTE: in order to avoid the rest nan which can't be interpolate
            self.fillna_method = 'mean'
            self._fill_nan()
        elif self.fillna_method == '0':
            self.df = self.df.fillna(0)
        elif self.fillna_method == 'ffill':
            self.df = self.df.fillna(method='ffill')
        elif self.fillna_method == 'mean':
            for column in list(self.df.columns[self.df.isnull().sum() > 0]):
                mean_val = self.df[column].mean()
                self.df[column].fillna(mean_val, inplace=True)

        # if self.df.isna().any().sum() > 0:
        #    raise Exception('error: still have nan value')

        return self.df

    def _denoise(self):

        # TODO: 2D wave packet method

        # class wavelet
        w = pywt.Wavelet(self.denoise_method)

        maxlev = pywt.dwt_max_level(self.timestep, w.dec_len)
        print("maximum level is " + str(maxlev))

        for j in range(self.df.shape[1]):
            coeffs = pywt.wavedec(
                self.df.iloc[:, j], self.denoise_method, level=maxlev)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(
                    coeffs[i], self.wavelet_threshold * max(coeffs[i]))
            self.df.iloc[:, j] = pywt.waverec(
                coeffs, self.denoise_method).reshape(-1, 1)

        self.timestep, _ = self.df.shape
        # self.df = pd.DataFrame(self.df, columns=self.df_name)

        return self.df

    def _split_train_valid(self):

        if self.split_method == 'default':
            # use train_test_split
            self.df_train, self.df_valid = train_test_split(self.df,
                                                            test_size=self.split_size,
                                                            shuffle=False)
        elif self.split_method == 'year':
            # use the last year
            self.df_train = self.df[:-365*24*2]
            self.df_valid = self.df[-365*24*2:]

        return self.df_train, self.df_valid

    def _normalization(self):

        if self.norm_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.norm_method == 'z-score':
            scaler = StandardScaler()

        self.df_train = np.array(self.df_train)
        self.df_valid = np.array(self.df_valid)

        for i in range(self.df_train.shape[1]):
            self.df_train[:, i] = np.squeeze(scaler.fit_transform(
                self.df_train[:, i].reshape(-1, 1)))
            self.df_valid[:, i] = np.squeeze(scaler.transform(
                self.df_valid[:, i].reshape(-1, 1)))

        return self.df_train, self.df_valid

    # NOTE: only generate lagged time series of y to represent its memory
    def _gen_lagged_series(self):

        columns_name = []

        for i, window_idx in enumerate(self.window_range):
            res_size = self.timestep - window_idx
            self.lagged_series[:, i] = np.concatenate(
                (self.y[res_size:], self.y[:res_size]), axis=0)
            columns_name.append('SWC_' + str(window_idx))

        self.df = pd.concat(
            [self.df, pd.DataFrame(self.lagged_series, columns=columns_name)], axis=1)

        return self.lagged_series

    # NOTE: generate day of year and fourier series
    def _gen_date_array(self):

        # Initialize the list from begin_date to end_date
        dates = []
        # Initialize the timeindex for append in dates array.
        _dates = dt.datetime.strptime(self.begin_date, "%Y-%m-%d")
        # initialized the timeindex for decide whether break loop
        _date = self.begin_date[:]
        # main loop
        while _date <= self.end_date:
            # pass date in the array
            dates.append(_dates)
            # refresh date by step 1
            _dates = _dates + dt.timedelta(1)
            # changed condition by step 1
            _date = _dates.strftime("%Y-%m-%d")

        if self.temporal_resolution == 'DD':
            self.dates = dates
        elif self.temporal_resolution == 'HH':
            self.dates = 48 * dates
            print(len(self.dates))
            for idx, i in enumerate(dates):
                self.dates[idx * 24:idx * 24 + 8] = (i,)
        elif self.temporal_resolution == '3HH':
            self.dates = []
            for idx, i in enumerate(dates):
                for j in range(8):
                    self.dates.append(i)
                #self.dates[idx * 8:idx * 24 + 8] = (i,)
        return self.dates

    def _gen_date_jd(self):

        # changed DatatimeIndex array in time format
        self.dates = pd.to_datetime(self.dates, format='%Y-%m-%d')
        # create DatatimeIndex array of 1st day of each year
        new_year_day = pd.to_datetime([pd.Timestamp(year=i, month=1, day=1)
                                       for i in self.dates.year])
        # caculate corresponding day in each year of the time series
        # NOTE: list_to_numpy cuz only numpy could caculate *2
        self.jd = np.array(list((self.dates - new_year_day).days + 1))

        self.df = pd.concat([self.df,
                             pd.DataFrame(self.jd, columns=['Day_of_Year']),
                             pd.DataFrame(self.dates.year, columns=['Year'])], axis=1)

        return self.jd

    def _gen_fourier_series(self):

        N = np.arange(1, self.timestep + 1)

        columns_name = []

        for i in range(self.num_annual):
            self.sin_annual_series[:, i] = np.sin(
                (2*N*(i + 1)*math.pi) / len(self.jd))
            self.cos_annual_series[:, i] = np.cos(
                (2*N*(i + 1)*math.pi) / len(self.jd))
            columns_name.append('sin_annual_' + str(i))
            columns_name.append('cos_annual_' + str(i))

        for j in range(self.num_seasonal):

            self.sin_seasonal_series[:, j] = np.sin(
                2 * (self.jd) * (j + 1) * math.pi / 365)
            self.cos_seasonal_series[:, j] = np.cos(
                2*(self.jd)*(j + 1)*math.pi / 365)
            columns_name.append('sin_seasonal_' + str(j))
            columns_name.append('cos_seasonal_' + str(j))

        period = np.concatenate([self.sin_annual_series,
                                 self.cos_annual_series,
                                 self.sin_seasonal_series,
                                 self.cos_seasonal_series], axis=1)

        self.df = pd.concat([self.df, pd.DataFrame(
            period, columns=columns_name)], axis=1)

    def gen_2D_df(self):

        self.lagged_series = np.zeros(
            (self.timestep, len(self.window_range)))
        self.sin_annual_series = np.zeros(
            (self.timestep, self.num_annual))
        self.cos_annual_series = np.zeros(
            (self.timestep, self.num_annual))
        self.sin_seasonal_series = np.zeros(
            (self.timestep, self.num_seasonal))
        self.cos_seasonal_series = np.zeros(
            (self.timestep, self.num_seasonal))

        # generate lagged and time series
        self._gen_date_array()
        self._gen_date_jd()
        self._gen_fourier_series()

        self._gen_lagged_series()  # NOTE:only generate lagged y.

        # fix missing value
        self._fix_nan()

        # qc control
        # NOTE: only useful for flx
        if self.qc:
            self.qc = self.df.iloc[:, 1]
            self._qc_control()
        #
        self._idx_nonan()

        # TODO: ensure this site is good
        # nan value less than 0.1 of the site len.

        if self.df.iloc[:, 0].isna().sum() < 0.8 * self.df.shape[0]:

            if self.fillna_method:
                self._fill_nan()

            if self.denoise:
                self._denoise()

            print("OUR INPUT DATAFRAME IS\n")
            print(self.df.head(10))
            print(self.df.columns.values)

            self.timestep = self.df.shape[0]
            # used for generate lagged series, could be other columns
            self.y = self.df.iloc[:, 0]
            self.x = self.df.iloc[:, 2:]

            self._split_train_valid()

            if self.normalization:
                self._normalization()

            self._x_train = self.df_train[:, 2:]
            self._x_valid = self.df_valid[:, 2:]
            self._y_train = self.df_train[:, 0]
            self._y_valid = self.df_valid[:, 0]

        else:
            self._gen_bad()

    def gen_3D_df(self):

        if self._y_train.size != 0:

            _, _x_input_train = self._gen_batch(
                self._x_train[:, :-8])
            _, _x_input_valid = self._gen_batch(
                np.concatenate([self._x_train[-15:, :-8], self._x_valid[:, :-8]], axis=0))

            self._x_train_, self._y_train_ = self._gen_batch(
                self._y_train.reshape(-1, 1))
            self._x_valid_, self._y_valid_ = self._gen_batch(
                np.concatenate([self._y_train[-15:], self._y_valid], axis=0).reshape(-1, 1))

            self._x_train = np.concatenate(
                [self._x_train_, np.squeeze(_x_input_train)[:, :, np.newaxis]], axis=1)
            self._x_valid = np.concatenate(
                [self._x_valid_, np.squeeze(_x_input_valid)[:, :, np.newaxis]], axis=1)

            self._y_train = self._y_train_
            self._y_valid = self._y_valid_

    def _gen_batch(self, data):

        total_start_points = len(data) - self.input_len - \
            self.output_len - self.window_size

        start_idx = range(total_start_points)
        # np.random.choice(
        # range(total_start_points), total_start_points, replace=False)

        batch_size = len(start_idx)

        input_batch_idxs = [(range(i, i + self.input_len)) for i in start_idx]
        output_batch_idxs = [
            (range(i + self.input_len + self.window_size,
                   i+self.input_len+self.output_len+self.window_size)) for i in start_idx]

        input_ = np.take(data, input_batch_idxs, axis=0).reshape(
            batch_size, self.input_len, data.shape[1])

        output_ = np.take(data, output_batch_idxs, axis=0).reshape(
            batch_size, self.output_len, data.shape[1])

        return input_, output_

    def output(self):

        return self._x_train, self._x_valid, self._y_train, self._y_valid

    def _gen_bad(self):

        _x_train, _x_valid, _y_train, _y_valid = np.array(
            []), np.array([]), np.array([]), np.array([])


def gen_flx_x_y(file_path,
                name_cols=['SWC_F_MDS_1', 'SWC_F_MDS_1_QC', 'P_F', 'TA_F', 'SW_IN_F',
                           'LW_IN_F', 'PA_F', 'TS_F_MDS_1'],
                time_resolution='DD',
                begin_date='2019-05-03',
                end_date='2020-05-03',
                batch=True):

    read = data_read(file_path, name_cols)
    df = read.parse_csv()

    begin_date, end_date = read.gen_begin_end_date()

    prep = data_prep(df, begin_date=begin_date, end_date=end_date,
                     temporal_resolution=time_resolution, batch=batch)
    _x_train, _x_valid, _y_train, _y_valid = prep.output()

    return _x_train, _x_valid, _y_train, _y_valid


def gen_esa_x_y(df,
                time_resolution='DD',
                begin_date='2002-06-19',
                end_date='2011-06-19',
                batch=True):

    prep = data_prep(df, begin_date=begin_date, end_date=end_date,
                     temporal_resolution=time_resolution, batch=batch)
    _x_train, _x_valid, _y_train, _y_valid = prep.output()
    return _x_train, _x_valid, _y_train, _y_valid


def gen_smap_x_y(df,
                 time_resolution='3HH',
                 begin_date='2015-03-31',
                 end_date='2019-01-24',
                 batch=True):

    prep = data_prep(df, begin_date=begin_date, end_date=end_date,
                     temporal_resolution=time_resolution, batch=batch)
    _x_train, _x_valid, _y_train, _y_valid = prep.output()
    return _x_train, _x_valid, _y_train, _y_valid



if __name__ == "__main__":

    file_path = "/Users/lewlee/Desktop/Research/test_data/FLX2015/test_DD.csv"
    name_cols = ['SWC_F_MDS_1', 'P_F', 'TA_F', 'SW_IN_F',
                 'LW_IN_F', 'PA_F', 'TS_F_MDS_1']
    R = data_read(file_path, params=name_cols)
    df = R.parse_csv()

    P = data_prep(df, begin_date='1996-01-01', end_date='2014-12-31')
    P.preprocess()
    """
    ################################################################

    # generate GLDAS forcing data
    folder_path = os.getcwd()+"/GLDAS-CLSM/*.nc4"
    path_list = folder_list(folder_path, datasets=3)
    # print(path_list)
    begin = time.time()

    wind = parse_nc(path_list, 'Wind_f_tavg')
    np.save(os.getcwd()+'/wind.npy', wind)
    del wind

    rain = parse_nc(path_list, 'Rainf_f_tavg')
    np.save(os.getcwd()+'/rain.npy', rain)
    del rain

    Tair = parse_nc(path_list, 'Tair_f_tavg')
    np.save(os.getcwd()+'/Tair.npy', Tair)
    del Tair

    Qair = parse_nc(path_list, 'Qair_f_tavg')
    np.save(os.getcwd()+'/Qair.npy', Qair)
    del Qair

    Psurf = parse_nc(path_list, 'Psurf_f_tavg')
    np.save(os.getcwd()+'/Psurf.npy', Psurf)
    del Psurf

    SWdown = parse_nc(path_list, 'SWdown_f_tavg')
    np.save(os.getcwd()+'/SWdown.npy', SWdown)
    del SWdown

    LWdown = parse_nc(path_list, 'LWdown_f_tavg')
    np.save(os.getcwd()+'/LWdown.npy', LWdown)
    del LWdown

    end = time.time()
    print('preprocessing one variables used {} seconds'.format(end - begin))

    ########################################################################

    # generate ESA-CCI soil moisture data
    folder_path = "/hard/lilu/soilHydro/ESA-CCI/*.nc"
    path_list = folder_list(folder_path, datasets=1)

    sm = parse_nc(path_list, 'sm')
    np.save(os.getcwd() + "/SM.npy", sm)
    del sm
    """
    ########################################################################
    """
    # generate forcing and soil moisture data from SMAP
    folders = os.listdir("/hard/lilu/SMAP_L4/SMAP_L4/")
    folders.sort()

    # folders = folder_list("/hard/lilu/SMAP_L4/SMAP_L4/*", datasets=0)
    print(folders)

    # generate us region data of SMAP
    f = h5py.File(
        '/hard/lilu/SMAP_L4/SMAP_L4/2015.03.31/SMAP_L4_SM_gph_20150331T013000_Vv4030_001.h5')
    SMAP_LAT_RANGE = np.array(f['cell_lat'][:, 1])
    SMAP_LON_RANGE = np.array(f['cell_lon'][1, :])

    lat_range = gen_region_index(SMAP_LAT_RANGE, [25, 53])
    lon_range = gen_region_index(SMAP_LON_RANGE, [-125, -67])

    lat_index = np.arange(lat_range[0], lat_range[-1])
    lon_index = np.arange(lon_range[0], lon_range[-1])

    sm = np.zeros((len(folders) - 2, 8, len(lat_index), len(lon_index)))
    p = np.zeros((len(folders) - 2, 8, len(lat_index), len(lon_index)))
    st = np.zeros((len(folders) - 2, 8, len(lat_index), len(lon_index)))

    for i in range(2, len(folders)):
        folder_path = "/hard/lilu/SMAP_L4/SMAP_L4/"+folders[i] + "/*.h5"
        l = folder_list(folder_path, datasets=2)
        print(l[0])
        sm[i - 2, :, :, :] = parse_hdf(l,
                                       'sm_surface')[:, lat_range[0]:lat_range[-1], lon_range[0]:lon_range[-1]]
        p[i-2, :, :, :] = parse_hdf(l, 'precipitation_total_surface_flux')[
            :, lat_range[0]:lat_range[-1], lon_range[0]:lon_range[-1]]
        st[i-2, :, :, :] = parse_hdf(l,
                                     'soil_temp_layer1')[:, lat_range[0]:lat_range[-1], lon_range[0]:lon_range[-1]]

    sm = sm.reshape(-1, len(lat_index), len(lon_index))
    p = p.reshape(-1, len(lat_index), len(lon_index))
    st = st.reshape(-1, len(lat_index), len(lon_index))

    np.save("/hard/lilu/SMAP_L4/results/SMAP_sm_usa.npy", sm)
    np.save("/hard/lilu/SMAP_L4/results/SMAP_p_usa.npy", p)
    np.save("/hard/lilu/SMAP_L4/results/SMAP_st_usa.npy", st)
    """
    ########################################################################
    """
    ########################################################################
    # generate USA region data GLDAS-FORCING
    GLDAS_LAT_RANGE = np.arange(-59.875, 89.875, 0.25)
    GLDAS_LON_RANGE = np.arange(-179.975, 179.875, 0.25)

    lat_range = gen_region_index(GLDAS_LAT_RANGE, [25, 53])
    lon_range = gen_region_index(GLDAS_LON_RANGE, [-125, -67])
    print(np.shape(lat_range))

    data_list = ["rain", "wind", "Psurf",
                 "SWdown", "LWdown", "Qair", "Tair"]

    begin = time.time()

    for data_name in data_list:
        path = os.getcwd()+"/"+data_name+".npy"
        data = np.load(path)
        # print(data.shape)
        print(data_name)
        gen_region_data(data, lat_range, lon_range, data_name)
        del data

    end = time.time()
    print("Construct one pixel input data needs {} seconds".format(end - begin))

    # generate ESA-CCI
    ESACCI_LON_RANGE = np.arange(-179.975, 179.875, 0.25)
    ESACCI_LAT_RANGE = np.arange(89.875, -89.975, -0.25)

    lat_range = gen_index(ESACCI_LAT_RANGE, [25, 53])
    lon_range = gen_index(ESACCI_LON_RANGE, [-125, -67])

    def gen_data_usa(data, lat_range, lon_range, name):
        data = data[:, lat_range[-1]:lat_range[0]
            :-1, lon_range[0]:lon_range[-1]]
        print(name)
        np.save(os.getcwd() + '/' + name + '_usa.npy', data)

    path = os.getcwd()+"/SM.npy"
    data = np.load(path)
    gen_data_usa(data, lat_range, lon_range, "SM")
    del data

    ################################################################################
    # generate annual/seasonal time series
    sin_annual_series, cos_annual_series, \
        sin_seasonal_series, cos_seasonal_series = gen_fourier_series(
            5, 4)
    period_terms = np.concatenate(
        (sin_annual_series, cos_annual_series,
         sin_seasonal_series, cos_seasonal_series), axis=0)

    period_terms = np.transpose(period_terms)

    print(np.shape(period_terms))
    np.save(os.getcwd()+'/period.npy', period_terms)
    """
