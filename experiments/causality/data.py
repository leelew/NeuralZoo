#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:32:50 2019

@author: lewlee
"""
import numpy as np
import glob
import netCDF4 as nc
import pandas as pd
import h5py
#from CausalityTest import timer
#import matlab.engine as me
#import shapely.geometry as geometry
#from collections import defaultdict
#import shapefile


class Data():

    def __init__(self, file_path, folder_path):

        self.file_path = file_path
        self.folder_path = folder_path

    def _read_lat_lon_time(self, file_path):
        """read latitude, longitude, time dimimsizes

        single_path: single nc file path

        """

        nc_obj = nc.Dataset(file_path)

        # latitude, longitude for ERA5
        #_lat = nc_obj.variables['latitude'][:]
        #_lon = nc_obj.variables['longitude'][:]
        #_time = nc_obj.variables['time'][:]

        # lat, lon for GLDAS
        _lat = nc_obj.variables['lat][:]
        _lon = nc_obj.variables['lon'][:]
        _time = nc_obj.variables['time'][:]

        return _lat, _lon, _time

    def _read_miss(self, file_path):
        """read missing value of target nc file"""

        # class
        nc_obj = nc.Dataset(file_path)
        # get keys
        _keys = nc_obj.variables.keys()
        # get last key of list
        _key_index = list(_keys)[-1:]
        # get missing value of each variable
        return nc_obj.variables[_key_index[0]].missing_value

    def _read_list_path(self, folder_path):
        """read mutiply nc files paths in a specific folder
        """

        # return list of paths in folder.
        return glob.glob(folder_path, recursive=True)

    def _read_nc_single(self, file_path, param):
        """return matrix of data shape as (time,lat,lon) in nc file"""

        # class dataset
        nc_obj = nc.Dataset(file_path)
        # get keys
            #_keys = nc_obj.variables.keys()
        # get last key of list
            #_key_index = list(_keys)[-1:]
        # get matrix of last key
        return nc_obj.variables[param][:]
               #nc_obj.variables[_key_index[0]][:], _key_index[0]

    def _miss2nan(self):
        """***changed missing value in data to np.nan"""
        pass
        #        	for key in data.keys():
        #        		data.replace(df_missing_value[key],np.nan)
        #
        #        	return data

    def _read_single(self, file_path, lat_index, lon_index, param):

        # read (time,lat,lon) data. shape as np.array
        data, string = self._read_nc_single(file_path, param)
        # select lat/lon data
        return pd.Series(data[:, lat_index, lon_index]), string

    def _read(self, folder_path, lat_index, lon_index):

        # initialize a dataframe
        df = pd.DataFrame()
        # get list of folder
        _list_folder = self._read_list_path(folder_path)
        # loop path of folder
        for path in _list_folder:
            # get data and string for target lat,lon index
            data, string = self._read_single(path, lat_index, lon_index)
            # pass string and corresponding data in dataframe
            df[string] = data

        return df

    def _shp_polygon(self, file_path, shp_path):
        """
        TODO: python function need improved.
        to get polygon of shapefile, read index_china.m file.
        """

#        # start MATLAB
#        MATLAB = me.start_matlab()
#        # get lat,lon of nc file
#        lat,lon,time = self._read_lat_lon_time(file_path)
#        # get index of lat,lon locate in shapefile
#        index = MATLAB.read_lat_lon("bou2_4p.shp",lat,lon)
#        sio.savemat('latlon.mat', {'Flat': flat_lat,
#                       'Flon': flat_lon,
#                       'Ftime': time,
#                       'path': "bou2_4p.shp"})

#        _border = shapefile.Reader("bou2_4p.shp",encoding='GB2312')
#        border = _border.shapes()
#        recd = _border.records()
#        grid_lon, grid_lat = np.meshgrid(lon,lat)
#        flat_lon = grid_lon.flatten()
#        flat_lat = grid_lat.flatten()
#        flat_points = np.column_stack((flat_lon,flat_lat))
#        in_shape_points = []
#        for pt in flat_points:
#            if geometry.Point(pt).within(geometry.shape(border)):
#                in_shape_points.append(pt)

#        return index

    def _avg_shp_single(self, file_path, index_path):
        """
        index_path: path of index .npy file created by MATLAB
        """
        index_ = np.load(index_path)
        # read (time,lat,lon) data. shape as np.array
        data, string = self._read_nc_single(file_path)
        i, j = np.nonzero(index_)
        # print(np.shape(data[:,i,j]))
        data_ = data[:, i, j].mean(-1)
        # print(np.shape(data_))

        return pd.Series(data_), string

#    @timer
    def _avg_shp(self, folder_path, index_path):

        # initialize a dataframe
        df = pd.DataFrame()
        # get list of folder
        _list_folder = self._read_list_path(folder_path)
        # loop path of folder
        for path in _list_folder:
            # get data and string for target lat,lon index
            data, string = self._avg_shp_single(path, index_path)
            # pass string and corresponding data in dataframe
            df[string] = data

        return df

    def _mat2npy(self, mat_path, npy_path):

        data = h5py.File(mat_path)
        data_ = data['index_']
        data = np.transpose(data_)
        np.save(npy_path, data)


# class DataDaily(Data):


# class DataMonthly(Data):

if __name__ == "__main__":
    DA = Data(file_path="GLDAS_CLSM025_D.A20101028.020.nc4",
              folder_path="GLDAS/")
    _lat, _lon, _time = DA._read_lat_lon_time()
    _path = DA._read_list_path()
    DA._read_nc_single(file_path=)

