from numpy.core.fromnumeric import var
from data.ops.time import TimeManager
import netCDF4 as nc
import numpy as np
import os

#TODO:sadd to RawERA5Preprocesser

def read_era5(input_dir, save_dir, var_list, begin_date, end_date, lat_lower,
              lat_upper, lon_left, lon_right, var_name, smap_lat, smap_lon):

    # get dates array according to begin/end dates
    dates = TimeManager().get_date_array(begin_date, end_date)

    # init
    init_file = input_dir + '/1999/land.19990101.nc'
    f = nc.Dataset(init_file, 'r')
    lat, lon = f['latitude'][:], f['longitude'][:]
    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    # read
    data = np.full((len(var_list), len(lat_idx), len(lon_idx)), np.nan)

    for i, date in enumerate(dates):
        print('preprocessing {} {}'.format(date, var_name))
        # get path
        path = input_dir + '/' + str(date.year) + \
               '/land.{year}{month:02}{day:02}.nc'.format(
               year=date.year, month=date.month, day=date.day)

        # read variables
        f = nc.Dataset(path, 'r')

        for k, var in enumerate(var_list):
            tmp = f[var][:, lat_idx, :][:, :, lon_idx]
            tmp[tmp == -9999] = np.nan
            data[k] = np.nanmean(tmp, axis=0)
        #data1 = era2smap(data, lat[lat_idx], lon[lon_idx], smap_lat, smap_lon)
        #print(data1.shape)
        # save to nc files
        filename = 'ERA5_land_{var_name}_{year}{month:02}{day:02}.nc'.format(
            var_name=var_name, year=date.year, month=date.month, day=date.day)

        # judge already exist file
        if os.path.exists(save_dir + filename):
            pass
        else:
            f = nc.Dataset(save_dir + filename, 'w', format='NETCDF4')

            f.createDimension('longitude', size=len(lon_idx))#smap_lon.shape[0])
            f.createDimension('latitude', size=len(lat_idx))#smap_lat.shape[0])
            f.createDimension('feature', size=data.shape[-3])

            longitude = f.createVariable('longitude',
                                         'f4',
                                         dimensions='longitude')
            latitude = f.createVariable('latitude',
                                        'f4',
                                        dimensions='latitude')
            ssm = f.createVariable(var_name,
                                   'f4',
                                   dimensions=('feature', 'latitude',
                                               'longitude'))
            #FIXME: Change stupid transform code
            longitude[:], latitude[:] = lon[lon_idx], lat[lat_idx]#smap_lon, smap_lat
            ssm[:] = data

            f.close()

    return data


def era2smap(era5, era_lat, era_lon, smap_lat, smap_lon):
    # era5 (t, 1, 300, 600)
    data = np.full((era5.shape[0], len(smap_lat), len(smap_lon)), np.nan)
    for i, lat in enumerate(smap_lat):
        for j, lon in enumerate(smap_lon):
            lat_idx = np.where((era_lat >= lat - 0.1)
                               & (era_lat <= lat + 0.1))[0]
            lon_idx = np.where((era_lon >= lon - 0.1)
                               & (era_lon <= lon + 0.1))[0]
            tmp = era5[:, lat_idx, :][:, :, lon_idx]
            data[:, i, j] = np.nanmean(tmp, axis=(-1,-2))

    return data


if __name__ == '__main__':

    f = nc.Dataset('/hard/lilu/SMAP_L4/test/SSM/train/SMAP_L4_SSM_20200701.nc',
                   'r')
    smap_lon, smap_lat = f['longitude'], f['latitude']

    read_era5(input_dir='/hard/lilu/ERA52',
              save_dir='/hard/lilu/SMAP_L4/era5/SM/',
              var_list=['swvl1'],
              begin_date='1999-01-01',
              end_date='2009-12-31',
              lat_lower=14.7,
              lat_upper=53.5,
              lon_left=72.3,
              lon_right=135,
              var_name='SM',
              smap_lat=smap_lat,
              smap_lon=smap_lon)

    read_era5(input_dir='/hard/lilu/ERA52',
              save_dir='/hard/lilu/SMAP_L4/era5/forcing/',
              var_list=['tp', 'strd', 'ssrd', 't2m', 'sp', 'e', 'u10'],
              begin_date='1999-01-01',
              end_date='2009-12-31',
              lat_lower=14.7,
              lat_upper=53.5,
              lon_left=72.3,
              lon_right=135,
              var_name='forcing',
              smap_lat=smap_lat,
              smap_lon=smap_lon)
