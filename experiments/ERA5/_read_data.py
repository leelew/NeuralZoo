
import glob
import re
import time

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


def _get_folder_list(folder_path, file_type='nc'):
    """Get list of files in target folder.

    Args:
        folder_path (str):
            path of main fold of files
        file_type (str):
            type of files in folder
    """
    # get list
    l = glob.glob(folder_path + 'ERA5*' + file_type, recursive=True)
    print(l)

    # sort list
    num_in_str = []

    # TODO: split name automatically, rather than manually.
    for i, file_path in enumerate(l):
        year = file_path.split('_')[2]
        month = file_path.split('_')[3]
        num_in_str.append(int(year + month))

    # index in order
    indices = np.argsort(np.array(num_in_str))

    # order list
    sorted_l = [l[i] for i in indices]
    return sorted_l


def _read_nc(file_path):
    """load and process single netcdf in ERA5 case.

    Args:
        file_path ([type]): [description]
    """
    obj = netCDF4.Dataset(file_path)

    # Notes:
    #   may be raise keyerror, ensure the key of target variables must
    #   lay the last of variables keys sets.
    targ_var_name = list(obj.variables.keys())[-1]
    targ_var = obj[targ_var_name][:]

    # Notes:
    #   range of fill value must be noticed, which is case-different.
    targ_var[targ_var < 1e-4] = np.nan

    # get shape of raw data
    Ntime, Nlat, Nlon = targ_var.shape

    # init average matrix and average spatiotemporally
    avg_spatial_targ_var = np.full((Ntime, Nlat//4, Nlon//4), np.nan)
    for i in range(Nlat // 4):
        for j in range(Nlon // 4):
            _spatial_targ_var = targ_var[:, 4 * i:4 * i + 4, 4 * j:4 * j + 4]
            avg_spatial_targ_var[:, i, j] = np.nanmean(
                _spatial_targ_var, axis=(-1, -2))

    avg_time_spatial_targ_var = np.full((Ntime//24, Nlat//4, Nlon//4), np.nan)
    for t in range(Ntime // 24):
        _time_targ_var = avg_spatial_targ_var[24 * t:24 * t + 24, :, :]
        avg_time_spatial_targ_var[t, :, :] = np.mean(_time_targ_var, axis=0)

    return avg_time_spatial_targ_var


def main(folder_path='/hard/lilu/2m_temperature',
         output_path='Desktop/',):
    """load and process netcdf in list."""
    sorted_list = _get_folder_list(folder_path=folder_path)
    print(sorted_list)
    # utilize `np.append` func to generate
    targ_var = np.full((1, 180, 360), np.nan)
    for index, path in enumerate(sorted_list):
        print(index)
        _single_targ_var = _read_nc(file_path=path)
        targ_var = np.append(targ_var, _single_targ_var, axis=0)

    targ_var = targ_var[1:, :, :]
    np.save('t2m.npy', targ_var)


if __name__ == "__main__":
    _get_folder_list(folder_path='/hard/lilu/2m_temperature/')
    # main()
