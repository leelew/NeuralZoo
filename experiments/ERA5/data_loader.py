
import glob
import re
import time

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import sys

sys.path.append('../../')

from MetReg.utils.utils import _get_folder_list, _read_nc




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
