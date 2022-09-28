import sys
sys.path.append('../../')

import numpy as np
from MetReg.utils.utils import _get_folder_list, _read_nc


def main(raw_path, out_path, save_name, NLAT=180, NLON=360):
    """load and process netcdf in list.

    Args:
        raw_path (str):
            absolute path of raw datasets.
        out_path (str):
            absolute path of preliminary datasets.
        NLAT (int):
            number of latitude.
        NLON (int):
            number of longitude.

    Notes:: Utilize `np.append` func to generate cost space memory.
    """
    # get sorted list from raw dataset folder
    sorted_list = _get_folder_list(folder_path=raw_path)

    # init
    targ_var = np.full((1, NLAT, NLON), np.nan)

    # reading
    for index, path in enumerate(sorted_list):
        print('Reading {} file of {} files!'.format(index, len(sorted_list)))
        _single_targ_var = _read_nc(file_path=path)
        targ_var = np.append(targ_var, _single_targ_var, axis=0)

    # remove the first own design dimensions.
    targ_var = targ_var[1:, :, :]

    # remove south polar and greeland datasets.
    targ_var[:, 0:32, 299:346] = np.nan
    targ_var[:, 0:18, 287:300] = np.nan

    # save
    np.save(save_name+'.npy', targ_var)


if __name__ == "__main__":

    from parser import get_parse
    config = get_parse()

    main(raw_path=config.raw_path,
         out_path=config.prelminary_path,
         save_name=config.save_name,
         NLAT=config.NLAT,
         NLON=config.NLON)
