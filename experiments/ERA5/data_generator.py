import sys
sys.path.append('../../')
from MetReg.utils.utils import (_get_nan_mask, _get_task_from_regions,
                                save2pickle)
from MetReg.data.data_preprocessor import Data_preprocessor
from MetReg.data.data_generator import Data_generator
import numpy as np
import os



def main(X,
         y,
         input_path,
         intervel=18,
         len_inputs=10,
         window_size=7,):

    # mkdir input path using for generating inputs.
    if not os.path.exists(input_path):
        os.mkdir(input_path)

    # get region
    region = _get_task_from_regions(
        X.shape[-3], X.shape[-2], interval=intervel)

    # generate
    for num_jobs, attr in enumerate(region):
        print('Processing job {} of {} jobs'.format(num_jobs, len(region)))

        # generate X, y for region
        X_region = X[:, attr[0]:attr[0]+intervel, attr[1]:attr[1]+intervel, :]
        y_region = y[:, attr[0]:attr[0]+intervel, attr[1]:attr[1]+intervel, :]
        assert y_region.shape[-2] == intervel and y_region.shape[-3] == intervel

        # generate nan mask for region
        mask = _get_nan_mask(y_region[:, :, :, 0])

        # data preprocess
        dp = Data_preprocessor(X_region, y_region,
                               interp=True, normalize=True)
        X_region, y_region = dp()

        # data generator
        dg = Data_generator(X_region, y_region,
                            train_valid_ratio=0.2,
                            len_inputs=len_inputs,
                            window_size=window_size)
        data = dg()

        save2pickle(data, input_path, 'ERA5_DD_A1_case_' +
                    str(num_jobs)+'.pickle')
        save2pickle(mask, input_path, 'nan_mask_case_' +
                    str(num_jobs)+'.pickle')


if __name__ == "__main__":

    preliminary_path = '/hard/lilu/ERA5/preliminary/'
    st = np.load(preliminary_path+'ERA5_1981_2017_DD_A1_st_lv1.npy')
    swv = np.load(preliminary_path+'ERA5_1981_2017_DD_A1_swv_lv1.npy')
    mtpr = np.load(preliminary_path + 'ERA5_1981_2017_DD_A1_mtpr.npy')
    t2m = np.load(preliminary_path + 'ERA5_1981_2017_DD_A1_t2m.npy')

    X = np.concatenate((swv, st, mtpr, t2m), axis=-1)
    y = swv

    del swv, st, mtpr, t2m

    main(X,
         y,
         input_path='/hard/lilu/ERA5/inputs/',
         intervel=18,
         len_inputs=10,
         window_size=7,
         )
