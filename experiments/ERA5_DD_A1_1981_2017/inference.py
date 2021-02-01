import pickle

import numpy as np
from MetReg.api.model_io import model_benchmarker, model_loader
from sklearn.metrics import r2_score

from _data_generator import _get_task_from_regions
from main import _read_inputs


def inference(X,
              y,
              task,
              mdl_name='ml.tree.lightgbm',
              save_path='/hard/lilu/saved_model/', ):
    # load pickle
    f = open(save_path + mdl_name + '/saved_model_' +
             str(task) + '.pickle', 'rb')
    saved_model = pickle.load(f)

    # get shape
    N, _, nlat, nlon, _ = y.shape

    r2 = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            mdl = saved_model[i][j]
            if mdl is not None:
                y_predict = mdl.predict(X[:, :, i, j, :].reshape(N, -1))
                r2[i, j] = r2_score(y[:, 0, i, j, 0], y_predict)
            else:
                r2[i, j] = np.nan
    return r2


if __name__ == "__main__":

    region = _get_task_from_regions(180, 360, 18)
    r2_world = np.full((180, 360), np.nan)

    for num_jobs, attr in enumerate(region):

        print('now processing jobs {}'.format(num_jobs))
        X_train, X_valid, y_train, y_valid, mask = _read_inputs(task=num_jobs)
        r2 = inference(X_valid, y_valid, task=num_jobs)
        r2_world[attr[0]:attr[0]+18, attr[1]:attr[1]+18] = r2

    r2_world = np.concatenate((r2_world[:, 181:], r2_world[:, :181]), axis=-1)

    np.save('r2_ml_tree_lightgbm.npy', r2_world)

    """
    lon, lat = np.meshgrid(np.arange(-180, 180, 1), np.arange(90, -90, -1))

    plt.figure()

    m = Basemap()
    m.drawcoastlines(linewidth=0.2)
    x, y = m(lon, lat)

    sc = m.pcolormesh(x, y, r2_world, axis=)

    # plt.imshow(np.mean(avg_targ_var, axis=0))
    # plt.colorbar()
    plt.savefig('1.pdf')
    """
