import numpy as np
import sys
import time
import os
import pickle
import glob
# sys.path.append('/model.py')


def folder_list(folder_path, datasets=1):

    # return list of files in folder
    l = glob.glob(folder_path, recursive=True)

    # sorted using the file name (for ESACCI, SMAP, GLDAS et al)
    num_in_string = []
    # print(l[0].split("_"))
    for i in range(len(l)):
        if datasets == 0:  # flx
            # l.sort()
            return l
        if datasets == 1:
            num_in_string.append(
                int(re.sub("\D", "", l[i].split("-")[6])))  # ESACCI
        elif datasets == 2:
            num_in_string.append(
                int(re.sub("\D", "", l[i].split("_")[6])))  # SMAP
        elif datasets == 3:
            num_in_string.append(
                int(re.sub("\D", "", l[i].split(".")[1])))  # GLDAS
    # print(num_in_string)

    # sorted index
    sorted_index = sorted(range(len(num_in_string)),
                          key=lambda i: num_in_string[i])

    # sorted list by index
    sorted_l = []
    for i in range(len(sorted_index)):
        sorted_l.append(l[sorted_index[i]])
    return sorted_l  # list

# decorate


def tictoc(func):

    def wrapper(*args, **kwargs):

        begin = time.time()
        log = func(*args, **kwargs)
        end = time.time()
        print('cost {} second'.format(end-begin))

        return log

    return wrapper

# metrics


def r2(y_true, y_pred):

    y_mean = np.mean(y_true)
    sstotal = np.sum((y_true-y_mean)**2)
    ssres = np.sum((y_true-y_pred)**2)
    score = 1 - (ssres / sstotal)

    return np.array(score)


def gen_nest_dict(length):

    dict_ = {}
    for i in range(length):
        dict_[i] = {}

    return dict_


# save
def save_log(log,
             out_path="/work/lilu/Soil-pred/results/FLX/",
             out_file='compare_models.pickle'):

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # save model
    handle = open(out_path+out_file, 'wb')
    pickle.dump(log, handle, protocol=4)
    handle.close()


# generate default
def gen_test_data(ML=True):

    if ML:
        x_train = np.random.randn(1000, 5)
        x_valid = np.random.randn(100, 5)

        y_train = np.random.randn(1000, 1)
        y_valid = np.random.randn(100, 1)

    else:
        x_train = np.random.randn(1000, 60, 5)
        x_valid = np.random.randn(100, 60, 5)

        y_train = np.random.randn(1000, 1)
        y_valid = np.random.randn(100, 1)

    return x_train, x_valid, y_train, y_valid


def gen_params():
    """
    params = {
        'learning_rate': 0.3,
        'boosting': 'gbdt',
        #    'min_data_in_leaf': 5,
        'metric': 'r2',
        # 'max_depth': 3
        'max_bin': 100,
        # 'num_leaves': 31,
        #    'bagging_fraction': 0.5,
        #    'feature_fraction': 0.5,
        # 'min_split_gain': 0.3,
        #    'bagging_freq': 100,
        #    'lambda_l1': 0.1,
        #    'lambda_l2': 0.1
        # 'objective': 'regression'
    }
    """
    parameters = {}

    return parameters


if __name__ == "__main__":

    x_train, x_valid, y_train, y_valid = gen_test_data()
