from __future__ import print_function

import glob
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import gen_convLSTM_input, gen_esa_x_y, gen_flx_x_y
from model import (DL_Models, ML_Models, compare_models_esa,
                   compare_models_flx, compare_models_smap)
from utils import folder_list, gen_nest_dict, r2, save_log, tictoc


@tictoc
def main_flx(folder_path="/hard/lilu/Fluxnet/FLX2015_Tier1/FLX*",
             out_path="/work/lilu/Soil-pred/results/FLX/",
             out_file='compare_models_flx_short.pickle',
             time_resolution='DD',
             cv=False):

    dict_ = dict()

    # get the list in sorted way
    l = folder_list(folder_path, datasets=0)

    # loop all folders
    for index in range(len(l)):

        # generate target file path
        file_path = folder_list(
            l[index] + "/*FULLSET_" + time_resolution + "*", datasets=0)

        # if file path is not empty
        if file_path:

            # print and save site name
            print("\n***********************************")
            print("\nsite name is {}".format(
                file_path[0].split("_")[2]))  # changed in different env

            # train mode
            try:
                log = compare_models_flx(file_path[0])

                # output
                dict_[file_path[0].split("_")[2]] = log

                print(dict_.keys())
            except:
                print("\nThis site have no swc")
        else:
            print("\n***********************************")
            print("\nThis site have no file")

    # save model
    save_log(dict_, out_path=out_path, out_file=out_file)


@tictoc
def main_esa(folder_path='/hard/lilu/LDAS/GLDAS/Result',
             out_path="/work/lilu/Soil-pred/results/ESA/",
             out_file='compare_models_esa_short_1.pickle',
             time_resolution='DD',
             cv=False):

    # load data
    sm = np.load(folder_path + "/SM_usa.npy")[1:, np.newaxis, :, :]

    timesteps, _, lat, lon = np.shape(sm)

    rain = np.load(folder_path+"/rain_usa.npy")[0:3288, np.newaxis, :, :]
    wind = np.load(folder_path+"/wind_usa.npy")[0:3288, np.newaxis, :, :]
    Psurf = np.load(folder_path+"/Psurf_usa.npy")[0:3288, np.newaxis, :, :]
    LWdown = np.load(folder_path+"/LWdown_usa.npy")[0:3288, np.newaxis, :, :]
    SWdown = np.load(folder_path+"/SWdown_usa.npy")[0:3288, np.newaxis, :, :]
    Qair = np.load(folder_path+"/Qair_usa.npy")[0:3288, np.newaxis, :, :]
    Tair = np.load(folder_path + "/Tair_usa.npy")[0:3288, np.newaxis, :, :]
    # period = np.load(folder_path + '/period.npy')[0:3288, np.newaxis, :, :]

    qc = np.ones((timesteps, 1, lat, lon))

    input_ = np.concatenate((sm, qc, rain, wind, Psurf, LWdown,
                             SWdown, Qair, Tair), axis=1)

    del sm, rain, wind, Psurf, LWdown, SWdown, Qair, Tair

    dict_ = gen_nest_dict(len(range(1, lat, 20)))

    for i, lat_index in enumerate(range(1, lat, 20)):
        for j, lon_index in enumerate(range(1, lon, 20)):

            print("\n***********************************")
            print("\nNOW WE ARE HANDLING LAT {} AND LON {}".format(
                lat_index, lon_index))

            df = pd.DataFrame(input_[:, :, lat_index, lon_index], columns=[
                'SM', 'QC', 'RAIN', 'wind', 'Psurf', 'LWdown', 'SWdown', 'Qair', 'Tair'])

            try:
                log = compare_models_esa(df)

                dict_[i][j] = log
                print(dict_)

            except:
                print("\n***********************************")
                print("\nThis site have no swc")

    save_log(dict_, out_path=out_path, out_file=out_file)


@tictoc
def main_smap(folder_path='/hard/lilu/SMAP_L4/results',
              out_path="/work/lilu/Soil-pred/results/SMAP/",
              out_file='compare_models_smap_short.pickle',
              time_resolution='3HH',
              cv=False):

    # load USA data
    sm = np.load(folder_path+'/SMAP_sm_usa.npy')[:, np.newaxis, :, :]
    p = np.load(folder_path+'/SMAP_p_usa.npy')[:, np.newaxis, :, :]
    st = np.load(folder_path + '/SMAP_st_usa.npy')[:, np.newaxis, :, :]

    # set dims
    timesteps, _, lat, lon = np.shape(sm)

    print(lat)
    print(lon)

    # NOTE: construct a default QC
    qc = np.ones((timesteps, 1, lat, lon))

    # concat input
    input_ = np.concatenate((sm, qc, p, st), axis=1)

    # clear cache
    del sm, qc, p, st

    # output
    dict_ = gen_nest_dict(len(range(1, lat, 30)))

    for i, lat_index in enumerate(range(1, lat, 30)):
        for j, lon_index in enumerate(range(1, lon, 30)):

            print("\n***********************************")
            print("\nNOW WE ARE HANDLING LAT {} AND LON {}".format(
                lat_index, lon_index))

            df = pd.DataFrame(input_[:, :, lat_index, lon_index], columns=[
                'SM', 'QC', 'RAIN', 'ST'])

            try:

                log = compare_models_smap(df)

                # output
                dict_[i][j] = log
                print(dict_.keys())

            except:

                dict_[i][j] = {}
                print("\n***********************************")
                print("\nThis site have no swc")

    save_log(dict_, out_path=out_path, out_file=out_file)


@tictoc
def main_convLSTM_smap(folder_path='/hard/lilu/SMAP_L4/results',
                       out_path="/work/lilu/Soil-pred/results/SMAP/",
                       out_file='convLSTM_smap_medium.pickle',
                       time_resolution='3HH',
                       cv=False):

    sm = np.load(folder_path + '/SMAP_sm_usa.npy')[:, :, :]
    n = round(0.8 * sm.shape[0])

    y_train, y_valid = sm[:n, 125:155, 275:305], sm[n:, 125:155, 275:305]

    print(y_train.shape)
    print(y_valid.shape)

    del sm

    _x_train, _y_train = gen_convLSTM_input(y_train)
    _x_valid, _y_valid = gen_convLSTM_input(
        np.concatenate((y_train[-156:, :, :], y_valid), axis=0))

    del y_train, y_valid

    print(_x_train.ndim)
    print(_y_train.ndim)

    model = DL_Models(_x_train, _x_valid, _y_train, _y_valid, DNN=False)
    model.convLSTM()
    log = model.train()

    save_log(log, out_path=out_path, out_file=out_file)


@tictoc
def main_convLSTM_esacci(folder_path='/hard/lilu/LDAS/GLDAS/Result',
                         out_path="/work/lilu/Soil-pred/results/ESA/",
                         out_file='convLSTM_esa_short.pickle',
                         time_resolution='DD',
                         cv=False):

    # load data
    sm = np.load(folder_path + "/SM_usa.npy")[1:, :, :]
    n = round(0.8 * sm.shape[0])
    sm[sm == -9999] = np.nan
    sm = sm / 100

    y_train, y_valid = sm[:n, 40:70, 110:140], sm[n:, 40:70, 110:140]

    print(y_train.shape)
    print(y_valid.shape)

    del sm

    _x_train, _y_train = gen_convLSTM_input(y_train)
    _x_valid, _y_valid = gen_convLSTM_input(
        np.concatenate((y_train[-31:, :, :], y_valid), axis=0))

    del y_train, y_valid

    print(_x_train.ndim)
    print(_y_train.ndim)

    model = DL_Models(_x_train, _x_valid, _y_train, _y_valid, DNN=False)
    model.convLSTM()
    log = model.train()

    save_log(log, out_path=out_path, out_file=out_file)


@tictoc
def main_att_convlstm_smap(folder_path='/hard/lilu/SMAP_L4/results',
                           out_path="/work/lilu/Soil-pred/results/SMAP/",
                           out_file='SMAP_3HH_7D_AttConvLSTM_10.pickle',
                           time_resolution='3HH',
                           cv=False):

    sm = np.load(folder_path + '/SMAP_sm_usa.npy')[:, :, :]
    n = round(0.8 * sm.shape[0])

    y_train, y_valid = sm[:n, 145:155, 295:305], sm[n:, 145:155, 295:305]

    print(y_train.shape)
    print(y_valid.shape)

    del sm

    _x_train, _y_train = gen_convLSTM_input(y_train)
    _x_valid, _y_valid = gen_convLSTM_input(
        np.concatenate((y_train[-156:, :, :], y_valid), axis=0))

    del y_train, y_valid

    print(_x_train.ndim)
    print(_y_train.ndim)

    model = DL_Models(_x_train, _x_valid, _y_train, _y_valid, DNN=False)
    model.att_convLSTM()
    log = model.train()

    save_log(log, out_path=out_path, out_file=out_file)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False)

    parser.add_argument('--epoch', type=int, default=10000,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The size of batch per gpu')
    parser.add_argument('--print_freq', type=int, default=500,
                        help='The number of image_print_freqy')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='The number of ckpt_save_freq')

    parser.add_argument('--g_opt', type=str, default='adam',
                        help='learning rate for generator')
    parser.add_argument('--d_opt', type=str, default='adam',
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0004,
                        help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--gpl', type=float, default=10.0,
                        help='The gradient penalty lambda')

    parser.add_argument('--z_dim', type=int, default=128,
                        help='Dimension of noise vector')
    parser.add_argument('--image_size', type=int,
                        default=64, help='The size of image')
    parser.add_argument('--sample_num', type=int, default=16,
                        help='The number of sample images')

    parser.add_argument('--g_conv_filters', type=int,
                        default=16, help='basic filter num for generator')
    parser.add_argument('--g_conv_kernel_size', type=int,
                        default=4, help='basic kernel size for generator')
    parser.add_argument('--d_conv_filters', type=int,
                        default=16, help='basic filter num for disciminator')
    parser.add_argument('--d_conv_kernel_size', type=int,
                        default=4, help='basic kernel size for disciminator')

    parser.add_argument('--restore_model', action='store_true',
                        default=False, help='the latest model weights')
    parser.add_argument('--g_pretrained_model', type=str,
                        default=None, help='path of the pretrained model')
    parser.add_argument('--d_pretrained_model', type=str,
                        default=None, help='path of the pretrained model')

    parser.add_argument('--data_path', type=str, default='./data')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return parser.parse_args()


if __name__ == "__main__":

    # main_flx()
    # main_esa()
    # main_smap()
    # main_convLSTM_smap()
    # main_convLSTM_esacci()
    main_att_convlstm_smap()
