# from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns


def output_preprocessor(mdl_list=['Ridge', 'KNN', 'SVM', 'ELM', 'RF', 'Adaboost',
                                  'GBDT', 'Xgboost', 'LightGBM', 'ConvLSTM'],
                        file_type='.npy',
                        file_path='/hard/lilu/score/'):

    score = np.full((len(mdl_list), 14, 150, 360), np.nan)

    for i, mdl_name in enumerate(mdl_list):
        # generate score path of model
        path = file_path + mdl_name.lower() + '_score' + file_type

        # load score file
        score[i, :, :, :] = np.load(path)[:, :150, :]

    score[:, :, 0:32, 299:346] = np.nan
    score[:, :, 0:18, 287:300] = np.nan

    bias = score[:, 0, :, :]
    rmse = score[:, 1, :, :]
    nse = score[:, 2, :, :]
    r2 = score[:, 3, :, :]
    wi = score[:, 4, :, :]
    kge = score[:, 5, :, :]
    r = score[:, 6, :, :]
    m1 = score[:, 7, :, :]  # true
    m2 = score[:, 8, :, :]  # predict
    mae = score[:, 9, :, :]
    mse = score[:, 10, :, :]
    score_ = score[:, 11, :, :]
    std1 = score[:, 12, :, :]
    std2 = score[:, 13, :, :]

    bias[r2 < 0] = np.nan
    nse[r2 < 0] = np.nan
    rmse[r2 < 0] = np.nan
    wi[r2 < 0] = np.nan
    kge[r2 < 0] = np.nan
    r[r2 < 0] = np.nan
    mae[r2 < 0] = np.nan
    mse[r2 < 0] = np.nan
    score_[r2 < 0] = np.nan
    r2[r2 < 0] = np.nan

    return bias, rmse, nse, wi, kge, r, m1, m2, mae, mse, score_, std1, std2


def get_na_mask_data(data):
    mask = ~np.isnan(data)
    data = [d[m] for d, m in zip(data, mask)]
    return data


def figure1(mdl_list=['Ridge', 'KNN', 'SVM', 'ELM', 'RF', 'Adaboost',
                      'GBDT', 'Xgboost', 'LightGBM', 'ConvLSTM'],
            color_list=['pink', 'lightblue', 'gray', 'yellow', 'lightgreen',
                        'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen',
                        'red'],
            file_type='.npy',
            file_path='/hard/lilu/score/'):
    bias, rmse, nse, wi, kge, r, m1, m2, mae, mse, score_ = output_preprocessor(
        mdl_list=mdl_list,
        file_path=file_path,
        file_type=file_type)

    # boxplot
    plt.figure(figsize=(20, 11))

    # ----------------------------------
    ax1 = plt.subplot(3, 3, 1)
    data = get_na_mask_data(bias.reshape(len(mdl_list), -1))

    plot1 = ax1.boxplot(data, vert=True, patch_artist=True,  # labels=model_list, \
                        showfliers=False, showmeans=True)
    plt.xticks(rotation=300)
    ax1.set_title('(a) Bias')

    ax1.axhline(y=0, c="black", lw=0.2)

    # -------------------------------------
    ax2 = plt.subplot(3, 3, 2)
    data = get_na_mask_data(rmse.reshape(len(mdl_list), -1))
    plot2 = ax2.boxplot(data, vert=True, patch_artist=True,  # labels=model_list, \
                        showfliers=False, showmeans=True)
    plt.xticks(rotation=300)
    ax2.set_title('(b) Root Mean Squared Error')

    # -------------------------------------
    ax3 = plt.subplot(3, 3, 3)

    data = get_na_mask_data(nse.reshape(len(mdl_list), -1))
    plot3 = ax3.boxplot(data,
                        vert=True,
                        patch_artist=True,  # labels=model_list, \
                        showfliers=False,
                        showmeans=True)
    plt.xticks(rotation=300)
    ax3.set_title('(c) Nash-Sutcliffe Efficiency Coefficient')

    # -------------------------------------
    ax4 = plt.subplot(3, 3, 4)

    data = get_na_mask_data(wi.reshape(len(mdl_list), -1))

    plot4 = ax4.boxplot(data, vert=True, patch_artist=True,  # labels=mdl_list, \
                        showfliers=False, showmeans=True)

    plt.xticks(rotation=300)
    ax4.set_title('(d) Willmott Index')

    # -------------------------------------
    ax5 = plt.subplot(3, 3, 5)

    data = get_na_mask_data(kge.reshape(len(mdl_list), -1))

    plot5 = ax5.boxplot(data, vert=True, patch_artist=True,  # labels=mdl_list, \
                        showfliers=False, showmeans=True)

    plt.xticks(rotation=300)
    ax5.set_title('(e) Kling-Gupta Efficiency')

    # -------------------------------------
    ax6 = plt.subplot(3, 3, 6)

    data = get_na_mask_data(r.reshape(len(mdl_list), -1))

    plot6 = ax6.boxplot(data, vert=True, patch_artist=True,  # labels=mdl_list, \
                        showfliers=False, showmeans=True)

    plt.xticks(rotation=300)
    ax6.set_title('(f) Pearson’s Correlation Index')

    # -------------------------------------
    ax7 = plt.subplot(3, 3, 7)

    data = get_na_mask_data(mae.reshape(len(mdl_list), -1))

    plot7 = ax7.boxplot(data, vert=True, patch_artist=True, labels=mdl_list,
                        showfliers=False, showmeans=True)

    plt.xticks(rotation=300)
    ax7.set_title('(g) Mean Absolute Error')

    # -------------------------------------
    ax8 = plt.subplot(3, 3, 8)

    data = get_na_mask_data(mse.reshape(len(mdl_list), -1))

    plot8 = ax8.boxplot(data, vert=True, patch_artist=True, labels=mdl_list,
                        showfliers=False, showmeans=True)

    plt.xticks(rotation=300)
    ax8.set_title('(h) Mean Squared Error')

    # -------------------------------------
    ax9 = plt.subplot(3, 3, 9)

    data = get_na_mask_data(score_.reshape(len(mdl_list), -1))

    plot9 = ax9.boxplot(data, vert=True, patch_artist=True, labels=mdl_list,
                        showfliers=False, showmeans=True)

    plt.xticks(rotation=300)
    ax9.set_title('(i) MetReg score')

    for bplot in (plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9):
        for patch, color in zip(bplot['boxes'], color_list):
            patch.set_facecolor(color)

    plt.savefig('/hard/lilu/boxplot_model_score.pdf')


def figure3(mdl_list=['Ridge', 'KNN', 'SVM', 'ELM', 'RF', 'Adaboost',
                      'GBDT', 'Xgboost', 'LightGBM', 'ConvLSTM'],
            color_list=['pink', 'lightblue', 'gray', 'yellow', 'lightgreen',
                        'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen',
                        'red'],
            file_type='.npy',
            file_path='/hard/lilu/score/'):

    koppen_index = np.load('koppen_index.npy').astype('float')
    koppen_index[koppen_index == 0] = np.nan
    koppen_index = np.concatenate(
        (koppen_index[:, 181:], koppen_index[:, :181]), axis=-1)

    for i in range(1, 4):
        koppen_index[koppen_index == i] = 0
    for i in range(4, 6):
        koppen_index[koppen_index == i] = 1
    for i in range(6, 8):
        koppen_index[koppen_index == i] = 2
    for i in range(8, 11):
        koppen_index[koppen_index == i] = 3
    for i in range(11, 14):
        koppen_index[koppen_index == i] = 4
    for i in range(14, 17):
        koppen_index[koppen_index == i] = 5
    for i in range(17, 21):
        koppen_index[koppen_index == i] = 6
    for i in range(21, 25):
        koppen_index[koppen_index == i] = 7
    for i in range(25, 29):
        koppen_index[koppen_index == i] = 8
    for i in range(29, 31):
        koppen_index[koppen_index == i] = 9

    bias, rmse, nse, wi, kge, r, m1, m2, mae, mse, score_ = output_preprocessor(
        mdl_list=mdl_list,
        file_path=file_path,
        file_type=file_type)

    plt.figure()
    for i in range(10):
        print(i)
        mask = np.where(koppen_index[:150, :] == i)

        score = score_[:, mask[0], mask[1]]
        print(score.shape)
        data = get_na_mask_data(score)

        ax = plt.subplot(5, 2, i+1)
        ax.boxplot(data, vert=True, patch_artist=True,
                   showfliers=False, )  # showmeans=True)

    plt.savefig('/hard/lilu/boxplot_koppen_index.pdf')


def figure4(mdl_list=['Ridge', 'KNN', 'SVM', 'ELM', 'RF', 'Adaboost',
                      'GBDT', 'Xgboost', 'LightGBM', 'ConvLSTM'],
            color_list=['pink', 'lightblue', 'gray', 'yellow', 'lightgreen',
                        'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen',
                        'red'],
            file_type='.npy',
            file_path='/hard/lilu/score/'):
    bias, rmse, nse, wi, kge, r, m1, m2, mae, mse, score_, std1, std2 = output_preprocessor(
        mdl_list=mdl_list,
        file_path=file_path,
        file_type=file_type)

    koppen_index = np.load('koppen_index.npy').astype('float')
    koppen_index[koppen_index == 0] = np.nan
    koppen_index = np.concatenate(
        (koppen_index[:, 181:], koppen_index[:, :181]), axis=-1)

    for i in range(1, 4):
        koppen_index[koppen_index == i] = 0
    for i in range(4, 6):
        koppen_index[koppen_index == i] = 1
    for i in range(6, 8):
        koppen_index[koppen_index == i] = 2
    for i in range(8, 11):
        koppen_index[koppen_index == i] = 3
    for i in range(11, 14):
        koppen_index[koppen_index == i] = 4
    for i in range(14, 17):
        koppen_index[koppen_index == i] = 5
    for i in range(17, 21):
        koppen_index[koppen_index == i] = 6
    for i in range(21, 25):
        koppen_index[koppen_index == i] = 7
    for i in range(25, 29):
        koppen_index[koppen_index == i] = 8
    for i in range(29, 31):
        koppen_index[koppen_index == i] = 9

    #ref_std = np.nanmean(std1)
    #std = np.nanmean(std2, axis=(-1, -2))

    #nse = np.nanmean(nse, axis=(-1, -2))
    color_list = color_list

    from MetReg.plot.__plotting import plot_taylor_diagram

    for i in range(10):

        print(i)
        mask = np.where(koppen_index[:150, :] == i)

        sns.kdeplot(nse[-1, mask[0], mask[1]])

        """
        std_ = np.nanmean(std2[:, mask[0], mask[1]], axis=-1)
        ref_std_ = np.nanmean(std1[:, mask[0], mask[1]])
        nse_ = np.nanmean(r[:, mask[0], mask[1]], axis=-1)
        fig = plt.figure()
        
        plot_taylor_diagram(std_, nse_, ref_std_, fig=fig, colors=color_list)
        
        plt.savefig('/hard/lilu/taylor_diagram'+'_'+str(i)+'.pdf')
        """
    plt.savefig('/hard/lilu/kde.pdf')


if __name__ == '__main__':
    # figure1()
    # figure3()
    figure4()
