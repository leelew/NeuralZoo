import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


a = np.load('lightgbm_score.npy')[3,:150,:].reshape(-1, 1)
b = np.load('xgboost_score.npy')[3, :150,:].reshape(-1,1)
c = np.load('rf_score.npy')[3,:150,:].reshape(-1,1)
d = np.load('gbdt_score.npy')[3,:150,:].reshape(-1,1)
e = np.load('adaboost_score.npy')[3,:150,:].reshape(-1,1)

f = np.load('linear_score.npy')[3,:150,:].reshape(-1,1)
g = np.load('svm_score.npy')[3,:150,:].reshape(-1,1)
h = np.load('ridge_score.npy')[3,:150,:].reshape(-1,1)
i = np.load('elm_score.npy')[3,:150,:].reshape(-1,1)
j = np.load('knn_score.npy')[3,:150,:].reshape(-1,1)

k = np.load('convlstm_score.npy')[3,:150,:].reshape(-1,1)
r2 = np.load('convlstm_score.npy')[3,:150,:]
r21 = np.load('lightgbm_score.npy')[3,:150,:]
r2[np.isnan(r21)] = np.nan
k = r2.reshape(-1,1)


t = np.concatenate((a,b,c,d,e,f,g,h,i,j,k), axis=-1)
t[t<0] = np.nan
t[t>1] = np.nan

mask = ~np.isnan(t)

data = [d[m] for d, m in zip(t.T, mask.T)]


plt.boxplot(data, labels=['lightgbm', 'xgboost','rf','gbdt','adaboost','lisvr','svm','ridge','elm','knn','convlstm'],showmeans=True,showfliers=False)
plt.xticks(rotation=300)