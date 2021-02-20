import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

from MetReg.utils.utils import _get_folder_list

l = glob.glob('./score/*score.npy', recursive=True)
print(l)

score = np.full((len(l), 4, 150, 360),np.nan)
model_list = []

for i, file_path in enumerate(l):
    file_name = file_path.split('/')[-1]
    print(file_name)
    score[i, :, :,:] = np.load(file_path)[:,:150,:]
    model = file_name.split('_')[0]
    model_list.append(model)

color_list=['pink','lightblue','lightgreen','lightgreen','lightgreen', \
        'red','yellow','lightgreen','gray','gray','lightgreen']
print(score.shape)

bias = score[:,0,:,:]
rmse = score[:,1,:,:]
nse = score[:,2,:,:]
r2 = score[:,3,:,:]

bias[bias< -0.1] = np.nan
bias[bias>0.1] = np.nan


r2[r2<0] = np.nan
#r2[r2>1] = np.nan

nse[nse<0] = np.nan
rmse[rmse<0] = np.nan


print(len(model_list))
print(len(color_list))


# boxplot
plt.figure(figsize=(20,10))

#----------------------------------
ax1 = plt.subplot(2,2,1)

bias = bias.reshape(11, -1)

mask = ~np.isnan(bias)
data = [d[m] for d, m in zip(bias, mask)]

print(len(data))
plot1 = ax1.boxplot(data, vert=True, patch_artist=True, #labels=model_list, \
showfliers=False, showmeans=True)
plt.xticks(rotation=300)
ax1.set_title('(a) bias')

ax1.axhline(y=0, c="black", lw=0.2)

#-------------------------------------
ax2 = plt.subplot(2,2,2)

rmse = rmse.reshape(11, -1)

mask = ~np.isnan(rmse)
data = [d[m] for d, m in zip(rmse, mask)]

print(len(data))
plot2 = ax2.boxplot(data, vert=True, patch_artist=True, #labels=model_list, \
showfliers=False, showmeans=True)

plt.xticks(rotation=300)
ax2.set_title('(b) RMSE')

ax2.axhline(y=0, c="black", lw=0.2)

#-------------------------------------
ax3 = plt.subplot(2,2,3)

nse = nse.reshape(11, -1)

mask = ~np.isnan(nse)
data = [d[m] for d, m in zip(nse, mask)]

print(len(data))
plot3 = ax3.boxplot(data, vert=True, patch_artist=True, labels=model_list, \
showfliers=False, showmeans=True)

plt.xticks(rotation=300)
ax3.set_title('(c) NSE')

#ax2.axhline(y=0, c="black", lw=0.2)

"""
#-------------------------------------
ax4 = plt.subplot(2,2,4)

r2 = r2.reshape(11, -1)

mask = ~np.isnan(r2)
data = [d[m] for d, m in zip(r2, mask)]

print(len(data))
plot4 = ax4.boxplot(data, vert=True, patch_artist=True, labels=model_list, \
showfliers=False, showmeans=True)

plt.xticks(rotation=300)
ax4.set_title('(d) R${^2}$')

#ax2.axhline(y=0, c="black", lw=0.2)
"""


for bplot in (plot1, plot2,plot3):
    for patch, color in zip(bplot['boxes'], color_list):
        patch.set_facecolor(color)

plt.savefig('boxplot_model_score.pdf')


"""




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
"""