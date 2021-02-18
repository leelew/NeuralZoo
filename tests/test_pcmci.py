# Imports
from CausalityTest import Granger
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# matplotlib inline
# use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn
import h5py
import pandas as pd

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

# load data
# data = pd.read_csv('Data/data_site.csv',
#                   usecols=['P_F', 'SWC_F_MDS_1', 'TA_F', 'PA_F', 'TS_F_MDS_1'])
data = pd.read_csv('Data/data_site.csv',
                   usecols=['SWC_F_MDS_1', 'P_F'])
# data = pd.read_csv('Data/data_site.csv',
#                   usecols=['P_F', 'SWC_F_MDS_1', 'TA_F'])
data = data.replace(-9999, np.nan)
data = data.fillna(data.mean()).values
#var_names = [r'$TA$', r'$PA$', r'$P$', r'$T$', r'$SWC$']
#var_names = [r'$P$', r'$SWC$']
#var_names = [r'$P$', r'$SWC$', r'$TA$']
var_names = [r'$P$', r'$SWC$', r'$TA$', r'$PA$', r'$T$']

# print data
dataframe = pp.DataFrame(np.array(data), var_names=var_names)
tp.plot_timeseries(dataframe)
plt.show()

# main for PCMCI
parcorr = ParCorr()
pcmci_parcorr = PCMCI(dataframe=dataframe,
                      cond_ind_test=parcorr,
                      verbosity=2)
results = pcmci_parcorr.run_pcmci(tau_max=2,
                                  pc_alpha=0.2)
pcmci_parcorr.print_significant_links(p_matrix=results['p_matrix'],
                                      val_matrix=results['val_matrix'],
                                      alpha_level=0.01)
link_matrix = pcmci_parcorr.return_significant_parents(pq_matrix=results['p_matrix'],
                                                       val_matrix=results['val_matrix'],
                                                       alpha_level=0.01)['link_matrix']
# Plot time series graph
tp.plot_time_series_graph(figsize=(6, 3),
                          val_matrix=results['val_matrix'],
                          link_matrix=link_matrix,
                          var_names=var_names,
                          link_colorbar_label='MCI')
