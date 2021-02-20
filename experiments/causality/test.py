#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:20:04 2019

@author: lewlee
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


time_len = 3000
ay, by = 1,2
ey = np.random.uniform(ay,by,time_len).reshape(-1,1)
ax, bx = 3,4
ex = np.random.uniform(ax,bx,time_len).reshape(-1,1)
#ez = np.random.uniform(0.3,0.4,time_len).reshape(-1,1)
#ew = np.random.uniform(1,2,time_len).reshape(-1,1)
#ez = np.random.normal(loc=0.0, scale=1.0, size=(time_len,1))
#ew = np.random.normal(loc=0.0, scale=1.0, size=(time_len,1))
ez = np.random.poisson(lam=5, size=time_len)
ew = np.random.poisson(lam=3, size=time_len)

a,b,c,d,e = 0.5, 0.4, 1.2, 1, 1.5

# construct x,y
Y = ey*np.ones((time_len,1))
X = a*Y+ex*np.ones((time_len,1))

zz = 0
ww = 0
Z = np.zeros((time_len,1))
W = np.zeros((time_len,1))

# construct z,w
for i in range(time_len-1):
    Z[i] = zz
    W[i] = ww
    zz = b*zz+ez[i+1] # z(1) = b*z(0)+c*y(0)+ez(1)   c*Y[i]
    ww = d*ww+e*zz+ew[i+1] # w(1) = d*w(0)+e*z(1)+ew(1)
    
    
"""
Main Test
"""
from CausalityTest import Corr, Granger, CMI, DirectNGAM
from Figure import Figure
from Data import Data
from Detrend import Detrend

df = pd.DataFrame({
     'X': pd.Series(np.squeeze(X)),
     'Y': pd.Series(np.squeeze(Y)),
     'Z': pd.Series(np.squeeze(Z)),
     'W': pd.Series(np.squeeze(W))})
    
#df = pd.DataFrame({
#     'Z': pd.Series(np.squeeze(Z)),
#     'W': pd.Series(np.squeeze(W))})
    
df = df.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
# class  
cr = Corr(df,2)
gr = Granger(df,2)
cmi = CMI(df,2)
dn = DirectNGAM(df,2)
f = Figure(df,2)
# test
corr_parents = cr._correlation_test() 
gc_parents = gr._granger_test(corr_parents)
cmi_parents,return_score = cmi._CMI_test(df.values,max_tau=2,max_cond_dims=1,parents=gc_parents,alpha=0.1)
parents = dn._append_comteporaneous_link(df,cmi_parents)
# figure
f._plot_graph_single(tau=2,parents=corr_parents)  
f._plot_graph_single(tau=2,parents=gc_parents)  
f._plot_graph_single(tau=2,parents=cmi_parents)  
f._plot_graph_single(tau=2,parents=parents)  


