import numpy as np
from MetReg.data.data_preprocessor import Data_preprocessor
from MetReg.data.data_generator import Data_generator

swc = np.load('/work/lilu/swvl1.npy')
st = np.load('/work/lilu/stl1.npy')

X = np.concatenate((swc, st), axis=-1)
y = swc

dp = Data_preprocessor(X, y)

del X, y, swc, st

X,y = dp()
print(dp)
print(X.shape)
print(y.shape)

dg = Data_generator(X,y)
dg()