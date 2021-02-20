import numpy as np
from MetReg.data.data_preprocessor import Data_preprocessor
from MetReg.data.data_generator import Data_generator
from MetReg.data.data_analyser import Data_analyser_sts, Data_analyser_ts
from MetReg.data.data_loader import Data_loader
from MetReg.data.data_validator import Data_validator


def test_data_preprocessor(): pass


swc = np.load('/work/lilu/swvl1.npy')
st = np.load('/work/lilu/stl1.npy')

X = np.concatenate((swc, st), axis=-1)
y = swc

dp = Data_preprocessor(X, y)

del X, y, swc, st

X, y = dp()
print(dp)
print(X.shape)
print(y.shape)

dg = Data_generator(X, y)
dg()
