from enum import auto
import numpy as np
import autokeras as ak
import os
from tcn import TCN
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

os.environ["OPENBLAS_NUM_THREADS"] = "2"
x_train = np.load('../test/139x139/x_train.npy')[:, 0, :1, :]
y_train = np.load('../test/139x139/y_train.npy')[:, 0,:1, :]
x_train = np.concatenate([x_train, y_train], axis=-1)[:-3]
y_train = y_train[3:]

x_test = np.load('../test/139x139/x_test.npy')[:, 0, :1, :]
y_test = np.load('../test/139x139/y_test.npy')[:, 0, :1, :]
x_test = np.concatenate([x_test, y_test], axis=-1)[:-3]
y_test = y_test[3:]

print(y_test.shape)


tcn_layer = TCN(input_shape=(1, 7))
m = Sequential([
       tcn_layer,
       #LSTM(256, return_sequences=True, input_shape=(1,7)),
       Dense(1)
])

m.compile(optimizer='adam', loss='mse')

m.fit(x_train, y_train, epochs=50)

y_pred = m.predict(x_test)
print(y_pred.shape)
#print(x_train.shape)
#from autosklearn.regression import AutoSklearnRegressor
#from flaml import AutoML

# Initialize an AutoML instance
#automl = AutoML()
# Specify automl goal and constraint
#automl_settings = {
#    "time_budget": 10,  # in seconds
#    "metric": 'r2',
#    "task": 'regression',
#}
#X_train, y_train = fetch_california_housing(return_X_y=True)
# Train with labeled input data
#automl.fit(X_train=x_train, y_train=y_train,
#           **automl_settings)
# Predict
# Export the best model
#print(automl.model)
#reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
#reg.fit(x_train, y_train, batch_size=128,  epochs=50)
"""
predict_from = 1
predict_until = 10
lookback = 3
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective="val_loss",
)
# Train the TimeSeriesForecaster with train data
clf.fit(
    x=x_train,
    y=y_train,
    #validation_data=(data_x_val, data_y_val),
    batch_size=30,
    epochs=10,
)
"""
# Predict with the best model(includes original training data).
#y_pred = reg.predict(x_test)
# Evaluate the best model with testing data.

#automl = AutoSklearnRegressor(
#    time_left_for_this_task=120, 
#    per_run_time_limit=30,
#    tmp_folder='/hard/lilu/HRSEPP/tmp/autosklearn_regression_tmp')
print('1')
#automl.fit(x_train, y_train)
print('2')
#print(automl.leaderboard())
print('3')
#y_pred = automl.predict(x_test)
from sklearn.metrics import r2_score
print("Test R2 score:", r2_score(np.squeeze(y_test), np.squeeze(y_pred)))
