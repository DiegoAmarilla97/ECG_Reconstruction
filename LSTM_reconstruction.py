# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:08:25 2022

@author: Diego
"""

import h5py
import numpy as np
import pandas as pd
import ecg_plot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# set parameters
sample_size= 827
window_size = 5
time_step = 1
n_units = 50
Draw = False
Error = {"Model": [], "Window Size": [], "R2": [], "MAE": [], "RMSE": [], "OutputC": [], "Sample Size":[]}

# load the dataset
filename = "C:/TESIS/ecg_tracings.hdf5"
with h5py.File(filename, "r") as f:
    data = np.array(f['tracings'])
    
  
aux = data[:sample_size, :, :]

del data

#take the windows
aux2 = []
for i in range(sample_size):
     for j in range(800, 3296 - window_size, time_step):
         aux2.append(aux[i, j:j + window_size, :])

aux2 = np.array(aux2)

# split the dataset
split = int(aux2.shape[0]*0.8)
train = aux2[:split, :, :]
test = aux2[split:, :, :]
del aux2

# 0,  1  , 2   , 3   , 4  , 5 , 6 , 7 , 8 , 9 , 10, 11
# DI, DII, DIII, AVL, AVF, AVR, V1, V2, V3, V4, V5, V6
input_channels = [0, 1, 7]
output_channel = 3
input_size = len(input_channels)

# split into input and outputs
train_X, train_y = train[:, :,input_channels], train[:, -1, output_channel]
test_X, test_y = test[:, :,input_channels], test[:, -1, output_channel]
del train, test

# define model
model = Sequential()
model.add(LSTM(n_units, input_shape=(window_size, input_size)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# fit network
model.fit(train_X, train_y, epochs=300, batch_size=2**32,
          validation_data=(test_X, test_y),
          callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])

# Save the weights
model.save_weights('C:/TESIS/Scripts/Checkpoints/V4_final')

if(Draw):
    #print ECG to be reconstructed
    ecg_size = 4096-1600
    ecg_number = 1
    ecg1 = test_y[ecg_size*(ecg_number-1):ecg_size*ecg_number]
    ecg1 = np.reshape(ecg1, (1, ecg_size))
    ecg_plot.plot(ecg1, sample_rate=400,
                  title='ECG Original (DIII)', columns=1, row_height=15)
    ecg_plot.show()
    #Reconstruct ECG and print it
    ecgaux = test_X[ecg_size*(ecg_number-1):ecg_size*ecg_number, :, :]
    yhat = model.predict(ecgaux)
    ecg2 = np.reshape(yhat, (1, ecg_size))
    ecg_plot.plot(ecg2, sample_rate=400,
                  title='ECG Reconstruido (DIII)', columns=1, row_height=15)
    ecg_plot.show()
    #

#Calculate RMSE and Correlation
yhat2 = model.predict(test_X)
test_y = np.reshape(test_y, (-1, 1))
rmse = np.sqrt(mean_squared_error(test_y, yhat2))
r2 = r2_score(test_y, yhat2)
mae = mean_absolute_error(test_y, yhat2)

#save yhat2
hf = h5py.File('C:/TESIS/Scripts/finalModel/AVL.h5','w')
hf.create_dataset('AVL_yhat', data=yhat2)
hf.create_dataset('AVL_test_y', data=test_y)
hf.close()

#log the error
Error["Model"].append(f"LSTM_{n_units}")
Error["Window Size"].append(window_size)
Error["R2"].append(r2)
Error["MAE"].append(mae)
Error["RMSE"].append(rmse)
Error["OutputC"].append(output_channel)
Error["Sample Size"].append(sample_size)

#save csv
pd.DataFrame(Error).to_csv("./Results/LSTM2_finalv4.csv")
