# -*- coding: utf-8 -*-
"""
Created on Sep 07 17:05:12 2021

@author: Yurim
"""
#https://github.com/minji-OH/Analysis_in_Python/blob/master/Comparing%20GRU%20and%20LSTM%20with%20Stock%20data.ipynb
#https://data-analysis-expertise.tistory.com/67

import os
import numpy as np
import pandas as pd 
os.chdir(r'E:\2021\study\1_도체\2_가격 예측 관련\2차년도 보고서 작성\2022예측\도매가격')


####### 데이터 전처리
all_data = pd.read_csv('220221_22도매가격예측.csv')
all_data.head(10)

del all_data['date']
del all_data['year']
#del all_data['month']
all_data.head(10)

all_data.dtypes


# scale the data
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range=(0,1))
data_scaled = sc.fit_transform(all_data)

data_scaled[:,1] = all_data['month'].values
data_scaled[:,2] = all_data['요일'].values


for epochs_num in [500, 1000]:
    for batch_size_num in [32, 64, 128]:

        def LSTM_model(X_train, y_train, X_test, for_periods):
            # create a model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
            from tensorflow.keras.optimizers import SGD
            
            # The LSTM architecture
            my_LSTM_model = Sequential()
            my_LSTM_model.add(LSTM(units = 100, 
                                   return_sequences = True, 
                                   input_shape = (X_train.shape[1],X_train.shape[2]), 
                                   activation = 'tanh'))
            #my_LSTM_model.add(Dropout(0.01)) #추가
            #my_LSTM_model.add(LSTM(units = 50, activation = 'tanh', return_sequences = True))
            #my_LSTM_model.add(Dropout(0.2)) #추가
            my_LSTM_model.add(LSTM(units = 100, activation = 'tanh'))
            #my_LSTM_model.add(Dropout(0.01)) #추가
            my_LSTM_model.add(Dense(units=for_periods))
            my_LSTM_model.summary()
            # Compiling 
            my_LSTM_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            #my_LSTM_model.compile(optimizer = SGD(lr = 0.01, decay = 1e-7, momentum = 0.9, nesterov = False), loss = 'mean_squared_error')
            
            # Fitting to the training set 
            my_LSTM_model.fit(X_train, y_train, epochs = epochs_num, batch_size = batch_size_num, verbose = 1)
            
            LSTM_prediction = my_LSTM_model.predict(X_test)
            #LSTM_prediction = sc.inverse_transform(LSTM_prediction)
            
            return my_LSTM_model, LSTM_prediction
        
        
        def actual_pred_plot(preds, y_test):
            """
            Plot the actual vs predition
            """
            actual_pred = pd.DataFrame(columns = ['y', 'prediction'])
            actual_pred['y'] = y_test[:,0,0]
            actual_pred['prediction'] = preds[:,0]
            
            from tensorflow.keras.metrics import MeanSquaredError 
            m = MeanSquaredError()
            m.update_state(np.array(actual_pred['y']), np.array(actual_pred['prediction']))
            
            return (m.result().numpy(), actual_pred.plot())
        
        
        def confirm_result(y_test, y_pred):
            MAE = mean_absolute_error(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
            MSLE = mean_squared_log_error(y_test, y_pred)
            RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
            R2 = r2_score(y_test, y_pred)
            
            pd.options.display.float_format = '{:.5f}'.format
            Result = pd.DataFrame(data=[MAE, MSE, RMSE, RMSLE, R2],
                                 index = ['MAE','MSE', 'RMSE', 'RMSLE', 'R2'],
                                 columns=['Results'])
            return Result
        
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
        
        
        ####### main
        X_data = [] 
        y_data = [] 
        time_steps = 190
        for_periods = 102
        
        
        for i in range(time_steps, len(data_scaled)-for_periods):
            X_data.append(data_scaled[i-time_steps:i, ])
            #y_data.append(data_scaled[i:i+for_periods, 0])
            y_data.append(data_scaled[i:i+for_periods, 0].reshape(for_periods,-1))  #여러개를 예측할때
            
        X_data, y_data = np.array(X_data), np.array(y_data)
        
        #y_data = y_data.reshape(-1,1)
        y_data.shape
        X_data.shape
            
        
        #ts_test_len = len(data_scaled) - ts_train_len
        #ts_len = len(X_data) - 1 - time_steps
        ts_len = len(X_data) - for_periods
        X_train = X_data[:ts_len,:,:]
        y_train = y_data[:ts_len,:] 
        X_test = X_data[-1:,:,:]
        y_test = y_data[-1:,:] 
        
        
        my_LSTM_model, LSTM_prediction = LSTM_model(X_train, y_train, X_test, for_periods)
        actual_pred_plot(LSTM_prediction, y_test)
        print(confirm_result(y_test.reshape(-1,for_periods)[:,], LSTM_prediction[:,]))
        
        
        
        
        #원래 값으로 되돌리기
        LSTM_prediction_all = my_LSTM_model.predict(X_test)
           
        data_scaled_y = sc.fit_transform(pd.DataFrame(all_data.iloc[:,0]))
        LSTM_prediction_sc = sc.inverse_transform(LSTM_prediction_all) 
        
        
        
        
        ######## prediction save
        LSTM_prediction_sc = pd.DataFrame(LSTM_prediction_sc)
        
        pred_result = LSTM_prediction_sc.transpose()
        pred_result.columns = ['pred']
        
        pred_result.to_csv("0111_"+str(epochs_num)+"_"+str(batch_size_num)+".csv", header=True)
        my_LSTM_model.save("0111_"+str(epochs_num)+"_"+str(batch_size_num)+".h5")
        

