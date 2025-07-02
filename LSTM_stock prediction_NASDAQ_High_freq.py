# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

data_path = 'C://Users//JunmingLao//Desktop//LSTM_Stock_prediction//'
    
class LSTM_stock_predict:
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.X_scaler = MinMaxScaler()
        self.Y_scaler = MinMaxScaler()
        
    def Model_train(self, data_path, freq):
        self.df = pd.read_csv(data_path + 'NDX_{}min_sample.csv'.format(freq))
        
        train_size = int(len(self.df)*0.85)
        self.df_train = self.df.iloc[:train_size]
        self.df_test = self.df.iloc[train_size:]
        
        self.X_train = self.df_train[['timestamp','open','high','low']].values #The 0 of the timestapm refers to the first min of the data
        self.Y_train = self.df_train[['close']].values
        
        self.X_test = self.df_test[['timestamp','open','high','low']].values 
        self.Y_test = self.df_test[['close']].values
        
        self.X_train_scaled = self.X_scaler.fit_transform(self.X_train)
        self.Y_train_scaled = self.Y_scaler.fit_transform(self.Y_train)
        self.X_test_scaled = self.X_scaler.transform(self.X_test)
        self.Y_test_scaled = self.Y_scaler.transform(self.Y_test)
        self.X_train_scaled = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1])
        self.X_test_scaled = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1])
        #print(X_train_scaled.shape)
        
        self.model = Sequential([
            LSTM(256, return_sequences=True, input_shape=(1, self.X_train.shape[1])),
            LSTM(128, return_sequences=False),
            Dense(64),
            Dense(1)
            ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-9,
            verbose=1
            )
        
        self.train = self.model.fit(
            self.X_train_scaled,
            self.Y_train_scaled,
            epochs=50,
            batch_size=2,
            validation_split=0.15,
            callbacks=[lr],
            verbose=1
            )
        
        test_loss, test_mse = self.model.evaluate(self.X_test_scaled, self.Y_test_scaled, verbose=0)
        print(f'\nTest loss: {test_loss:.4f}')
        
        self.model.save('LSTM_stock prediciton_NASDAQ_High_freq.h5')
        
        self.Y_predict = self.model.predict(self.X_test_scaled)
        self.Y_predict = self.Y_scaler.inverse_transform(self.Y_predict)
        
        fig, axs = plt.subplots(2, 2, figsize=(12,10))
        
        axs[0, 0].plot(self.train.history['loss'],label='Training loss')
        axs[0, 0].plot(self.train.history['val_loss'],label='Validation loss')
        axs[0, 0].set_title('Model Loss During Training')
        axs[0, 0].set_xlabel('Epoches')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        axs[0, 1].scatter(self.Y_test, self.Y_predict, alpha=0.5)
        min_val = min(np.min(self.Y_test), np.min(self.Y_predict))
        max_val = max(np.max(self.Y_test), np.max(self.Y_predict))
        axs[0, 1].plot([min_val, max_val], [min_val, max_val], 'r-', label='y=x')
        axs[0, 1].plot([min_val, max_val], [min_val+0.03*(max_val-min_val), max_val+0.03*(max_val-min_val)], 'k--', label='+3%')
        axs[0, 1].plot([min_val+0.03*(max_val-min_val), max_val], [min_val, max_val-0.03*(max_val-min_val)], 'k--', label='-3%')
        axs[0, 1].set_xlim(min_val, max_val)
        axs[0, 1].set_ylim(min_val, max_val)
        axs[0, 1].set_title('Actual vs. Prediction')
        axs[0, 1].set_xlabel('Actual values')
        axs[0, 1].set_ylabel('Predicted values')
        axs[0, 1].legend()
        
        axs[1, 0].plot(self.df['timestamp'], self.df['close'],label='Raw data')
        axs[1, 0].plot(self.df['timestamp'][-(len(self.Y_predict)+1):-1], self.Y_predict[:,0],label='Predicted data')
        axs[1, 0].set_xlabel('Duration (day)')
        axs[1, 0].set_ylabel('Close Price ($)')
        axs[1, 0].set_title('Raw data vs. Prediction')
        axs[1, 0].legend()
        
        axs[1, 1].plot(self.df['timestamp'][-(len(self.Y_predict)+1):-1], self.df['close'][-(len(self.Y_predict)+1):-1],label='Raw data')
        axs[1, 1].plot(self.df['timestamp'][-(len(self.Y_predict)+1):-1], self.Y_predict[:,0],label='Predicted data')
        axs[1, 1].set_xlabel('Duration (day)')
        axs[1, 1].set_ylabel('Close Price ($)')
        axs[1, 1].set_title('Zoom in Raw data vs. Prediction')
        axs[1, 1].legend()
        
        plt.show()
        
                
    def Model_load(self, data_path, data):
        data_scaled = self.X_scaler.transform(data)
        data_scaled = data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1])
        
        self.model = load_model(data_path+'LSTM_stock prediciton_NASDAQ_High_freq.h5')
        
        prediction_scaled = self.model.predict(data_scaled)
        prediction = self.Y_scaler.inverse_transform(prediction_scaled)
        
        return self.model, prediction
    
if __name__ == '__main__':
    predictor = LSTM_stock_predict()
    predictor.Model_train(data_path, freq=1) #freq option: 1, 5 and 30