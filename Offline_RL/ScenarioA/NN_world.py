import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

import tensorflow as tf


from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


from tensorflow.keras import Sequential,callbacks,layers
from tensorflow.keras.layers import Dense, LSTM,Dropout,GRU,Bidirectional
from tensorflow.keras import regularizers

class NN_world:

    def create_model_bilstm_t0(self,units,X_train):
        model = Sequential()
        model.add(Bidirectional(LSTM(units = units,return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Bidirectional(LSTM(units = units)))
        model.add(Dense(1))
        #Compile model
        model.compile(loss='mse', optimizer='adam')
        return model

    # Create BiLSTM model
    def create_model_bilstm(self,units,X_train):
        model = Sequential()
        model.add(Bidirectional(LSTM(units = units,return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units = units)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        #Compile model
        model.compile(loss='mse', optimizer='adam')
        return model

    def prediction_norm(self,model,data):
        prediction_norm = model.predict(data)
        return prediction_norm

    def fit_model(self,model,X_train, y_train):
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 20)
        history = model.fit(X_train, y_train, epochs = 100,
                            validation_split = 0.2, batch_size = 16,
                            shuffle = False, callbacks = [early_stop])
        return history

    def step_data_model(self,X,y,prediction_bilstm_norm,model_number):
        X_new=[]
        y_new=[]
        for i in range(len(X)-model_number):
            V=X[i+model_number].copy()
            V[self.TIME_STEPS-1][0]=prediction_bilstm_norm[i]
            X_new.append(V)
            lable=y[model_number+i]
            y_new.append(lable)
        return np.array(X_new), np.array(y_new)

    def step_data_model_t2(self,X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,model_number):
        X_new=[]
        y_new=[]
        for i in range(len(prediction_bilstm_norm_t1)-model_number):
            V=X[i+model_number].copy()
            V[self.TIME_STEPS-2][0]=prediction_bilstm_norm[i]
            V[self.TIME_STEPS-1][0]=prediction_bilstm_norm_t1[i]
            X_new.append(V)
            lable=y[model_number+i]
            y_new.append(lable)
        return np.array(X_new), np.array(y_new)

    def step_data_model_t3(self,X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,model_number):
        X_new=[]
        y_new=[]
        for i in range(len(prediction_bilstm_norm_t1)-model_number):
            V=X[i+model_number].copy()
            V[self.TIME_STEPS-3][0]=prediction_bilstm_norm[i]
            V[self.TIME_STEPS-2][0]=prediction_bilstm_norm_t1[i]
            V[self.TIME_STEPS-1][0]=prediction_bilstm_norm_t2[i]
            X_new.append(V)
            lable=y[model_number+i]
            y_new.append(lable)
        return np.array(X_new), np.array(y_new)

    def step_data_model_t4(self,X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,model_number):
        X_new=[]
        y_new=[]
        for i in range(len(prediction_bilstm_norm_t1)-model_number):
            V=X[i+model_number].copy()
            V[self.TIME_STEPS-4][0]=prediction_bilstm_norm[i]
            V[self.TIME_STEPS-3][0]=prediction_bilstm_norm_t1[i]
            V[self.TIME_STEPS-2][0]=prediction_bilstm_norm_t2[i]
            V[self.TIME_STEPS-1][0]=prediction_bilstm_norm_t3[i]
            X_new.append(V)
            lable=y[model_number+i]
            y_new.append(lable)
        return np.array(X_new), np.array(y_new)

    def step_data_model_t5(self,X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,model_number):
        X_new=[]
        y_new=[]
        for i in range(len(prediction_bilstm_norm_t1)-model_number):
            V=X[i+model_number].copy()
            V[self.TIME_STEPS-5][0]=prediction_bilstm_norm[i]
            V[self.TIME_STEPS-4][0]=prediction_bilstm_norm_t1[i]
            V[self.TIME_STEPS-3][0]=prediction_bilstm_norm_t2[i]
            V[self.TIME_STEPS-2][0]=prediction_bilstm_norm_t3[i]
            V[self.TIME_STEPS-1][0]=prediction_bilstm_norm_t4[i]
            X_new.append(V)
            lable=y[model_number+i]
            y_new.append(lable)
        return np.array(X_new), np.array(y_new)

    def step_data_model_t6(self,X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,model_number):
        X_new=[]
        y_new=[]
        for i in range(len(prediction_bilstm_norm_t1)-model_number):
            V=X[i+model_number].copy()
            V[self.TIME_STEPS-6][0]=prediction_bilstm_norm[i]
            V[self.TIME_STEPS-5][0]=prediction_bilstm_norm_t1[i]
            V[self.TIME_STEPS-4][0]=prediction_bilstm_norm_t2[i]
            V[self.TIME_STEPS-3][0]=prediction_bilstm_norm_t3[i]
            V[self.TIME_STEPS-2][0]=prediction_bilstm_norm_t4[i]
            V[self.TIME_STEPS-1][0]=prediction_bilstm_norm_t5[i]
            X_new.append(V)
            lable=y[model_number+i]
            y_new.append(lable)
        return np.array(X_new), np.array(y_new)



    def build_world(self,X_train, y_train,timestep):      

      self.TIME_STEPS=timestep
      model_bilstm = self.create_model_bilstm_t0(64,X_train)
      model_bilstm_t1 = self.create_model_bilstm_t0(64,X_train)
      model_bilstm_t2 = self.create_model_bilstm_t0(64,X_train)
      model_bilstm_t3 = self.create_model_bilstm(64,X_train)
      model_bilstm_t4 = self.create_model_bilstm(64,X_train)
      model_bilstm_t5 = self.create_model_bilstm(64,X_train)
      model_bilstm_t6 = self.create_model_bilstm(64,X_train)
      model_bilstm_t7 = self.create_model_bilstm(64,X_train)
      model_bilstm_t8 = self.create_model_bilstm(64,X_train)
      model_bilstm_t9 = self.create_model_bilstm(64,X_train)
      model_bilstm_t10 = self.create_model_bilstm(64,X_train)
      model_bilstm_t11 = self.create_model_bilstm(64,X_train)
      model_bilstm_t12 = self.create_model_bilstm(64,X_train)
      model_bilstm_t13 = self.create_model_bilstm(64,X_train)
      model_bilstm_t14 = self.create_model_bilstm(64,X_train)
      model_bilstm_t15 = self.create_model_bilstm(64,X_train)

      history_bilstm = self.fit_model(model_bilstm,X_train, y_train)


      print('-----1-----')
      prediction_bilstm_norm = self.prediction_norm(model_bilstm,X_train)

      X_new_t1,y_new_t1=self.step_data_model(X_train,y_train,prediction_bilstm_norm,1)

      history_bilstm_t1 = self.fit_model(model_bilstm_t1,X_new_t1,y_new_t1)

      print('-----2-----')
      prediction_bilstm_norm_t1 = self.prediction_norm(model_bilstm_t1,X_new_t1)

      X_new_t2,y_new_t2=self.step_data_model_t2(X_train,y_train,prediction_bilstm_norm,prediction_bilstm_norm_t1,2)

      history_bilstm_t2 = self.fit_model(model_bilstm_t2,X_new_t2,y_new_t2)
      print('-----3-----')
      prediction_bilstm_norm_t2 = self.prediction_norm(model_bilstm_t2,X_new_t2)

      X_new_t3,y_new_t3=self.step_data_model_t3(X_train,y_train,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,3)

      history_bilstm_t3 = self.fit_model(model_bilstm_t3,X_new_t3,y_new_t3)
      print('-----4-----')
      prediction_bilstm_norm_t3 = self.prediction_norm(model_bilstm_t3,X_new_t3)

      X_new_t4,y_new_t4=self.step_data_model_t4(X_train,y_train,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,4)

      history_bilstm_t4 = self.fit_model(model_bilstm_t4,X_new_t4,y_new_t4)
      print('----5------')
      prediction_bilstm_norm_t4 = self.prediction_norm(model_bilstm_t4,X_new_t4)

      X_new_t5,y_new_t5=self.step_data_model_t5(X_train,y_train,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,5)

      history_bilstm_t5 = self.fit_model(model_bilstm_t5,X_new_t5,y_new_t5)

      print('-----6-----')
      prediction_bilstm_norm_t5 = self.prediction_norm(model_bilstm_t5,X_new_t5)

      X_new_t6,y_new_t6=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,6)

      history_bilstm_t6 = self.fit_model(model_bilstm_t6,X_new_t6,y_new_t6)

      print('-----7-----')
      prediction_bilstm_norm_t6 = self.prediction_norm(model_bilstm_t6,X_new_t6)

      X_new_t7,y_new_t7=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,prediction_bilstm_norm_t6,7)

      history_bilstm_t7 = self.fit_model(model_bilstm_t7,X_new_t7,y_new_t7)

      print('----8------last')
      prediction_bilstm_norm_t7 = self.prediction_norm(model_bilstm_t7,X_new_t7)

      X_new_t8,y_new_t8=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,prediction_bilstm_norm_t6,prediction_bilstm_norm_t7,8)

      history_bilstm_t8 = self.fit_model(model_bilstm_t8,X_new_t8,y_new_t8)


      print('----9------')
      prediction_bilstm_norm_t8 = self.prediction_norm(model_bilstm_t8,X_new_t8)

      X_new_t9,y_new_t9=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,prediction_bilstm_norm_t6,prediction_bilstm_norm_t7,prediction_bilstm_norm_t8,9)

      history_bilstm_t9 = self.fit_model(model_bilstm_t9,X_new_t9,y_new_t9)

      print('----10------')
      prediction_bilstm_norm_t9 = self.prediction_norm(model_bilstm_t9,X_new_t9)

      X_new_t10,y_new_t10=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,prediction_bilstm_norm_t6,prediction_bilstm_norm_t7,prediction_bilstm_norm_t8,prediction_bilstm_norm_t9,10)

      history_bilstm_t10 = self.fit_model(model_bilstm_t10,X_new_t10,y_new_t10)

      print('----11------')
      prediction_bilstm_norm_t10 = self.prediction_norm(model_bilstm_t10,X_new_t10)

      X_new_t11,y_new_t11=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t5,prediction_bilstm_norm_t6,prediction_bilstm_norm_t7,prediction_bilstm_norm_t8,prediction_bilstm_norm_t9,prediction_bilstm_norm_t10,11)

      history_bilstm_t11 = self.fit_model(model_bilstm_t11,X_new_t11,y_new_t11)

      print('----12------')
      prediction_bilstm_norm_t11 = self.prediction_norm(model_bilstm_t11,X_new_t11)

      X_new_t12,y_new_t12=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t6,prediction_bilstm_norm_t7,prediction_bilstm_norm_t8,prediction_bilstm_norm_t9,prediction_bilstm_norm_t10,prediction_bilstm_norm_t11,12)

      history_bilstm_t12 = self.fit_model(model_bilstm_t12,X_new_t12,y_new_t12)

      print('----13------')
      prediction_bilstm_norm_t12 = self.prediction_norm(model_bilstm_t12,X_new_t12)

      X_new_t13,y_new_t13=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t7,prediction_bilstm_norm_t8,prediction_bilstm_norm_t9,prediction_bilstm_norm_t10,prediction_bilstm_norm_t11,prediction_bilstm_norm_t12,13)

      history_bilstm_t13 = self.fit_model(model_bilstm_t13,X_new_t13,y_new_t13)

      print('----14------')
      prediction_bilstm_norm_t13 = self.prediction_norm(model_bilstm_t13,X_new_t13)

      X_new_t14,y_new_t14=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t8,prediction_bilstm_norm_t9,prediction_bilstm_norm_t10,prediction_bilstm_norm_t11,prediction_bilstm_norm_t12,prediction_bilstm_norm_t13,14)

      history_bilstm_t14 = self.fit_model(model_bilstm_t14,X_new_t14,y_new_t14)


      print('----15------last')
      prediction_bilstm_norm_t14 = self.prediction_norm(model_bilstm_t14,X_new_t14)

      X_new_t15,y_new_t15=self.step_data_model_t6(X_train,y_train,prediction_bilstm_norm_t9,prediction_bilstm_norm_t10,prediction_bilstm_norm_t11,prediction_bilstm_norm_t12,prediction_bilstm_norm_t13,prediction_bilstm_norm_t14,15)

      history_bilstm_t15 = self.fit_model(model_bilstm_t15,X_new_t15,y_new_t15)


      model_bilstm.save("modelt1")
      model_bilstm_t1.save("modelt2")
      model_bilstm_t2.save("modelt3")
      model_bilstm_t3.save("modelt4")
      model_bilstm_t4.save("modelt5")
      model_bilstm_t5.save("modelt6")
      model_bilstm_t6.save("modelt7")
      model_bilstm_t7.save("modelt8")
      model_bilstm_t8.save("modelt9")
      model_bilstm_t9.save("modelt10")
      model_bilstm_t10.save("modelt11")
      model_bilstm_t11.save("modelt12")
      model_bilstm_t12.save("modelt13")
      model_bilstm_t13.save("modelt14")
      model_bilstm_t14.save("modelt15")
      model_bilstm_t15.save("modelt16")

      return model_bilstm,model_bilstm_t1,model_bilstm_t2,model_bilstm_t3,model_bilstm_t4,model_bilstm_t5,model_bilstm_t6,model_bilstm_t7,model_bilstm_t8,model_bilstm_t9,model_bilstm_t10,model_bilstm_t11,model_bilstm_t12,model_bilstm_t13,model_bilstm_t14,model_bilstm_t15