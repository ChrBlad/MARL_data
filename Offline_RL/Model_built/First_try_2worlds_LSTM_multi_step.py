import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential,callbacks,layers
from tensorflow.keras.layers import Dense, LSTM,Dropout,GRU,Bidirectional
from tensorflow.keras import regularizers
print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from numpy.random import seed 
import random

random_seed=2017
seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)

# N_VALIDATION = int(1e3)
# N_TRAIN = int(1e4)
# BUFFER_SIZE = int(1e4)
# BATCH_SIZE = 32
# STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE


data_zone1=[]
data_actions=[]
data_out=[]
with open('output_zone1.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone1.append(row)

csvFile.close()

with open('output_actions.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_actions.append(row)

csvFile.close()


with open('output_out.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_out.append(row)

csvFile.close()


df_actions = pd.DataFrame(data_actions)
df_actions.columns = ["Valve1", "Supplytemp","Reward","Hardcon","price_reward","sum_mix_reward","change_in_Supply_temp"]

df_zone1 = pd.DataFrame(data_zone1)
df_zone1.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv','tod',"time"]

df_out = pd.DataFrame(data_out)
df_out.columns = ['RoomTemperature1','RoomTemperature2','RoomTemperature3','RoomTemperature4','Valveout1','Valveout2','Valveout3','Valveout4','Power1','Power2','Power3','Power4','Tamb1','Sun1','Tambforecast1','Tambforecast2','Tambforecast3','Sunforecast1','Sunforecast2','Sunforecast3','Powerwater1','Powerwater2','Powerwater3','Powerwater4','Treturn1','Treturn2','Treturn3','Treturn4','flow1','flow2','flow3','flow4']


#remove Hardcon and HCV'Hardcon','Hcv','Hardcon': 'Hardcon_t', 'Hcv': 'Hcv_t',

data = pd.concat([df_actions, df_zone1,df_out], axis=1, sort=False)


#remove Hardcon and HCV'Hardcon','Hcv','Hardcon': 'Hardcon_t', 'Hcv': 'Hcv_t',
dataset_t = data[['Troom', 'Tamb', 'Valveout1','Supplytemp','tod','Sun1','Sunforecast1','Tambforecast1']].copy()
dataset_t = dataset_t.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t','Valveout1':'Valve_t' , 'Supplytemp': 'Supplytemp_t','tod':'tod_t','Sun1':'Sun1_t','Sunforecast1':'Sunforecast1_t','Tambforecast1':'Tambforecast1_t'}, inplace = False)

lables=data[['Troom']].copy()
lables.reset_index(inplace=True, drop=True)


dataset = pd.concat([dataset_t], axis=1, join='inner')#,dataset_tm1, dataset_tm2,dataset_tm3,dataset_tm4,dataset_tm5,dataset_tm6, dataset_tm7,dataset_tm8], axis=1, join='inner')#,dataset_tm5,dataset_tm6

len_lables=len(lables)

##dataset.drop([len_lables], inplace=True)#,len_dataset+1,len_dataset+2,len_dataset+3,len_dataset+4,len_dataset+5,len_dataset+6,len_dataset+7,len_dataset+8#,len_dataset+5,len_dataset+6


dataset = pd.concat([lables,dataset_t], axis=1, join='inner')#,dataset_tm1, dataset_tm2,dataset_tm3,dataset_tm4,dataset_tm5, dataset_tm6,dataset_tm7,dataset_tm8], axis=1, join='inner')#,dataset_tm5,dataset_tm6


dataset=dataset.astype(float)
#dataset=dataset.round(decimals=4)


train_size = int(108*300)
print('lenght of traing data[days]', train_size/108)
#test_size = int(len(dataset)*0.30)
train_dataset, test_dataset = dataset.iloc[:train_size],dataset.iloc[train_size:]
#Nothing, test_dataset = test_dataset.iloc[:test_size],test_dataset.iloc[test_size:]
cold_set=train_dataset.copy()
cold_set_2=train_dataset.copy()
cold_set.drop(cold_set.index[0:4000], inplace=True)
cold_set.drop(cold_set.index[1300:], inplace=True)
cold_set.reset_index(inplace=True, drop=True)
cold_set_2.drop(cold_set_2.index[0:2100], inplace=True)
cold_set_2.drop(cold_set_2.index[500:], inplace=True)
cold_set_2.reset_index(inplace=True, drop=True)







print(train_dataset.head(30))
print(train_dataset.tail())
train_dataset_2=train_dataset.copy()
train_dataset_3=train_dataset.copy()
train_dataset_4=train_dataset.copy()
train_dataset_5=train_dataset.copy()
train_dataset = pd.concat([cold_set_2,cold_set,train_dataset,cold_set_2,cold_set,train_dataset_2,cold_set_2,cold_set,train_dataset_3,cold_set_2,cold_set,train_dataset_4,cold_set_2,cold_set,train_dataset_5,cold_set_2,cold_set], axis=0, join='inner')
train_dataset.reset_index(inplace=True, drop=True)
# print(train_dataset.tail())
# avg_amb_temp_list=[]
# #day_list=[]
# for i in range(len(train_dataset)//108):
#     avg_amb_temp=sum(train_dataset['Tamb_t'][i*108:i*108+108])/108
#     avg_amb_temp_list.append(avg_amb_temp)
#     #day_list.append(i)


# day_to_include=[]
# bookkeeping=[]
# Tamb_to_keep=[]
# days_to_drop=[]

# for i in range(len(avg_amb_temp_list)):
#     for n in range(30):
#         if  262+n <= avg_amb_temp_list[i] <= 262+n+1:
#             bookkeeping.append(n)
#             if bookkeeping.count(n) < 11:
#                 include=1
#             else:
#                 include=0
#     if include==0:
#         days_to_drop.append(i)
#     if include==1:
#         day_to_include.append(i)
#         Tamb_to_keep.append(avg_amb_temp_list[i])


# drop_list=[]
# for i in range(len(days_to_drop)):
#     for n in range(108):
#         a=days_to_drop[i]*108+n
#         drop_list.append(a)
        
#print(drop_list)
# print(len(train_dataset))

# train_dataset.drop(train_dataset.index[drop_list], inplace=True)
# train_dataset.reset_index(inplace=True, drop=True)

# train_dataset_2=train_dataset.copy()
# train_dataset_3=train_dataset.copy()
# train_dataset_4=train_dataset.copy()
# train_dataset_5=train_dataset.copy()
# train_dataset = pd.concat([train_dataset,train_dataset_2,train_dataset_3], axis=0, join='inner')
# train_dataset.reset_index(inplace=True, drop=True)

# print(len(train_dataset))
# print(len(day_to_include))
# print(len(days_to_drop))
# print(len(avg_amb_temp_list))
# print(train_dataset.tail())
#print(avg_amb_temp_list)

#print(train_dataset)

f1 = plt.figure(0)


x=train_dataset['Tamb_t']
plt.hist(x, bins = 20)

df_zone1=df_zone1.astype(float)
df_out=df_out.astype(float)
f1 = plt.figure(1)
#plt.style.use('ggplot')
#plt.plot(df_zone1['time'],df_out['Sun1'],label='action', alpha=0.7)
plt.plot(cold_set['Troom_t'],label='action', alpha=0.7)
plt.plot(cold_set['Tamb_t'],label='action', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Tamb'],label='Hardconstraint', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

f2 = plt.figure(2)
#plt.style.use('ggplot')
#plt.plot(df_zone1['time'],df_out['Sun1'],label='action', alpha=0.7)
plt.plot(cold_set_2['Troom_t'],label='action', alpha=0.7)
plt.plot(cold_set_2['Tamb_t'],label='action', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Tamb'],label='Hardconstraint', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

f3 = plt.figure(3)
#plt.style.use('ggplot')
#plt.plot(df_zone1['time'],df_out['Sun1'],label='action', alpha=0.7)
plt.plot(train_dataset['Troom_t'],label='action', alpha=0.7)
plt.plot(train_dataset['Tamb_t'],label='action', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Tamb'],label='Hardconstraint', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

plt.show()








# Split train data to X and y
X_train = train_dataset.drop('Troom', axis = 1)
y_train = train_dataset.loc[:,['Troom']]
# Split test data to X and y
X_test = test_dataset.drop('Troom', axis = 1)
y_test = test_dataset.loc[:,['Troom']]


print(X_train.head(20))
#print(y_train.head(10))

# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))
# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train)
output_scaler = scaler_y.fit(y_train)
# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)
# Apply the scaler to test data
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)

X_train_real=X_train
X_test_real=X_test.copy()
#print(train_x_norm[0:1,:])
#print('lable----',train_y_norm[1])
#print(train_dataset.head())



#
def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        #print('-------',i+time_steps)
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
        #print(Xs[i])
        #print(ys[i])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 6
steps=5
number_of_runs=15

# for m in range(number_of_runs):
#   n=m*50
#   for i in range(steps): 
#     i=i+TIME_STEPS+n
#     X_test_real.iloc[i]['Troom_t']=0

X_train, y_train = create_dataset(train_x_norm, train_y_norm, 
                                  TIME_STEPS)

X_test, y_test = create_dataset(test_x_norm, test_y_norm,   
                                TIME_STEPS)

# print('X_train.shape: ', X_train.shape)
# print('y_train.shape: ', y_train.shape)
# print('X_test.shape: ', X_test.shape)
# print('y_test.shape: ', y_test.shape)
# a=X_train[1]
# print(X_train[1])
# print(a[TIME_STEPS-1][0])
# print('lable',y_train[0])
# print('-------')
# print(X_train[1])
# print('lable',y_train[1])
# print('-------')
# print(X_train[2])
# print('lable',y_train[2])
# print('-------')
print(X_train[1])
print('lable-test_label',y_train[0])
num_of_model=30

#print(X_test[:20][:])

def create_model_bilstm_t0(units):
    model = Sequential()
    model.add(Bidirectional(LSTM(units = units,return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model

# Create BiLSTM model
def create_model_bilstm(units):
    model = Sequential()
    model.add(Bidirectional(LSTM(units = units,return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units = units,return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dropout(0.2))
    model.add(Dense(units = units))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    # 0.0005,
    # decay_steps=STEPS_PER_EPOCH*100,
    # decay_rate=2,
    # staircase=True)
    # optimizer = tf.keras.optimizers.Adam(lr_schedule)
    # model.compile(loss='mse',
    #               optimizer=optimizer,
    #               metrics=['mae', 'mse'])
    return model


# Create LSTM or GRU model
# def create_model(units, m):
#     model = Sequential()
#     model.add(m (units = units, return_sequences = True,input_shape = [X_train.shape[1], X_train.shape[2]]))
#     model.add(Dropout(0.3))
#     model.add(m (units = units, return_sequences = True))
#     model.add(Dropout(0.3))
#     model.add(m (units = units))
#     model.add(Dropout(0.3))
#     model.add(Dense(units = 1))
#     #Compile model
#     model.compile(loss='mse', optimizer='adam')
#     # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     # 0.0005,
#     # decay_steps=STEPS_PER_EPOCH*1000,
#     # decay_rate=2,
#     # staircase=True)
#     # optimizer = tf.keras.optimizers.Adam(lr_schedule)

#     # model.compile(loss='mse',
#     #               optimizer=optimizer,
#     #               metrics=['mae', 'mse'])
#     return model

model_list=[]
# BiLSTM
model_bilstm = create_model_bilstm_t0(64)
model_bilstm_t1 = create_model_bilstm_t0(64)
model_bilstm_t2 = create_model_bilstm_t0(64)
model_bilstm_t3 = create_model_bilstm_t0(64)
model_bilstm_t4 = create_model_bilstm_t0(64)
model_bilstm_t5 = create_model_bilstm_t0(64)
model_list.append(model_bilstm)
model_list.append(model_bilstm_t1)
model_list.append(model_bilstm_t2)
model_list.append(model_bilstm_t3)
model_list.append(model_bilstm_t4)
model_list.append(model_bilstm_t5)
for i in range(num_of_model-6):
    model = create_model_bilstm_t0(64)
    model_list.append(model)

# GRU and LSTM
# # model_bilstm = create_model(64 , GRU)
# # model_bilstm_t1 = create_model(64, GRU)
# # model_bilstm_t2 = create_model(64, GRU)
# # model_bilstm_t3 = create_model(64, GRU)
# # model_bilstm_t4 = create_model(64, GRU)
# # model_bilstm_t5 = create_model(64, GRU)
# # model_bilstm_t6 = create_model(64, GRU)
# # model_bilstm_t7 = create_model(64, GRU)
# # model_bilstm_t8 = create_model(64, GRU)
#model_lstm = create_model(64, LSTM)


def fit_model(model,X_train, y_train):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 20)
    history = model.fit(X_train, y_train, epochs = 100,
                        validation_split = 0.2, batch_size = 16,
                        shuffle = False, callbacks = [early_stop])
    return history





# Plot train loss and validation loss
def plot_loss (history):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

# Make prediction

def prediction_norm(model,data):
    prediction_norm = model.predict(data)
    #prediction = scaler_y.inverse_transform(prediction)
    return prediction_norm

def prediction(model,data):
    prediction = model.predict(data)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction

def evaluate_prediction(predictions, actual, model_name,modelnumber):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print('Mean Absolute Error[Kelvin] for model t:',modelnumber, mae)

def step_data_model(X,y,prediction_bilstm_norm,model_number):
    X_new=[]
    y_new=[]
    for i in range(len(X)-model_number):
        V=X[i+model_number].copy()
        V[TIME_STEPS-1][0]=prediction_bilstm_norm[i]
        X_new.append(V)
        lable=y[model_number+i]
        y_new.append(lable)
    return np.array(X_new), np.array(y_new)

def step_data_model_t2(X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,model_number):
    X_new=[]
    y_new=[]
    for i in range(len(prediction_bilstm_norm_t1)-model_number):
        V=X[i+model_number].copy()
        V[TIME_STEPS-2][0]=prediction_bilstm_norm[i]
        V[TIME_STEPS-1][0]=prediction_bilstm_norm_t1[i]
        X_new.append(V)
        lable=y[model_number+i]
        y_new.append(lable)
    return np.array(X_new), np.array(y_new)

def step_data_model_t3(X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,model_number):
    X_new=[]
    y_new=[]
    for i in range(len(prediction_bilstm_norm_t1)-model_number):
        V=X[i+model_number].copy()
        V[TIME_STEPS-3][0]=prediction_bilstm_norm[i]
        V[TIME_STEPS-2][0]=prediction_bilstm_norm_t1[i]
        V[TIME_STEPS-1][0]=prediction_bilstm_norm_t2[i]
        X_new.append(V)
        lable=y[model_number+i]
        y_new.append(lable)
    return np.array(X_new), np.array(y_new)

def step_data_model_t4(X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,model_number):
    X_new=[]
    y_new=[]
    for i in range(len(prediction_bilstm_norm_t1)-model_number):
        V=X[i+model_number].copy()
        V[TIME_STEPS-4][0]=prediction_bilstm_norm[i]
        V[TIME_STEPS-3][0]=prediction_bilstm_norm_t1[i]
        V[TIME_STEPS-2][0]=prediction_bilstm_norm_t2[i]
        V[TIME_STEPS-1][0]=prediction_bilstm_norm_t3[i]
        X_new.append(V)
        lable=y[model_number+i]
        y_new.append(lable)
    return np.array(X_new), np.array(y_new)

def step_data_model_t5(X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,model_number):
    X_new=[]
    y_new=[]
    for i in range(len(prediction_bilstm_norm_t1)-model_number):
        V=X[i+model_number].copy()
        V[TIME_STEPS-5][0]=prediction_bilstm_norm[i]
        V[TIME_STEPS-4][0]=prediction_bilstm_norm_t1[i]
        V[TIME_STEPS-3][0]=prediction_bilstm_norm_t2[i]
        V[TIME_STEPS-2][0]=prediction_bilstm_norm_t3[i]
        V[TIME_STEPS-1][0]=prediction_bilstm_norm_t4[i]
        X_new.append(V)
        lable=y[model_number+i]
        y_new.append(lable)
    return np.array(X_new), np.array(y_new)

def step_data_model_t6(X,y,prediction_bilstm_norm,prediction_bilstm_norm_t1,prediction_bilstm_norm_t2,prediction_bilstm_norm_t3,prediction_bilstm_norm_t4,prediction_bilstm_norm_t5,model_number):
    X_new=[]
    y_new=[]
    for i in range(len(prediction_bilstm_norm_t1)-model_number):
        V=X[i+model_number].copy()
        V[TIME_STEPS-6][0]=prediction_bilstm_norm[i]
        V[TIME_STEPS-5][0]=prediction_bilstm_norm_t1[i]
        V[TIME_STEPS-4][0]=prediction_bilstm_norm_t2[i]
        V[TIME_STEPS-3][0]=prediction_bilstm_norm_t3[i]
        V[TIME_STEPS-2][0]=prediction_bilstm_norm_t4[i]
        V[TIME_STEPS-1][0]=prediction_bilstm_norm_t5[i]
        X_new.append(V)
        lable=y[model_number+i]
        y_new.append(lable)
    return np.array(X_new), np.array(y_new)





X_new_list=[]
y_new_list=[]
pre_list=[]
history_bilstm = fit_model(model_list[0],X_train, y_train)


print('-----1-----')
prediction_bilstm_norm = prediction_norm(model_list[0],X_train)
pre_list.append(prediction_bilstm_norm)
X_new_t1,y_new_t1=step_data_model(X_train,y_train,pre_list[0],1)

X_new_list.append(X_new_t1)
y_new_list.append(y_new_t1)

history_bilstm_t1 = fit_model(model_list[1],X_new_list[0],y_new_list[0])



print('-----2-----')
prediction_bilstm_norm_t1 = prediction_norm(model_list[1],X_new_list[0])
pre_list.append(prediction_bilstm_norm_t1)
X_new_t2,y_new_t2=step_data_model_t2(X_train,y_train,pre_list[0],pre_list[1],2)


X_new_list.append(X_new_t2)
y_new_list.append(y_new_t2)

history_bilstm_t2 = fit_model(model_list[2],X_new_list[1],y_new_list[1])

print('-----3-----')
prediction_bilstm_norm_t2 = prediction_norm(model_list[2],X_new_list[1])
pre_list.append(prediction_bilstm_norm_t2)
X_new_t3,y_new_t3=step_data_model_t3(X_train,y_train,pre_list[0],pre_list[1],pre_list[2],3)

X_new_list.append(X_new_t3)
y_new_list.append(y_new_t3)

history_bilstm_t3 = fit_model(model_list[3],X_new_list[2],y_new_list[2])


print('-----4-----')
prediction_bilstm_norm_t3 = prediction_norm(model_list[3],X_new_list[2])
pre_list.append(prediction_bilstm_norm_t3)
X_new_t4,y_new_t4=step_data_model_t4(X_train,y_train,pre_list[0],pre_list[1],pre_list[2],pre_list[3],4)

X_new_list.append(X_new_t4)
y_new_list.append(y_new_t4)

history_bilstm_t4 = fit_model(model_list[4],X_new_list[3],y_new_list[3])



print('----5------')
prediction_bilstm_norm_t4 = prediction_norm(model_list[4],X_new_list[3])
pre_list.append(prediction_bilstm_norm_t4)
X_new_t5,y_new_t5=step_data_model_t5(X_train,y_train,pre_list[0],pre_list[1],pre_list[2],pre_list[3],pre_list[4],5)

X_new_list.append(X_new_t5)
y_new_list.append(y_new_t5)

history_bilstm_t5 = fit_model(model_list[5],X_new_list[4],y_new_list[4])



print('-----6-----')
prediction_bilstm_norm_t5 = prediction_norm(model_list[5],X_new_list[4])
pre_list.append(prediction_bilstm_norm_t5)
X_new_t6,y_new_t6=step_data_model_t6(X_train,y_train,pre_list[0],pre_list[1],pre_list[2],pre_list[3],pre_list[4],pre_list[5],6)

X_new_list.append(X_new_t6)
y_new_list.append(y_new_t6)

history_bilstm_t6 = fit_model(model_list[6],X_new_list[5],y_new_list[5])



for i in range(num_of_model-7):
    i=i+6
    print('model number',i)
    prediction_bilstm_norm = prediction_norm(model_list[i],X_new_list[i-1])
    pre_list.append(prediction_bilstm_norm)
    X_new,y_new=step_data_model_t6(X_train,y_train,pre_list[i-5],pre_list[i-4],pre_list[i-3],pre_list[i-2],pre_list[i-1],pre_list[i],i+1)
    X_new_list.append(X_new)
    y_new_list.append(y_new)
    history_bilstm_t15 = fit_model(model_list[i+1],X_new_list[i],y_new_list[i])



# model1 = keras.models.load_model("modelt1")
# print('end1')
# model2 = keras.models.load_model("modelt2")
# model3 = keras.models.load_model("modelt3")
# model4 = keras.models.load_model("modelt4")
# model5 = keras.models.load_model("modelt5")
# model6 = keras.models.load_model("modelt6")
# print('end6')
# model7 = keras.models.load_model("modelt7")
# model8 = keras.models.load_model("modelt8")
# model9 = keras.models.load_model("modelt9")
# model10 = keras.models.load_model("modelt10")
# model11 = keras.models.load_model("modelt11")
# model12 = keras.models.load_model("modelt12")
# model13 = keras.models.load_model("modelt13")
# model14 = keras.models.load_model("modelt14")
# model15 = keras.models.load_model("modelt15")
# model16 = keras.models.load_model("modelt16")
# model17 = keras.models.load_model("modelt17")
# model18 = keras.models.load_model("modelt18")
# model19 = keras.models.load_model("modelt19")
# model20 = keras.models.load_model("modelt20")
# print('end20')
# model21 = keras.models.load_model("modelt21")
# model22 = keras.models.load_model("modelt22")
# model23 = keras.models.load_model("modelt23")
# model24 = keras.models.load_model("modelt24")
# model25 = keras.models.load_model("modelt25")
# model26 = keras.models.load_model("modelt26")
# model27 = keras.models.load_model("modelt27")
# model28 = keras.models.load_model("modelt28")
# model29 = keras.models.load_model("modelt29")
# model30 = keras.models.load_model("modelt30")

# model_list=[model1,model2,model3,model4,model5,model6,model7,model8,model9,model10]#,model11,model12,model13,model14,model15,model16,model17,model18,model19,model20,model21,model22,model23,model24,model25,model26,model27,model28,model29,model30]

# y_new_trans=[]


# for i in range(10):
#     y_train = scaler_y.inverse_transform(y_new_list[i])
#     y_new_trans.append(y_train)


def plot_future(prediction, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='True Future')     
    plt.plot(np.arange(range_future),np.array(prediction),
            label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time (steps)')
    plt.ylabel('')



y_test = scaler_y.inverse_transform(y_test)


X_new_test_list=[]
y_new_test_list=[]
pre_test_list=[]

prediction_test_norm_t0 = prediction_norm(model_list[0],X_test)

pre_test_list.append(prediction_test_norm_t0)

X_new_test_t1,y_new_test_t1=step_data_model(X_test,y_test,pre_test_list[0],1)
#y_new_test_t1 = scaler_y.inverse_transform(y_new_test_t1)
X_new_test_list.append(X_new_test_t1)
y_new_test_list.append(y_new_test_t1)

prediction_test_norm_t1 = prediction_norm(model_list[1],X_new_test_list[0])


pre_test_list.append(prediction_test_norm_t1)

X_new_test_t2,y_new_test_t2=step_data_model_t2(X_test,y_test,pre_test_list[0],pre_test_list[1],2)
#y_new_test_t2 = scaler_y.inverse_transform(y_new_test_t2)
X_new_test_list.append(X_new_test_t2)
y_new_test_list.append(y_new_test_t2)

prediction_test_norm_t2 = prediction_norm(model_list[2],X_new_test_list[1])

pre_test_list.append(prediction_test_norm_t2)

X_new_test_t3,y_new_test_t3=step_data_model_t3(X_test,y_test,pre_test_list[0],pre_test_list[1],pre_test_list[2],3)
#y_new_test_t3 = scaler_y.inverse_transform(y_new_test_t3)
X_new_test_list.append(X_new_test_t3)
y_new_test_list.append(y_new_test_t3)
prediction_test_norm_t3 = prediction_norm(model_list[3],X_new_test_list[2])

pre_test_list.append(prediction_test_norm_t3)

X_new_test_t4,y_new_test_t4=step_data_model_t4(X_test,y_test,pre_test_list[0],pre_test_list[1],pre_test_list[2],pre_test_list[3],4)
#y_new_test_t4 = scaler_y.inverse_transform(y_new_test_t4)
X_new_test_list.append(X_new_test_t4)
y_new_test_list.append(y_new_test_t4)
prediction_test_norm_t4 = prediction_norm(model_list[4],X_new_test_list[3])

pre_test_list.append(prediction_test_norm_t4)

X_new_test_t5,y_new_test_t5=step_data_model_t5(X_test,y_test,pre_test_list[0],pre_test_list[1],pre_test_list[2],pre_test_list[3],pre_test_list[4],5)
#y_new_test_t5 = scaler_y.inverse_transform(y_new_test_t5)
X_new_test_list.append(X_new_test_t5)
y_new_test_list.append(y_new_test_t5)
prediction_test_norm_t5 = prediction_norm(model_list[5],X_new_test_list[4])

pre_test_list.append(prediction_test_norm_t5)

X_new_test_t6,y_new_test_t6=step_data_model_t6(X_test,y_test,pre_test_list[0],pre_test_list[1],pre_test_list[2],pre_test_list[3],pre_test_list[4],pre_test_list[5],6)
#y_new_test_t6 = scaler_y.inverse_transform(y_new_test_t6)
X_new_test_list.append(X_new_test_t6)
y_new_test_list.append(y_new_test_t6)
prediction_test_norm_t6 = prediction_norm(model_list[6],X_new_test_list[5])

pre_test_list.append(prediction_test_norm_t6)




for i in range(num_of_model-7):
    i=i+6
    X_new_test,y_new_test=step_data_model_t6(X_test,y_test,pre_test_list[i-5],pre_test_list[i-4],pre_test_list[i-3],pre_test_list[i-2],pre_test_list[i-1],pre_test_list[i],i+1)
    #y_new_test = scaler_y.inverse_transform(y_new_test)
    X_new_test_list.append(X_new_test)
    y_new_test_list.append(y_new_test)
    prediction_test_norm = prediction_norm(model_list[i+1],X_new_test_list[i])
    pre_test_list.append(prediction_test_norm)




prediction_test_list=[]

prediction_bilstm_test = prediction(model_list[0],X_test)
prediction_test_list.append(prediction_bilstm_test)

evaluate_prediction(prediction_bilstm_test, y_test, 'Bidirectional LSTM',1)


prediction_bilstm_test = prediction(model_list[1],X_new_test_list[0])
prediction_test_list.append(prediction_bilstm_test)
evaluate_prediction(prediction_test_list[1], y_new_test_list[0], 'Bidirectional LSTM',2)


for i in range(num_of_model-2):
    i=i+1
    prediction_bilstm_test = prediction(model_list[i+1],X_new_test_list[i])
    prediction_test_list.append(prediction_bilstm_test)
    evaluate_prediction(prediction_test_list[i+1], y_new_test_list[i], 'Bidirectional LSTM',i+2)



# plot_future(prediction_bilstm_test, y_test)
# plot_future(prediction_bilstm_t1_test, y_new_test_t1)
# plot_future(prediction_bilstm_t4_test, y_new_test_t4)
# plot_future(prediction_bilstm_t8_test, y_new_test_t8)
# plot_future(prediction_bilstm_t12_test, y_new_test_t12)
# plot_future(prediction_bilstm_t15_test, y_new_test_t15)

for i in range(30):
    model_list[i].save_weights(f"./modelW1/model{i}")

for i in range(number_of_runs):
    i_i=i*1000
    
    predict_multi_step_multi_model=[]
    for n in range(num_of_model):
        predict_multi_step_multi_model.append(prediction_test_list[n][i_i])
    true_val=y_test[i_i:i_i+num_of_model]
    #predict_single_step_data=prediction_bilstm_test[i:i+16]

    plt.figure(i+50)
    plt.plot(true_val, label='True')
    #plt.plot(predict_data_multi_step,label='Predict')
    plt.plot(predict_multi_step_multi_model,label='Predict_multi_sep_multi_model')
    #plt.plot(predict_single_step_data,label='Predict_single_step')
    plt.legend()

plt.show()






