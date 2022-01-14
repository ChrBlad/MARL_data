
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


data_zone1=[]
data_actions=[]

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

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


df_actions = pd.DataFrame(data_actions)
df_actions.columns = ["Valve1", "Supplytemp","Reward","Hardcon","price_reward","sum_mix_reward","change_in_Supply_temp"]


df_zone1 = pd.DataFrame(data_zone1)
df_zone1.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"time"]

data = pd.concat([df_actions, df_zone1], axis=1, sort=False)



dataset_t = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_t = dataset_t.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t', 'Hardcon': 'Hardcon_t', 'Hcv': 'Hcv_t', 'Supplytemp': 'Supplytemp_t'}, inplace = False)

dataset_tm1 = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_tm1 = dataset_tm1.rename(columns = {'Troom': 'Troom_tm1', 'Tamb': 'Tamb_tm1', 'Hardcon': 'Hardcon_tm1', 'Hcv': 'Hcv_tm1', 'Supplytemp': 'Supplytemp_tm1'}, inplace = False)

dataset_tm2 = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_tm2 = dataset_tm2.rename(columns = {'Troom': 'Troom_tm2', 'Tamb': 'Tamb_tm2', 'Hardcon': 'Hardcon_tm2', 'Hcv': 'Hcv_tm2', 'Supplytemp': 'Supplytemp_tm2'}, inplace = False)

dataset_tm3 = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_tm3 = dataset_tm3.rename(columns = {'Troom': 'Troom_tm3', 'Tamb': 'Tamb_tm3', 'Hardcon': 'Hardcon_tm3', 'Hcv': 'Hcv_tm3', 'Supplytemp': 'Supplytemp_tm3'}, inplace = False)

dataset_tm4 = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_tm4 = dataset_tm4.rename(columns = {'Troom': 'Troom_tm4', 'Tamb': 'Tamb_tm4', 'Hardcon': 'Hardcon_tm4', 'Hcv': 'Hcv_tm4', 'Supplytemp': 'Supplytemp_tm4'}, inplace = False)

dataset_tm5 = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_tm5 = dataset_tm5.rename(columns = {'Troom': 'Troom_tm5', 'Tamb': 'Tamb_tm5', 'Hardcon': 'Hardcon_tm5', 'Hcv': 'Hcv_tm5', 'Supplytemp': 'Supplytemp_tm5'}, inplace = False)

dataset_tm6 = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
dataset_tm6 = dataset_tm6.rename(columns = {'Troom': 'Troom_tm6', 'Tamb': 'Tamb_tm6', 'Hardcon': 'Hardcon_tm6', 'Hcv': 'Hcv_tm6', 'Supplytemp': 'Supplytemp_tm6'}, inplace = False)


print(dataset_t.head())

dataset_t.drop([0], inplace=True)
dataset_tm1.drop([0,1], inplace=True)
dataset_tm2.drop([0,1,2], inplace=True)
dataset_tm3.drop([0,1,2,3], inplace=True)
dataset_tm4.drop([0,1,2,3,4], inplace=True)
dataset_tm5.drop([0,1,2,3,4,5], inplace=True)
dataset_tm6.drop([0,1,2,3,4,5,6], inplace=True)
dataset_t.reset_index(inplace=True, drop=True)
dataset_tm1.reset_index(inplace=True, drop=True)
dataset_tm2.reset_index(inplace=True, drop=True)
dataset_tm3.reset_index(inplace=True, drop=True)
dataset_tm4.reset_index(inplace=True, drop=True)
dataset_tm5.reset_index(inplace=True, drop=True)
dataset_tm6.reset_index(inplace=True, drop=True)
lables=data[['Troom']].copy()



dataset = pd.concat([dataset_t,dataset_tm1, dataset_tm2,dataset_tm3,dataset_tm4,dataset_tm5,dataset_tm6], axis=1, join='inner')

len_dataset=len(dataset)

lables.drop([len_dataset,len_dataset+1,len_dataset+2,len_dataset+3,len_dataset+4,len_dataset+5,len_dataset+6], inplace=True)


dataset = pd.concat([lables,dataset_t,dataset_tm1, dataset_tm2,dataset_tm3,dataset_tm4,dataset_tm5,dataset_tm6], axis=1, join='inner')
#print(dataset_t.tail())
#print(dataset_tm2.tail())

dataset=dataset.astype(float)
dataset=dataset.round(decimals=4)

print(dataset.head())
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Troom")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('Troom')
test_labels = test_dataset.pop('Troom')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
  model = keras.Sequential([
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='elu', input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.8),
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='elu'),
    layers.Dropout(0.8),
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='elu'),
    layers.Dense(1)
  ])

  lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)
  optimizer = tf.keras.optimizers.Adam(lr_schedule)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()



model.summary()

EPOCHS = 1000

normed_train_data = np.asarray(normed_train_data)
train_labels = np.asarray(train_labels)


normed_test_data = np.asarray(normed_test_data)
test_labels = np.asarray(test_labels)

history = model.fit(
  normed_train_data[:200], train_labels[:200],
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} Troom".format(mae))


test_predictions = model.predict(normed_test_data).flatten()


f1 = plt.figure(1)
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Troom]')
plt.ylabel('Predictions [Troom]')



f2 = plt.figure(2)
error = test_predictions - test_labels
plt.hist(error, bins = 200)
plt.xlabel("Prediction Error [Troom]")
plt.ylabel("Count")

plt.show()