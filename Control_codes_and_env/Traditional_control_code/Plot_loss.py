import matplotlib.pyplot as plt
import csv
from scipy import stats
import pandas as pd
import numpy as np

loss_1=[]
loss_2=[]
loss_3=[]
loss_4=[]
loss_mix=[]

with open('Loss_1.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        loss_1.append(row)
csvFile.close()

with open('Loss_2.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        loss_2.append(row)
csvFile.close()

with open('Loss_3.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        loss_3.append(row)
csvFile.close()

with open('Loss_4.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        loss_4.append(row)
csvFile.close()

with open('Loss_mix.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        loss_mix.append(row)
csvFile.close()

df_loss1 = pd.DataFrame(loss_1)
df_loss1.columns = ["loss","epsilon"]

df_loss2 = pd.DataFrame(loss_2)
df_loss2.columns = ["loss","epsilon"]

df_loss3 = pd.DataFrame(loss_3)
df_loss3.columns = ["loss","epsilon"]

df_loss4 = pd.DataFrame(loss_4)
df_loss4.columns = ["loss","epsilon"]

df_lossmix = pd.DataFrame(loss_mix)
df_lossmix.columns = ["loss","epsilon"]


df_loss1=df_loss1.astype(float)
df_loss2=df_loss2.astype(float)
df_loss3=df_loss3.astype(float)
df_loss4=df_loss4.astype(float)
df_lossmix=df_lossmix.astype(float)

f1 = plt.figure(1)
plt.style.use('ggplot')
plt.plot(df_loss1['loss'],label='RL', alpha=0.7)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.legend()

f2 = plt.figure(2)
plt.style.use('ggplot')
plt.plot(df_loss2['loss'],label='RL', alpha=0.7)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.legend()

f3 = plt.figure(3)
plt.style.use('ggplot')
plt.plot(df_loss3['loss'],label='RL', alpha=0.7)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.legend()

f4 = plt.figure(4)
plt.style.use('ggplot')
plt.plot(df_loss4['loss'],label='RL', alpha=0.7)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.legend()

f5 = plt.figure(5)
plt.style.use('ggplot')
plt.plot(df_lossmix['loss'],label='RL', alpha=0.7)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.legend()


plt.show()
