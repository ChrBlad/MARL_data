
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
import csv

class fun:
    def __init__(self):
        #tf.compat.v1.disable_eager_execution()
        self.TIME_STEPS =6
        self.Bw=500
        self.Br=343
        self.Ba=37
        self.Bs=0.4#0.0001
        self.rho_water=1000
        self.C_water=4186
        self.C_room=1005
        self.C_floor=750
        self.m_water=28.35#asumtions about lengt of pip - 100m diameter 0.019
        self.m_room=63.8 #asumtions about room - 20m2 and 2.5m in hight
        self.m_floor=8800 # concret 0.2m over 20m2

        #initail states

        self.flow=0
        self.T_room=20+273.15
        self.T_floor=23+273.15
        self.T_supply=40+273.15
        self.T_return=24+273.15
        self.T_amb=5+273.15
        self.t=0
        self.dt=10

    def load_extended_data(self):
        data_zone1=[]
        data_actions=[]
        data_out=[]
        with open('output_zone1_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_zone1.append(row)

        csvFile.close()

        with open('output_actions_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_actions.append(row)

        csvFile.close()

        with open('output_out_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_out.append(row)

        csvFile.close()


        self.df_out = pd.DataFrame(data_out)
        self.df_out.columns = ['RoomTemperature1','RoomTemperature2','RoomTemperature3','RoomTemperature4','Valveout1','Valveout2','Valveout3','Valveout4','Power1','Power2','Power3','Power4','Tamb1','Sun1','Tambforecast1','Tambforecast2','Tambforecast3','Sunforecast1','Sunforecast2','Sunforecast3','Powerwater1','Powerwater2','Powerwater3','Powerwater4','Treturn1','Treturn2','Treturn3','Treturn4','Flow1','Flow2','Flow3','Flow4']
        self.df_out=self.df_out.astype(float)

        df_actions = pd.DataFrame(data_actions)
        df_actions.columns = ["Valve1", "Supplytemp","Reward","Hardcon","price_reward","sum_mix_reward","change_in_Supply_temp"]


        self.df_zone1 = pd.DataFrame(data_zone1)
        self.df_zone1.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"TOD","time"]
        self.df_zone1=self.df_zone1.astype(float)
        data = pd.concat([df_actions, self.df_zone1], axis=1, sort=False)
        
        self.dataset = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp']].copy()
        print("----------Loading of extended data complet------------")
        return self.dataset,self.df_out


    def invers_y1(self,prediction):
        prediction = self.scaler_y1.inverse_transform(prediction)
        return prediction

    def invers_x1(self,x_data):
        x_data = self.scaler_x1.inverse_transform(x_data)
        return x_data

    def norm_x1(self,x_data):
        x_data = self.input_scaler1.transform(x_data)
        return x_data



    def invers_y2(self,prediction):
        prediction = self.scaler_y2.inverse_transform(prediction)
        return prediction

    def invers_x2(self,x_data):
        x_data = self.scaler_x2.inverse_transform(x_data)
        return x_data

    def norm_x2(self,x_data):
        x_data = self.input_scaler2.transform(x_data)
        return x_data



    def invers_y3(self,prediction):
        prediction = self.scaler_y3.inverse_transform(prediction)
        return prediction

    def invers_x3(self,x_data):
        x_data = self.scaler_x3.inverse_transform(x_data)
        return x_data

    def norm_x3(self,x_data):
        x_data = self.input_scaler3.transform(x_data)
        return x_data



    def invers_y4(self,prediction):
        prediction = self.scaler_y4.inverse_transform(prediction)
        return prediction

    def invers_x4(self,x_data):
        x_data = self.scaler_x4.inverse_transform(x_data)
        return x_data

    def norm_x4(self,x_data):
        x_data = self.input_scaler4.transform(x_data)
        return x_data


    def load_model(self):
        model_list1=[]
        model_list2=[]
        model_list3=[]
        model_list4=[]
        print('begin load')
        #T1=time.time()
        # model=keras.models.load_model("Zone1_models\modelt1")
        # T2=time.time()
        # print('end load',T2-T1)
        for i in range(30):
            T1=time.time()
            tf.keras.backend.clear_session()
            model_list1.append(keras.models.load_model(f"Zone{1}_models\modelt{i+1}"))
            tf.keras.backend.clear_session()
            T2=time.time()
            print('end load model 1 of 4',T2-T1)
            model_list2.append(keras.models.load_model(f"Zone{2}_models\modelt{i+1}"))
            tf.keras.backend.clear_session()
            model_list3.append(keras.models.load_model(f"Zone{3}_models\modelt{i+1}"))
            tf.keras.backend.clear_session()
            model_list4.append(keras.models.load_model(f"Zone{4}_models\modelt{i+1}"))
            tf.keras.backend.clear_session()
            print('end load model 4 of 4',i)
        # for i in range(30):
        #     model_list1.append(model)
        #     model_list2.append(model)
        #     model_list3.append(model)
        #     model_list4.append(model)
        return model_list1,model_list2,model_list3,model_list4



    def create_dataset (self,X, y, time_steps = 1):
        Xs, ys = [], []
        for i in range(len(X)-time_steps):
            v = X[i:i+time_steps, :]
            Xs.append(v)
            ys.append(y[i+time_steps])
        return np.array(Xs), np.array(ys)

    def load_data(self):

        data_zone1=[]
        data_zone2=[]
        data_zone3=[]
        data_zone4=[]
        data_actions=[]
        data_out=[]
        with open('output_zone1_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_zone1.append(row)

        csvFile.close()

        with open('output_zone2_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_zone2.append(row)

        csvFile.close()

        with open('output_zone3_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_zone3.append(row)

        csvFile.close()

        with open('output_zone4_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_zone4.append(row)

        csvFile.close()

        with open('output_actions_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_actions.append(row)

        csvFile.close()

        with open('output_out_extend.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                data_out.append(row)

        csvFile.close()


        self.df_out = pd.DataFrame(data_out)
        self.df_out.columns = ['RoomTemperature1','RoomTemperature2','RoomTemperature3','RoomTemperature4','Valveout1','Valveout2','Valveout3','Valveout4','Power1','Power2','Power3','Power4','Tamb1','Sun1','Tambforecast1','Tambforecast2','Tambforecast3','Sunforecast1','Sunforecast2','Sunforecast3','Powerwater1','Powerwater2','Powerwater3','Powerwater4','Treturn1','Treturn2','Treturn3','Treturn4','Flow1','Flow2','Flow3','Flow4']
        self.df_out=self.df_out.astype(float)


        df_actions = pd.DataFrame(data_actions)
        df_actions.columns = ["Valve1", "Supplytemp","Reward","Hardcon","price_reward","sum_mix_reward","change_in_Supply_temp"]
        df_actions=df_actions.astype(float)

        self.df_zone1 = pd.DataFrame(data_zone1)
        self.df_zone1.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater1','Hcv',"TOD","time"]

        self.df_zone2 = pd.DataFrame(data_zone2)
        self.df_zone2.columns = ["Troom", "Tamb", "Valve","Power2","Reward","action",'Price','Power','Powerwater2','Hcv',"TOD","time"]

        self.df_zone3 = pd.DataFrame(data_zone3)
        self.df_zone3.columns = ["Troom", "Tamb", "Valve","Power3","Reward","action",'Price','Power','Powerwater3','Hcv',"TOD","time"]

        self.df_zone4 = pd.DataFrame(data_zone4)
        self.df_zone4.columns = ["Troom", "Tamb", "Valve","Power4","Reward","action",'Price','Power','Powerwater4','Hcv',"TOD","time"]
        
        
        self.df_zone1=self.df_zone1.astype(float)
        self.df_zone2=self.df_zone2.astype(float)
        self.df_zone3=self.df_zone3.astype(float)
        self.df_zone4=self.df_zone4.astype(float)
        
        
        data1 = pd.concat([df_actions, self.df_zone1,self.df_out], axis=1, sort=False)
        data2 = pd.concat([df_actions, self.df_zone2,self.df_out], axis=1, sort=False)
        data3 = pd.concat([df_actions, self.df_zone3,self.df_out], axis=1, sort=False)
        data4 = pd.concat([df_actions, self.df_zone4,self.df_out], axis=1, sort=False)
        print(self.df_zone1.head())
        #self.dataset = data[['Troom', 'Tamb', 'Valve','Hardcon','Hcv','Supplytemp','TOD']].copy()
        #sself.dataset_t = self.dataset.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t',, 'Hardcon': 'Hardcon_t', 'Hcv': 'Hcv_t', 'Supplytemp': 'Supplytemp_t','tod':'tod_t'}, inplace = False)

        self.dataset1 = data1[['Troom', 'Tamb', 'Valveout1','Supplytemp','TOD','Sun1','Sunforecast1','Tambforecast1']].copy()
        self.dataset_t1 = self.dataset1.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t','Valve':'Valve_t' , 'Supplytemp': 'Supplytemp_t','tod':'tod_t','Sun1':'Sun1_t','Sunforecast1':'Sunforecast1_t','Tambforecast1':'Tambforecast1_t'}, inplace = False)

        self.dataset2 = data2[['Troom', 'Tamb', 'Valveout1','Supplytemp','TOD','Sun1','Sunforecast1','Tambforecast1']].copy()
        self.dataset_t2 = self.dataset2.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t','Valve':'Valve_t' , 'Supplytemp': 'Supplytemp_t','tod':'tod_t','Sun1':'Sun1_t','Sunforecast1':'Sunforecast1_t','Tambforecast1':'Tambforecast1_t'}, inplace = False)
        self.dataset3 = data3[['Troom', 'Tamb', 'Valveout1','Supplytemp','TOD','Sun1','Sunforecast1','Tambforecast1']].copy()
        self.dataset_t3 = self.dataset3.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t','Valve':'Valve_t' , 'Supplytemp': 'Supplytemp_t','tod':'tod_t','Sun1':'Sun1_t','Sunforecast1':'Sunforecast1_t','Tambforecast1':'Tambforecast1_t'}, inplace = False)
        self.dataset4 = data4[['Troom', 'Tamb', 'Valveout1','Supplytemp','TOD','Sun1','Sunforecast1','Tambforecast1']].copy()
        self.dataset_t4 = self.dataset4.rename(columns = {'Troom': 'Troom_t', 'Tamb': 'Tamb_t','Valve':'Valve_t' , 'Supplytemp': 'Supplytemp_t','tod':'tod_t','Sun1':'Sun1_t','Sunforecast1':'Sunforecast1_t','Tambforecast1':'Tambforecast1_t'}, inplace = False)


        self.lables1=data1[['Troom']].copy()
        self.lables1.reset_index(inplace=True, drop=True)
        self.dataset1 = pd.concat([self.dataset_t1], axis=1, join='inner')
        self.dataset1 = pd.concat([self.lables1,self.dataset_t1], axis=1, join='inner')
        self.dataset1=self.dataset1.astype(float)


        self.lables2=data2[['Troom']].copy()
        self.lables2.reset_index(inplace=True, drop=True)
        self.dataset2 = pd.concat([self.dataset_t2], axis=1, join='inner')
        self.dataset2 = pd.concat([self.lables2,self.dataset_t2], axis=1, join='inner')
        self.dataset2=self.dataset2.astype(float)



        self.lables3=data3[['Troom']].copy()
        self.lables3.reset_index(inplace=True, drop=True)
        self.dataset3 = pd.concat([self.dataset_t3], axis=1, join='inner')
        self.dataset3 = pd.concat([self.lables3,self.dataset_t3], axis=1, join='inner')
        self.dataset3=self.dataset3.astype(float)



        self.lables4=data4[['Troom']].copy()
        self.lables4.reset_index(inplace=True, drop=True)
        self.dataset4 = pd.concat([self.dataset_t4], axis=1, join='inner')
        self.dataset4 = pd.concat([self.lables4,self.dataset_t4], axis=1, join='inner')
        self.dataset4=self.dataset4.astype(float)



        print("----------Loading of data complet------------")
        return self.dataset1,self.dataset2,self.dataset3,self.dataset4,self.df_out,self.df_zone1,self.df_zone2,self.df_zone3,self.df_zone4,df_actions

    def norm_data1(self,dataset,timestep):
        print('begin norm and split')
        TIME_STEPS=timestep
        print('lenght of extended data',len(dataset)/108)
        train_size = int(len(dataset))#int(108*500)
        train_dataset, test_dataset = dataset.iloc[:train_size],dataset.iloc[train_size:]

        train_dataset = pd.concat([train_dataset], axis=0, join='inner')
        train_dataset.reset_index(inplace=True, drop=True)


        # Split train data to X and y
        X_train = train_dataset.drop('Troom', axis = 1)
        y_train = train_dataset.loc[:,['Troom']]


        # Different scaler for input and output
        self.scaler_x1 = MinMaxScaler(feature_range = (0,1))
        self.scaler_y1 = MinMaxScaler(feature_range = (0,1))
        # Fit the scaler using available training data
        self.input_scaler1 = self.scaler_x1.fit(X_train)
        self.output_scaler1 = self.scaler_y1.fit(y_train)
        # Apply the scaler to training data
        train_y_norm = self.output_scaler1.transform(y_train)
        train_x_norm = self.input_scaler1.transform(X_train)


        X_train, y_train = self.create_dataset(train_x_norm, train_y_norm,TIME_STEPS)
        #print('x train',X_train[0])
        #print('x train',y_train[0])
        print('end norm and split')
        return X_train, y_train



    def norm_data2(self,dataset,timestep):
        print('begin norm and split')
        TIME_STEPS=timestep
        print('lenght of extended data',len(dataset)/108)
        train_size = int(len(dataset))#int(108*500)
        train_dataset, test_dataset = dataset.iloc[:train_size],dataset.iloc[train_size:]

        train_dataset = pd.concat([train_dataset], axis=0, join='inner')
        train_dataset.reset_index(inplace=True, drop=True)


        # Split train data to X and y
        X_train = train_dataset.drop('Troom', axis = 1)
        y_train = train_dataset.loc[:,['Troom']]


        # Different scaler for input and output
        self.scaler_x2 = MinMaxScaler(feature_range = (0,1))
        self.scaler_y2 = MinMaxScaler(feature_range = (0,1))
        # Fit the scaler using available training data
        self.input_scaler2 = self.scaler_x2.fit(X_train)
        self.output_scaler2 = self.scaler_y2.fit(y_train)
        # Apply the scaler to training data
        train_y_norm = self.output_scaler2.transform(y_train)
        train_x_norm = self.input_scaler2.transform(X_train)


        X_train, y_train = self.create_dataset(train_x_norm, train_y_norm,TIME_STEPS)
        #print('x train',X_train[0])
        #print('x train',y_train[0])
        print('end norm and split')
        return X_train, y_train



    def norm_data3(self,dataset,timestep):
        print('begin norm and split')
        TIME_STEPS=timestep
        print('lenght of extended data',len(dataset)/108)
        train_size = int(len(dataset))#int(108*500)
        train_dataset, test_dataset = dataset.iloc[:train_size],dataset.iloc[train_size:]

        train_dataset = pd.concat([train_dataset], axis=0, join='inner')
        train_dataset.reset_index(inplace=True, drop=True)


        # Split train data to X and y
        X_train = train_dataset.drop('Troom', axis = 1)
        y_train = train_dataset.loc[:,['Troom']]


        # Different scaler for input and output
        self.scaler_x3 = MinMaxScaler(feature_range = (0,1))
        self.scaler_y3 = MinMaxScaler(feature_range = (0,1))
        # Fit the scaler using available training data
        self.input_scaler3 = self.scaler_x3.fit(X_train)
        self.output_scaler3 = self.scaler_y3.fit(y_train)
        # Apply the scaler to training data
        train_y_norm = self.output_scaler3.transform(y_train)
        train_x_norm = self.input_scaler3.transform(X_train)


        X_train, y_train = self.create_dataset(train_x_norm, train_y_norm,TIME_STEPS)
        #print('x train',X_train[0])
        #print('x train',y_train[0])
        print('end norm and split')
        return X_train, y_train



    def norm_data4(self,dataset,timestep):
        print('begin norm and split')
        TIME_STEPS=timestep
        print('lenght of extended data',len(dataset)/108)
        train_size = int(len(dataset))#int(108*500)
        train_dataset, test_dataset = dataset.iloc[:train_size],dataset.iloc[train_size:]

        train_dataset = pd.concat([train_dataset], axis=0, join='inner')
        train_dataset.reset_index(inplace=True, drop=True)


        # Split train data to X and y
        X_train = train_dataset.drop('Troom', axis = 1)
        y_train = train_dataset.loc[:,['Troom']]


        # Different scaler for input and output
        self.scaler_x4 = MinMaxScaler(feature_range = (0,1))
        self.scaler_y4 = MinMaxScaler(feature_range = (0,1))
        # Fit the scaler using available training data
        self.input_scaler4 = self.scaler_x4.fit(X_train)
        self.output_scaler4 = self.scaler_y4.fit(y_train)
        # Apply the scaler to training data
        train_y_norm = self.output_scaler4.transform(y_train)
        train_x_norm = self.input_scaler4.transform(X_train)


        X_train, y_train = self.create_dataset(train_x_norm, train_y_norm,TIME_STEPS)
        #print('x train',X_train[0])
        #print('x train',y_train[0])
        print('end norm and split')
        return X_train, y_train