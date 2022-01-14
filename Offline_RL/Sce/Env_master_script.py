#Master for collecting all actions before sending collected action to FMU(dymola model)
# 
import time
import csv
import pandas as pd
import numpy as np
import random
from fmu_stepping import fmu_stepping
from fmpy.util import plot_result
from ai_input_provider import AiInputProvider
from reward_calculator import RewardCalculator
from NN_world import NN_world
from functions import fun
from parameters import Params
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,callbacks,layers
from tensorflow.keras.layers import Dense, LSTM,Dropout,GRU,Bidirectional
from loadmodels import models

class Envmasterscript:    
    def __init__(self):
        #tf.compat.v1.disable_eager_execution()
        self.models1=[]
        self.models2=[]
        self.models3=[]
        self.models4=[]
        self.models=models()
        params = Params
        self.Valve_actions=[]
        self.Supply_action=[]
        self.prediction_list=[]
        self.timestep=6
        self.NN_world=NN_world()
        self.fun=fun()
        self.done_nn=0
        self.prediction_list1=[]
        self.prediction_list2=[]
        self.prediction_list3=[]
        self.prediction_list4=[]
        self.power= np.array([0.15,0.15,0.15,0.15,0.15,0.17,0.31,0.69,0.76,0.95,0.35,0.15,0.15,0.15,0.19,0.28,0.35,0.51,0.75,0.90,0.80,0.45,0.38,0.15])
        self.KL=np.array([0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,3,3,3,3,3,0,0])*100+self.power*(500*0.4)
        self.BR1=np.array([2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2])*100+self.power*(500*0.2)
        self.BR2=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1])*100+self.power*(500*0.2)
        self.WC=np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*100+self.power*(500*0.2)
        self.Valve2=0
        self.Valve3=0
        self.Valve4=0
        self.V1_count=0
        self.V2_count=0
        self.V3_count=0
        self.V4_count=0
        self.hard_V1=0
        self.hard_V2=0
        self.hard_V3=0
        self.hard_V4=0
        self.NN_world_count_2=0
        self.NN_world_count=1
        self.hard_V1=0
        self.start_1D=1
        self.len_of_dataset=0
        self.D1_world_count_fin=0
        self.D1_world_count_1=0
        self.D1_world_count_2=0
        self.D1_world_bool= 0
        self.ai_input_provider = AiInputProvider(params)
        self.iteration_number = 0
        self.iteration_number1 = 0
        self.iteration_number2 = 0
        self.count_hcv =0
        self.start_point_int=0
        self.iteration_number3 = 0
        self.iteration_number4 = 0
        self.iteration_number_mix = 0
        self.iteration_number1_fin = 0
        self.Hardconstraint = 0
        self.done=0
        self.done_1D=0
        self.V1_count=0
        self.beta=0.5
        self.print_number = 0
        self.tod=0
        self.tod_1D=0
        self.Valve1=0
        self.start=0
        self.NN_world_count3=0
        self.test_predictions_last=0
        self.temp_pre =273.15+40
        self.last_temp = 273.15+40
        self.last_temp_2 =273.15+40
        self.last_temp_3 =273.15+40
        self.last_temp_4 =273.15+40
        self.simulation_time =0
        self.last_temp_5 =273.15+40
        self.last_temp_6 =273.15+40
        self.Hardconstraint_bool =1
        self.tref=273.15+22
        self.dd = 0
        self.start=1
        self.supplyagent = 1 # if 0 supply=45DC if 1 supply=action +35 DC
        self.VS11= 0#switch 1 room 1 if 1 hyscontrol if 0 AI  ----- VS21 needs to be 1
        self.VS21= 1#switch 2 room 1 if 1 VS11 works if 0 AI/withhys
        self.VS12= 0#switch 1 room 2
        self.VS22= 1#switch 2 room 2
        self.VS13= 0#switch 1 room 2
        self.VS23= 1#switch 2 room 2
        self.VS14= 0#switch 1 room 2
        self.VS24= 1#switch 2 room 2
        #start_time = time.clock()
        filename ='TwoElementHouse_04room_0FMU_0weatherforcast_Houses_testHouse_0hyscontrol.fmu' 
        start_time=0.0
        self.NN_world_bool =0
        stop_time=3155692600
        self.sample_time=800
        parameters={}
        input={'SupplyTemperature','Valve1','Valve2','Valve3','Valve4','FreeHeatRoom1','FreeHeatRoom2','FreeHeatRoom3','FreeHeatRoom4','ValveS1Room1','ValveS2Room1','ValveS1Room2','ValveS2Room2','ValveS1Room3','ValveS2Room3','ValveS1Room4','ValveS2Room4','TempRef'}
        self.input_values ={'SupplyTemperature': 340,'Valve1': 1,'Valve2': 1,'Valve3': 1,'Valve4': 1,'FreeHeatRoom1': 0,'FreeHeatRoom2': 0,'FreeHeatRoom3': 0,'FreeHeatRoom4': 0,'ValveS1Room1': 0,'ValveS2Room1': 0,'ValveS1Room2': 0,'ValveS2Room2': 0,'ValveS1Room3': 0,'ValveS2Room3': 0,'ValveS1Room4': 0,'ValveS2Room4': 0,'TempRef':273.15+22}
        output={'RoomTemperature1','RoomTemperature2','RoomTemperature3','RoomTemperature4','Valveout1','Valveout2','Valveout3','Valveout4','Power1','Power2','Power3','Power4','Tamb1','Sun1','Tambforecast1','Tambforecast2','Tambforecast3','Sunforecast1','Sunforecast2','Sunforecast3','Powerwater1','Powerwater2','Powerwater3','Powerwater4','Treturn1','Treturn2','Treturn3','Treturn4','Flow1','Flow2','Flow3','Flow4'}
        FMU_stepping = fmu_stepping(filename = filename,
                            start_time=start_time,
                            stop_time=stop_time,
                            sample_time=self.sample_time,
                            parameters=parameters,
                            input=input,
                            output=output)
        self.env = FMU_stepping


    def create_model_bilstm_t0(self,units):
        model = Sequential()
        model.add(Bidirectional(LSTM(units = units,return_sequences=True),input_shape=(6, 8)))
        model.add(Bidirectional(LSTM(units = units)))
        model.add(Dense(1))
        #Compile model
        model.compile(loss='mse', optimizer='adam')
        return model


    def load(self):
        self.models1=[]

        T4=time.time()
        for i in range(30):
            model1 = self.create_model_bilstm_t0(64)
            model2 = self.create_model_bilstm_t0(64)
            model3 = self.create_model_bilstm_t0(64)
            model4 = self.create_model_bilstm_t0(64)
            model1.load_weights(f"./modelW1/model{i}")
            model2.load_weights(f"./modelW2/model{i}")
            model3.load_weights(f"./modelW3/model{i}")
            model4.load_weights(f"./modelW4/model{i}")
            self.models1.append(model1)
            self.models2.append(model2)
            self.models3.append(model3)
            self.models4.append(model4)
            T5=time.time()
            print('end',T5-T4)


        return self.models1,self.models2,self.models3,self.models4

    def userbehavior1(self,timeofday):

        TOD = round(timeofday)
        if TOD > 23:
            TOD = 23
        free1 = 0#self.KL[TOD]
        return free1
    def userbehavior2(self,timeofday):

        TOD = round(timeofday)
        if TOD > 23:
            TOD = 23
        free2 = 0#self.KL[TOD]
        return free2

    def userbehavior3(self,timeofday):

        TOD = round(timeofday)
        if TOD > 23:
            TOD = 23
        free3 = 0#self.KL[TOD]
        return free3

    def userbehavior4(self,timeofday):

        TOD = round(timeofday)
        if TOD > 23:
            TOD = 23
        free4 = 0#self.KL[TOD]
        return free4

    def nightsetback(self,timeofday):
        TOD = round(timeofday)
        if TOD >22 or TOD < 6:
            self.tref = 273.15+18
        else:
            self.tref = 273.15+22
        return self.tref


    def check(self):
        return self.iteration_number1,self.iteration_number2,self.iteration_number3,self.iteration_number4, self.iteration_number_mix

    def check_mix(self):
        return self.iteration_number1,self.iteration_number2,self.iteration_number3,self.iteration_number4, self.iteration_number_mix, self.NN_world_bool,self.iteration_number1_fin

    def zone1(self,action,NN_world_bool,type):
        self.iteration_number1 = 1+self.iteration_number1
        if action == 0:
            self.yoda = 0
        elif action==1:
            self.yoda = 1
        return self.iteration_number1,self.iteration_number2 ,self.iteration_number3,self.iteration_number4,self.iteration_number_mix

    def zone2(self,action,NN_world_bool,type):
        self.iteration_number2 = 1+self.iteration_number2 
        if action == 0:
            self.zod = 0
        elif action==1:
            self.zod = 1
        return self.iteration_number1,self.iteration_number2 ,self.iteration_number3,self.iteration_number4,self.iteration_number_mix

    def zone3(self,action,NN_world_bool,type):
        self.iteration_number3 = 1+self.iteration_number3
        if action == 0:
            self.pumba = 0
        elif action==1:
            self.pumba = 1
        return self.iteration_number1,self.iteration_number2 ,self.iteration_number3,self.iteration_number4,self.iteration_number_mix

    def zone4(self,action,NN_world_bool,type):
        self.iteration_number4 = 1+self.iteration_number4
        if action == 0:
            self.spock = 0
        elif action==1:
            self.spock = 1
        return self.iteration_number1,self.iteration_number2 ,self.iteration_number3,self.iteration_number4,self.iteration_number_mix

    def Tsupply(self,action,NN_world_bool,type):
        self.iteration_number_mix = 1+self.iteration_number_mix
        #self.temp = action*25+20+273.15# Box action
        if self.supplyagent == 1:
            self.temp =action*1.2+30+273.15# descretaction*7.5+37.5+273.15# Box actionaction*1+30+273.15# descret
        elif self.iteration_number_mix < 2:
            self.temp =273.15+45
        else:
            self.temp = -1*(self.out['Tamb1']-273.15)*0.6+42+273.15
        return self.iteration_number_mix#,self.iteration_number1

    def start_agent(self):
        self.supplyagent = 1 # if 0 supply=45DC if 1 supply=action +35 DC
        self.VS11= 0#switch 1 room 1 if 1 hyscontrol if 0 AI  ----- VS21 needs to be 1
        self.VS21= 1#switch 2 room 1 if 1 VS11 works if 0 AI/withhys
    def sendaction (self,action):
        if  self.iteration_number1 == self.iteration_number_mix==self.iteration_number4==self.iteration_number3==self.iteration_number2:
            self.iteration_number = 1+self.iteration_number 
            if self.iteration_number == 108*60:## Start NN Env
                self.start_agent()
                #initial=self.build_NN_world()
                self.NN_world_bool =1
            self.simulation_time = self.iteration_number*self.sample_time/86400
            timeofday = (self.simulation_time-(self.simulation_time//1))*24
            self.start_agent()
            free1 = self.userbehavior1(timeofday)
            free2 = self.userbehavior2(timeofday)
            free3 = self.userbehavior3(timeofday)
            free4 = self.userbehavior4(timeofday)
            self.input_values = {'SupplyTemperature': self.temp,'Valve1': self.Valve1,'Valve2': self.Valve2,'Valve3': self.Valve3,'Valve4': self.Valve4,'FreeHeatRoom1': free1,'FreeHeatRoom2': free2,'FreeHeatRoom3': free3,'FreeHeatRoom4': free4,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'ValveS1Room2': self.VS12,'ValveS2Room2': self.VS22,'ValveS1Room3': self.VS13,'ValveS2Room3': self.VS23,'ValveS1Room4': self.VS14,'ValveS2Room4': self.VS24,'TempRef':self.tref}
            self.out = self.env.step(self.input_values)
            
            #print(self.out['RoomTemperature1'])
            if self.NN_world_bool ==1:
                print('send out',self.out)
            self.iteration_number1_fin =self.iteration_number1
            actions_to_env=self.input_values
            self.dd=0
        else:
            actions_to_env=0
            print('sss')
   
        return self.out ,actions_to_env,self.tref,self.NN_world_bool


    def load_data(self):
        self.dataset1,self.dataset2,self.dataset3,self.dataset4,self.df_out,self.df_zone1,self.df_zone2,self.df_zone3,self.df_zone4,self.df_actions=self.fun.load_data()
        self.len_of_dataset=len(self.dataset1)
        return

    def build_NN_world(self):
        
        self.load_data()
        self.norm_data_X1,self.norm_data_y1=self.fun.norm_data1(self.dataset1.copy(),self.timestep)
        self.norm_data_X2,self.norm_data_y2=self.fun.norm_data2(self.dataset2.copy(),self.timestep)
        self.norm_data_X3,self.norm_data_y3=self.fun.norm_data3(self.dataset3.copy(),self.timestep)
        self.norm_data_X4,self.norm_data_y4=self.fun.norm_data4(self.dataset4.copy(),self.timestep)
        #X_train, y_train,X_test, y_test=self.fun.norm_and_split_data(self.dataset.copy(),self.timestep)
        #model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,model11,model12,model13,model14,model15,model16=self.NN_world.build_world(X_train, y_train,self.timestep)
        #self.dataset_extended_use,self.df=self.fun.load_extended_data()
        initial = 1
        return initial


    def flow_and_return1(self,valve,supply,room_temp):
        if valve==1:
            flow=0.05#2*0.01667#kg/s#1.666667*10**-5#m3/s
        else:
            flow=0
        diff=supply-(room_temp+20)
        return_temp=supply-(4+(1/2.5)*diff)#supply-((supply-25)/5)
        return flow,return_temp


    def flow_and_return2(self,valve,supply,room_temp):
        if valve==1:
            flow=0.071#2*0.01667#kg/s#1.666667*10**-5#m3/s
        else:
            flow=0
        diff=supply-(room_temp+20)
        return_temp=supply-(4+(1/2.5)*diff)#supply-((supply-25)/5)
        return flow,return_temp

    def flow_and_return3(self,valve,supply,room_temp):
        if valve==1:
            flow=0.0795#2*0.01667#kg/s#1.666667*10**-5#m3/s
        else:
            flow=0
        diff=supply-(room_temp+20)
        return_temp=supply-(7+(1/2.5)*diff)#supply-((supply-25)/5)
        return flow,return_temp

    def flow_and_return4(self,valve,supply,room_temp):
        if valve==1:
            flow=0.065#2*0.01667#kg/s#1.666667*10**-5#m3/s
        else:
            flow=0
        diff=supply-(room_temp+20)
        return_temp=supply-(4+(1/2.5)*diff)#supply-((supply-25)/5)
        return flow,return_temp





    def make_prediction(self,model,Point_in_time_data,number,valve_pos,pre_list,zone_nr):
        #print('point in time data',Point_in_time_data)
        #print('number',number)
        t0 = time.time()
        if number > 6:
            num_for=6
        else:
            num_for=number

        if number > 7:
            num_pre=6
        else:
            num_pre=number-1
        #print('point in time before modi',number)
        #print('point in time before modi',Point_in_time_data)
        #('point in time before',Point_in_time_data)
        #print('number',number)
        for i in range(num_for):
            i=i+1
            #print(i)
            if number > 6:
                i_modi=i+(num_for-6)
            else:
                i_modi=i
            Point_in_time_data[self.timestep-(i)][2]=valve_pos[i_modi-1]
            Point_in_time_data[self.timestep-(i)][3]=self.Supply_action[i_modi-1]
            
        for i in range(num_pre):
            i=i+1
            #print(i)
            if number > 6:
                i_modi=i+(num_pre-6)
            else:
                i_modi=i            
            Point_in_time_data[self.timestep-(i)][0]=pre_list[i_modi-1]
        #print('point in time after modi',Point_in_time_data)
        t1 = time.time()
        #print('t1t0',t1-t0)
        if zone_nr == 1:
            Point_in_time_data=self.fun.norm_x1(Point_in_time_data)
        elif zone_nr == 2:
            Point_in_time_data=self.fun.norm_x2(Point_in_time_data)
        elif zone_nr == 3:
            Point_in_time_data=self.fun.norm_x3(Point_in_time_data)
        else:
            Point_in_time_data=self.fun.norm_x4(Point_in_time_data)

        t2 = time.time()
        #print('t2t1',t2-t1)
        Point_in_time_data=np.expand_dims(Point_in_time_data, axis=0)
        t3 = time.time()
        #print('t3t2',t3-t2)
        #tf.compat.v1.disable_eager_execution()
        if zone_nr == 1:
            prediction=self.serve1(Point_in_time_data,model)#model.predict(Point_in_time_data)
        elif zone_nr == 2:
            prediction=self.serve2(Point_in_time_data,model)
        elif zone_nr == 3:
            prediction=self.serve3(Point_in_time_data,model)
        else:
            prediction=self.serve4(Point_in_time_data,model)
        #prediction=tf.make_tensor_proto(prediction)
        #prediction=tf.make_ndarray(prediction)
        t33 = time.time()
        #print('t33t3',t33-t3)
        return prediction
    
    @tf.function
    def serve1(self,x,model):
        prediction=model(x, training=False)
        return prediction

    @tf.function
    def serve2(self,x,model):
        prediction=model(x, training=False)
        return prediction

    @tf.function
    def serve3(self,x,model):
        prediction=model(x, training=False)
        return prediction

    @tf.function
    def serve4(self,x,model):
        prediction=model(x, training=False)
        return prediction


    def NN_world_send_action(self,action):
        t4 = time.time()
        #print('-------------Enter send action NN_world--------------')
        #Start position 
        if self.NN_world_count == 1:
            self.Valve1_actions=[]
            self.Valve2_actions=[]
            self.Valve3_actions=[]
            self.Valve4_actions=[]
            self.Supply_action=[]
            self.prediction1_list=[]
            self.prediction2_list=[]
            self.prediction3_list=[]
            self.prediction4_list=[]
            len_of_dataset=len(self.norm_data_X1)
            self.start_point_int=self.start_point_int#+500#random.randint(0, len_of_dataset-(self.timestep+20))
            #print(self.start_point_int)
            if self.start_point_int >len_of_dataset-100:
                self.start_point_int=0
        
        self.Valve1_actions.insert(0, self.Valve1)
        self.Valve2_actions.insert(0, self.Valve2)
        self.Valve3_actions.insert(0, self.Valve3)
        self.Valve4_actions.insert(0, self.Valve4)
        self.Supply_action.insert(0, self.temp)
        #print('supply',self.Supply_action)
        #print('valve',self.Valve_actions)
        self.start_point_int=self.start_point_int+1
        point_in_time_int=self.start_point_int#+self.NN_world_count
        # if len(self.Valve_actions)>30:
        #     del self.Valve_actions[-1]
        #     del self.Supply_action[-1]
        #     del self.prediction_list[-1]
        t5 = time.time()    
        #print('t5t4',t5-t4)
        Point_in_time_data1=self.norm_data_X1[point_in_time_int]
        Point_in_time_data2=self.norm_data_X2[point_in_time_int]

        Point_in_time_data3=self.norm_data_X3[point_in_time_int]
        Point_in_time_data4=self.norm_data_X4[point_in_time_int]
        real_val1=self.norm_data_y1[point_in_time_int]
        real_val2=self.norm_data_y2[point_in_time_int]

        real_val3=self.norm_data_y3[point_in_time_int]
        real_val4=self.norm_data_y4[point_in_time_int]
        Point_in_time_data1=self.fun.invers_x1(Point_in_time_data1)
        Point_in_time_data2=self.fun.invers_x2(Point_in_time_data2)

        Point_in_time_data3=self.fun.invers_x3(Point_in_time_data3)
        Point_in_time_data4=self.fun.invers_x4(Point_in_time_data4)
        real_val1=np.expand_dims(real_val1, axis=0)
        real_val2=np.expand_dims(real_val2, axis=0)
        real_val3=np.expand_dims(real_val3, axis=0)
        real_val4=np.expand_dims(real_val4, axis=0)
        real_val1=self.fun.invers_y1(real_val1)
        real_val2=self.fun.invers_y2(real_val2)
        real_val3=self.fun.invers_y3(real_val3)
        real_val4=self.fun.invers_y4(real_val4)
        prediction1=self.make_prediction(self.models1[self.NN_world_count-1],Point_in_time_data1,self.NN_world_count,self.Valve1_actions,self.prediction_list1,1)
        prediction2=self.make_prediction(self.models2[self.NN_world_count-1],Point_in_time_data2,self.NN_world_count,self.Valve2_actions,self.prediction_list2,2)
        prediction3=self.make_prediction(self.models3[self.NN_world_count-1],Point_in_time_data3,self.NN_world_count,self.Valve3_actions,self.prediction_list3,3)
        prediction4=self.make_prediction(self.models4[self.NN_world_count-1],Point_in_time_data4,self.NN_world_count,self.Valve4_actions,self.prediction_list4,4)
        t6 = time.time()    
        #print('t6t5',t6-t5)
        prediction1=self.fun.invers_y1(prediction1)
        prediction2=self.fun.invers_y2(prediction2)
        prediction3=self.fun.invers_y3(prediction3)
        prediction4=self.fun.invers_y4(prediction4)
        prediction1=prediction1[0].tolist()
        prediction2=prediction2[0].tolist()
        prediction3=prediction3[0].tolist()
        prediction4=prediction4[0].tolist()
        prediction1=prediction1[0]
        prediction2=prediction2[0]
        prediction3=prediction3[0]
        prediction4=prediction4[0]
        self.prediction_list1.insert(0, prediction1)
        self.prediction_list2.insert(0, prediction2)
        self.prediction_list3.insert(0, prediction3)
        self.prediction_list4.insert(0, prediction4)



        t7 = time.time()    
        #print('t7t6',t7-t6)

        self.NN_world_count_2=self.NN_world_count_2+1
        self.NN_world_count=self.NN_world_count+1

        if self.NN_world_count==31:
            self.NN_world_count=1
            self.done_nn=1
        elif self.start==1:
            self.done_nn=1
        else:
            self.done_nn=0


        # if done, create data for AI input provider
        free1 = self.userbehavior1(0)#time_of_day set zero
        free2 = self.userbehavior2(0)#time_of_day set zero
        free3 = self.userbehavior3(0)#time_of_day set zero
        free4 = self.userbehavior4(0)#time_of_day set zero
        if self.done_nn==1:
            for i in range(self.timestep):
                out=self.df_out.iloc[point_in_time_int+i]
                #self.input_values = {'SupplyTemperature': self.temp,'Valve1': self.Valve1,'Valve2': self.Valve2,'Valve3': self.Valve3,'Valve4': self.Valve4,'FreeHeatRoom1': free1,'FreeHeatRoom2': free2,'FreeHeatRoom3': free3,'FreeHeatRoom4': free4,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'ValveS1Room2': self.VS12,'ValveS2Room2': self.VS22,'ValveS1Room3': self.VS13,'ValveS2Room3': self.VS23,'ValveS1Room4': self.VS14,'ValveS2Room4': self.VS24,'TempRef':self.tref}
                actions_to_env={'SupplyTemperature': self.dataset1['Supplytemp_t'][point_in_time_int+i],'Valve1': self.df_out['Valveout1'][point_in_time_int+i],'Valve2': self.df_out['Valveout2'][point_in_time_int+i],'Valve3': self.df_out['Valveout3'][point_in_time_int+i],'Valve4': self.df_out['Valveout4'][point_in_time_int+i],'FreeHeatRoom1': free1,'FreeHeatRoom2': free2,'FreeHeatRoom3': free3,'FreeHeatRoom4': free4,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'ValveS1Room2': self.VS12,'ValveS2Room2': self.VS22,'ValveS1Room3': self.VS13,'ValveS2Room3': self.VS23,'ValveS1Room4': self.VS14,'ValveS2Room4': self.VS24,'TempRef':self.tref}
                Hardconstraint_mix=self.df_actions['Hardcon'][point_in_time_int+i]
                #Hardconstraint_V=self.dataset['Hcv'][point_in_time_int+i]
                last_price1=self.df_zone1['Price'][point_in_time_int+i]
                last_price2=self.df_zone2['Price'][point_in_time_int+i]
                last_price3=self.df_zone3['Price'][point_in_time_int+i]
                last_price4=self.df_zone4['Price'][point_in_time_int+i]
                tod=self.dataset1['TOD'][point_in_time_int+i]
                data_mix=self.ai_input_provider.calculate_ai_input_mix(out,actions_to_env,tod,Hardconstraint_mix,last_price1,last_price2,last_price3,last_price4)
                data_v1=self.ai_input_provider.calculate_ai_input_Zone1(out,actions_to_env,tod,last_price1)
                data_v2=self.ai_input_provider.calculate_ai_input_Zone2(out,actions_to_env,tod,last_price2)
                data_v3=self.ai_input_provider.calculate_ai_input_Zone3(out,actions_to_env,tod,last_price3)
                data_v4=self.ai_input_provider.calculate_ai_input_Zone4(out,actions_to_env,tod,last_price4)
        room1_temp=np.float64(prediction1)    
        room2_temp=np.float64(prediction2)    
        room3_temp=np.float64(prediction3)    
        room4_temp=np.float64(prediction4)
        #if self.done_nn==1:
            # print('real val',prediction2)
            # print('real val',real_val2)
            # print('Start point data',Point_in_time_data2)
            # print('Start point',self.start_point_int)
            # print('realval_from,',self.dataset2['Troom'][point_in_time_int+self.timestep+1])
        t8 = time.time()    
        #print('t7t6',t8-t7)    
        self.tod=self.dataset1['TOD'][point_in_time_int+self.timestep+1]
        flow1,return_temp1=self.flow_and_return1(self.Valve1,self.temp,room1_temp)
        flow2,return_temp2=self.flow_and_return2(self.Valve2,self.temp,room2_temp)
        flow3,return_temp3=self.flow_and_return3(self.Valve3,self.temp,room3_temp)
        flow4,return_temp4=self.flow_and_return4(self.Valve4,self.temp,room4_temp)
        self.input_values = {'SupplyTemperature': self.temp,'Valve1': self.Valve1,'Valve2': self.Valve2,'Valve3': self.Valve3,'Valve4': self.Valve4,'FreeHeatRoom1': free1,'FreeHeatRoom2': free2,'FreeHeatRoom3': free3,'FreeHeatRoom4': free4,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'ValveS1Room2': self.VS12,'ValveS2Room2': self.VS22,'ValveS1Room3': self.VS13,'ValveS2Room3': self.VS23,'ValveS1Room4': self.VS14,'ValveS2Room4': self.VS24,'TempRef':self.tref}
        #self.input_values = {'SupplyTemperature': self.temp,'Valve1': self.Valve1,'FreeHeatRoom1': free1,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'TempRef':self.tref}
        self.out={'RoomTemperature1':np.float64(prediction1),'RoomTemperature2':np.float64(prediction2),'RoomTemperature3':np.float64(prediction3),'RoomTemperature4':np.float64(prediction4),'Valveout1':self.Valve1,'Valveout2':self.Valve2,'Valveout3':self.Valve3,'Valveout4':self.Valve4,'Power1':self.df_out['Power1'][point_in_time_int+self.timestep+1],'Power2':self.df_out['Power2'][point_in_time_int+self.timestep+1],'Power3':self.df_out['Power3'][point_in_time_int+self.timestep+1],'Power4':self.df_out['Power4'][point_in_time_int+self.timestep+1],'Tamb1':self.df_out['Tamb1'][point_in_time_int+self.timestep+1],'Sun1':self.df_out['Sun1'][point_in_time_int+self.timestep+1],'Tambforecast1':self.df_out['Tambforecast1'][point_in_time_int+self.timestep+1],'Tambforecast2':self.df_out['Tambforecast2'][point_in_time_int+self.timestep+1],'Tambforecast3':self.df_out['Tambforecast3'][point_in_time_int+self.timestep+1],'Sunforecast1':self.df_out['Sunforecast1'][point_in_time_int+self.timestep+1],'Sunforecast2':self.df_out['Sunforecast2'][point_in_time_int+self.timestep+1],'Sunforecast3':self.df_out['Sunforecast3'][point_in_time_int+self.timestep+1],'Powerwater1':self.df_out['Powerwater1'][point_in_time_int+self.timestep+1],'Powerwater2':self.df_out['Powerwater2'][point_in_time_int+self.timestep+1],'Powerwater3':self.df_out['Powerwater3'][point_in_time_int+self.timestep+1],'Powerwater4':self.df_out['Powerwater4'][point_in_time_int+self.timestep+1],'Flow1':flow1,'Flow2':flow2,'Flow3':flow3,'Flow4':flow4,'Treturn1':return_temp1,'Treturn2':return_temp2,'Treturn3':return_temp3,'Treturn4':return_temp4}
        self.iteration_number1_fin =self.iteration_number1
        #print('send out from NN world',self.out)
        t5 = time.time()
        self.start=0
        if self.NN_world_count_2 == 1200*108:
            self.NN_world_bool =0
        return self.out ,self.input_values,self.tref,self.NN_world_bool,self.tod,self.done_nn
    
    def Hardconstraint_for_mixer(self,action):
        #print('print HARD con',self.Hardconstraint)
        return self.Hardconstraint 


    def Hardconstraint_masked_action(self,action):
        if self.Hardconstraint_bool > 0.1 and self.simulation_time > 0.001 :
            agent_action=action*1.4+30+273.15
            room_dif = abs(self.out['RoomTemperature2']-self.tref)
            #print('enter1',room_dif)
            if self.simulation_time > 150:
                alfa=0.6
            else:
                alfa=0.9
            if room_dif > alfa and self.out['Valveout2'] >0.1:
                #print('enter2',self.out['Valveout2'] )
                self.temp_check1 = -1*(self.out['Tamb1']-273.15)*0.6+44+273.15
                self.Masked_action_maybe=round((self.temp_check1-303.15)/1.4)
            else:
                self.temp_check1 = 0
                self.Masked_action=action
                self.Hardconstraint = 0
                #print('enter3',self.out['Valveout2'] )
            if self.temp_check1 > agent_action+1:
                self.Masked_action = self.Masked_action_maybe
                self.Hardconstraint = 1 + self.Hardconstraint
                #print('enter4',self.temp_check1)
                if self.Hardconstraint < 10:
                    self.Hardconstraint = self.Hardconstraint+10
                    #print('enter5',self.Hardconstraint)
            else:
                self.temp_check1 = 0
                self.Masked_action=action
                self.Hardconstraint = 0
                #print('enter6',action )
            if self.Masked_action > 14:
                self.Masked_action=14
        else:
            self.Masked_action=action
            self.Hardconstraint = 0
            #print('enter0')
        return self.Masked_action

    def Hardconstraint_send_V1(self):
        return self.hard_V1

    def Hardconstraint_send_V2(self):
        return self.hard_V2

    def Hardconstraint_send_V3(self):
        return self.hard_V3

    def Hardconstraint_send_V4(self):
        return self.hard_V4

    def Hardconstraint_for_V1(self,action):
        if self.Hardconstraint_bool > 0.1 and self.simulation_time > 0.05:
            room_dif_no_abs = self.out['RoomTemperature1']-self.tref
            tc=0
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve1=0
                tc=1
            elif room_dif_no_abs < -self.beta and action ==0 and tc==0:
                self.Valve1=1
                tc=1
            else:
                self.Valve1=action
                tc=0

            if tc==1:
                self.V1_count=self.V1_count+1
                self.hard_V1=(self.V1_count+5)*-1
            else:
                self.V1_count=0
                self.hard_V1=0

            Masked_action_V1=self.Valve1
        else:
            Masked_action_V1=action
            self.hard_V1=0
            self.Valve1=action
        return Masked_action_V1

    def Hardconstraint_for_V2(self,action):
        if self.Hardconstraint_bool > 0.1 and self.simulation_time > 0.05:
            room_dif_no_abs = self.out['RoomTemperature2']-self.tref
            tc=0
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve2=0
                tc=1
            elif room_dif_no_abs < -self.beta and action ==0 and tc==0:
                self.Valve2=1
                tc=1
            else:
                self.Valve2=action
                tc=0

            if tc==1:
                self.V2_count=self.V2_count+1
                self.hard_V2=(self.V2_count+5)*-1
            else:
                self.V2_count=0
                self.hard_V2=0

            Masked_action_V2=self.Valve2
        else:
            Masked_action_V2=action
            self.hard_V2=0
        return Masked_action_V2

    def Hardconstraint_for_V3(self,action):
        if self.Hardconstraint_bool > 0.1 and self.simulation_time > 0.05:
            room_dif_no_abs = self.out['RoomTemperature3']-self.tref
            tc=0
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve3=0
                tc=1
            elif room_dif_no_abs < -self.beta and action ==0 and tc==0:
                self.Valve3=1
                tc=1
            else:
                self.Valve3=action
                tc=0

            if tc==1:
                self.V3_count=self.V3_count+1
                self.hard_V3=(self.V3_count+5)*-1
            else:
                self.V3_count=0
                self.hard_V3=0

            Masked_action_V3=self.Valve3
        else:
            Masked_action_V3=action
            self.hard_V3=0
        return Masked_action_V3

    def Hardconstraint_for_V4(self,action):
        if self.Hardconstraint_bool > 0.1 and self.simulation_time > 0.05:
            room_dif_no_abs = self.out['RoomTemperature4']-self.tref
            tc=0
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve4=0
                tc=1 
            elif room_dif_no_abs < -self.beta and action ==0 and tc==0:
                self.Valve4=1
                tc=1
            else:
                self.Valve4=action
                tc=0

            if tc==1:
                self.V4_count=self.V4_count+1
                self.hard_V4=(self.V4_count+5)*-1
            else:
                self.V4_count=0
                self.hard_V4=0

            Masked_action_V4=self.Valve4
        else:
            Masked_action_V4=action
            self.hard_V4=0
        return Masked_action_V4

    def outzone2(self,it_num2):
        if it_num2 == self.iteration_number1_fin:
            out2 = self.out
            actions_to_env=self.input_values
            move=1
        else:
            out2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            actions_to_env=0
            move=0

        return out2 ,actions_to_env,self.tref,self.NN_world_bool,self.tod,self.done_nn,move


    def outzone3(self,it_num3):
        if it_num3 == self.iteration_number1_fin:
            out3 = self.out
            actions_to_env=self.input_values
            move=1
        else:
            out3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            actions_to_env=0
            move=0

        return out3 ,actions_to_env,self.tref,self.NN_world_bool,self.tod,self.done_nn,move

    def outzone4(self,it_num4):
        if it_num4 == self.iteration_number1_fin:
            out4 = self.out
            actions_to_env=self.input_values
            move=1
        else:
            out4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            actions_to_env=0
            move=0

        return out4 ,actions_to_env,self.tref,self.NN_world_bool,self.tod,self.done_nn,move

    def outmixing(self,it_num_mix):
        if self.iteration_number1_fin ==it_num_mix:
            out_mix = self.out
            actions_to_env=self.input_values
            move=1
        else:
            out_mix = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            actions_to_env = 0
            move=0

        return out_mix ,actions_to_env,self.tref,self.NN_world_bool,self.tod,self.done_nn,move


