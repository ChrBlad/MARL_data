#Master for collecting all actions before sending collected action to FMU(dymola model)
# 
import time
import numpy as np
from fmu_stepping import fmu_stepping
from fmpy.util import plot_result
from ai_input_provider import AiInputProvider
from reward_calculator import RewardCalculator
from parameters import Params
import random


class Envmasterscript:    
    def __init__(self):
        params = Params
        self.power= np.array([0.15,0.15,0.15,0.15,0.15,0.17,0.31,0.69,0.76,0.95,0.35,0.15,0.15,0.15,0.19,0.28,0.35,0.51,0.75,0.90,0.80,0.45,0.38,0.15])
        self.KL=np.array([0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,3,3,3,3,3,0,0])*100+self.power*(500*0.4)
        self.BR1=np.array([2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2])*100+self.power*(500*0.2)
        self.BR2=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1])*100+self.power*(500*0.2)
        self.WC=np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*100+self.power*(500*0.2)
        self.iteration_number = 0
        self.iteration_number1 = 0
        self.iteration_number2 = 0
        self.iteration_number3 = 0
        self.V1_count=0
        self.V2_count=0
        self.V3_count=0
        self.V4_count=0
        self.hard_V1=0
        self.hard_V2=0
        self.hard_V3=0
        self.hard_V4=0
        self.iteration_number4 = 0
        self.iteration_number_mix = 0
        self.iteration_number1_fin = 0
        self.Hardconstraint = 0
        self.print_number = 0
        self.temp_pre =273.15+40
        self.temp=273.15+40
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
        self.Valve1=0
        self.Valve2=0
        self.Valve3=0
        self.Valve4=0
        self.beta=0.6
        self.yoda = 0
        self.zod = 0
        self.pumba = 0
        self.spock = 0
        self.supplyagent = 0 # if 0 supply=45DC if 1 supply=action +35 DC
        self.VS11= 1#switch 1 room 1 if 1 hyscontrol if 0 AI  ----- VS21 needs to be 1
        self.VS21= 1#switch 2 room 1 if 1 VS11 works if 0 AI/withhys
        self.VS12= 1#switch 1 room 2
        self.VS22= 1#switch 2 room 2
        self.VS13= 1#switch 1 room 2
        self.VS23= 1#switch 2 room 2
        self.VS14= 1#switch 1 room 2
        self.VS24= 1#switch 2 room 2
        #start_time = time.clock()
        filename ='TwoElementHouse_04room_0FMU_0weatherforcast_Houses_testHouse_0hyscontrol.fmu' 
        start_time=0.0
        stop_time=3155692600
        self.sample_time=800
        parameters={}
        input={'SupplyTemperature','Valve1','Valve2','Valve3','Valve4','FreeHeatRoom1','FreeHeatRoom2','FreeHeatRoom3','FreeHeatRoom4','ValveS1Room1','ValveS2Room1','ValveS1Room2','ValveS2Room2','ValveS1Room3','ValveS2Room3','ValveS1Room4','ValveS2Room4','TempRef'}
        self.input_values ={'SupplyTemperature': 340,'Valve1': 1,'Valve2': 1,'Valve3': 1,'Valve4': 1,'FreeHeatRoom1': 0,'FreeHeatRoom2': 0,'FreeHeatRoom3': 0,'FreeHeatRoom4': 0,'ValveS1Room1': 0,'ValveS2Room1': 0,'ValveS1Room2': 0,'ValveS2Room2': 0,'ValveS1Room3': 0,'ValveS2Room3': 0,'ValveS1Room4': 0,'ValveS2Room4': 0,'TempRef':273.15+22}
        output={'RoomTemperature1','RoomTemperature2','RoomTemperature3','RoomTemperature4','Valveout1','Valveout2','Valveout3','Valveout4','Power1','Power2','Power3','Power4','Tamb1','Sun1','Tambforecast1','Tambforecast2','Tambforecast3','Sunforecast1','Sunforecast2','Sunforecast3','Powerwater1','Powerwater2','Powerwater3','Powerwater4'}
        FMU_stepping = fmu_stepping(filename = filename,
                            start_time=start_time,
                            stop_time=stop_time,
                            sample_time=self.sample_time,
                            parameters=parameters,
                            input=input,
                            output=output)
        self.env = FMU_stepping

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

    # def Mask_action_supply(self,action):
    #     Masked_action = 
    #     return Masked_action




    def zone(self,action):
        self.iteration_number1 = 1+self.iteration_number1 
        self.iteration_number_mix = 1+self.iteration_number_mix
        
        if action == 0:
            self.yoda = 0
            self.zod = 0
            self.pumba = 0
            self.spock = 0
        elif action==1:
            self.yoda = 1
            self.zod = 0
            self.pumba = 0
            self.spock = 0
        elif action==2:
            self.yoda = 0
            self.zod = 1
            self.pumba = 0
            self.spock = 0
        elif action==3:
            self.yoda = 0
            self.zod = 0
            self.pumba = 1
            self.spock = 0
        elif action==4:
            self.yoda = 0
            self.zod = 0
            self.pumba = 0
            self.spock = 1
        elif action==5:
            self.yoda = 1
            self.zod = 1
            self.pumba = 0
            self.spock = 0
        elif action==6:
            self.yoda = 1
            self.zod = 0
            self.pumba = 1
            self.spock = 0
        elif action==7:
            self.yoda = 1
            self.zod = 0
            self.pumba = 0
            self.spock = 1
        elif action==8:
            self.yoda = 0
            self.zod = 1
            self.pumba = 1
            self.spock = 0
        elif action==9:
            self.yoda = 0
            self.zod = 1
            self.pumba = 0
            self.spock = 1
        elif action==10:
            self.yoda = 0
            self.zod = 0
            self.pumba = 1
            self.spock = 1
        elif action==11:
            self.yoda = 1
            self.zod = 1
            self.pumba = 1
            self.spock = 0
        elif action==12:
            self.yoda = 1
            self.zod = 1
            self.pumba = 0
            self.spock = 1
        elif action==13:
            self.yoda = 1
            self.zod = 0
            self.pumba = 1
            self.spock = 1
        elif action==14:
            self.yoda = 0
            self.zod = 1
            self.pumba = 1
            self.spock = 1
        elif action==15:
            self.yoda = 1
            self.zod = 1
            self.pumba = 1
            self.spock = 1
        else:
            self.yoda =self.yoda 
            self.zod = self.zod
            self.pumba = self.pumba
            self.spock = self.spock
        return self.iteration_number1,self.iteration_number_mix


    def Tsupply(self,action):
        if self.supplyagent == 1:
            self.temp =action*1.2+30+273.15# descretaction*7.5+37.5+273.15# Box actionaction*1+30+273.15# descret
        elif self.iteration_number_mix < 2:
            self.temp =273.15+45
        else:
            self.temp = -1*(self.out['Tamb1']-273.15)*0.6+42+273.15


    def sendaction (self):
        if  self.iteration_number1 == self.iteration_number_mix:
            self.iteration_number = 1+self.iteration_number 
            self.simulation_time = self.iteration_number*self.sample_time/86400
            timeofday = (self.simulation_time-(self.simulation_time//1))*24
            #self.temp = 10+35+273.15
            #self.Valve2=1
            free1 = self.userbehavior1(timeofday)
            free2 = self.userbehavior2(timeofday)
            free3 = self.userbehavior3(timeofday)
            free4 = self.userbehavior4(timeofday)
            self.input_values = {'SupplyTemperature': self.temp,'Valve1': self.Valve1,'Valve2': self.Valve2,'Valve3': self.Valve3,'Valve4': self.Valve4,'FreeHeatRoom1': free1,'FreeHeatRoom2': free2,'FreeHeatRoom3': free3,'FreeHeatRoom4': free4,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'ValveS1Room2': self.VS12,'ValveS2Room2': self.VS22,'ValveS1Room3': self.VS13,'ValveS2Room3': self.VS23,'ValveS1Room4': self.VS14,'ValveS2Room4': self.VS24,'TempRef':self.tref}
            self.out = self.env.step(self.input_values)
            self.iteration_number1_fin =self.iteration_number1
            actions_to_env=self.input_values
            self.dd=0
            self.print_number = self.print_number+1
        else:
            actions_to_env=0
            print('sss')
        if self.print_number > 107:
            self.print_number =0
            print('---------------------time in simulation---------------------------------', self.simulation_time)
        return self.out ,actions_to_env,self.tref
    
    def Hardconstraint_for_mixer(self):
        return self.Hardconstraint 


    def Hardconstraint_masked_action(self,action):
        if action > 15:
            action= action-16
            if self.Hardconstraint_bool > 0.1 and self.simulation_time > 1 :
                room_dif = abs(self.out['RoomTemperature2']-self.tref)
                if self.simulation_time > 150:
                    alfa=0.6
                else:
                    alfa=1.2
                if room_dif > alfa and self.out['Valveout2'] >0.1:
                    self.temp_check1 = -1*(self.out['Tamb1']-273.15)*0.6+44+273.15
                    self.Masked_action_maybe=round((self.temp_check1-303.15)/1.2)
                else:
                    self.temp_check1 = 0
                    self.Masked_action=action
                    self.Hardconstraint = 0
                if self.temp_check1 > self.temp+1:
                    self.Masked_action = self.Masked_action_maybe
                    self.Hardconstraint = 1 + self.Hardconstraint
                    if self.Hardconstraint < 10:
                        self.Hardconstraint = self.Hardconstraint+10

                    if self.Hardconstraint > 50:
                        self.Hardconstraint=50
                if self.Masked_action > 14:
                    self.Masked_action=14
                    self.Tsupply(self.Masked_action)
                else:
                    self.Masked_action=self.Masked_action
                    self.Tsupply(self.Masked_action)
            else:
                self.Masked_action=action
                self.Hardconstraint = 0
                self.Tsupply(self.Masked_action)
        #else:
            #self.Masked_action=action
            #self.Tsupply(self.Masked_action)
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
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve1=0
                tc=1
            else:
                self.Valve1=action
                tc=0
            if tc==0:
                if room_dif_no_abs < -self.beta and action ==0:
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
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve2=0
                tc=1
            else:
                self.Valve2=action
                tc=0
            if tc==0:
                if room_dif_no_abs < -self.beta and action ==0:
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
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve3=0
                tc=1
            else:
                self.Valve3=action
                tc=0
            if tc==0:
                if room_dif_no_abs < -self.beta and action ==0:
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
            if room_dif_no_abs > self.beta and action ==1:
                self.Valve4=0
                tc=1
            else:
                self.Valve4=action
                tc=0
            if tc==0:
                if room_dif_no_abs < -self.beta and action ==0:
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

    def get_masked_action(self,selected_action):
        self.Masked_action_V1=self.Hardconstraint_for_V1(self.yoda)
        self.Masked_action_V2=self.Hardconstraint_for_V2(self.zod)
        self.Masked_action_V3=self.Hardconstraint_for_V3(self.pumba)
        self.Masked_action_V4=self.Hardconstraint_for_V4(self.spock)
        Masked_temp_action= self.Hardconstraint_masked_action(selected_action)

        if selected_action > 15:
            true_action = Masked_temp_action+15
        else:
            true_action = self.invers_valve_action()
        return true_action


    def invers_valve_action(self):
        if self.Masked_action_V1 == 0 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==0:
            true_action = 0
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==0:
            true_action = 1
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==0:
            true_action = 2
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==0:
            true_action = 3
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==1:
            true_action = 4
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==0:
            true_action = 5
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==0:
            true_action = 6
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==1:
            true_action = 7
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==0:
            true_action = 8
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==1:
            true_action = 9
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==1:
            true_action = 10
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==0:
            true_action = 11
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==0 and self.Masked_action_V4 ==1:
            true_action = 12
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 0 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==1:
            true_action = 13
        elif self.Masked_action_V1 == 0 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==1:
            true_action = 14
        elif self.Masked_action_V1 == 1 and self.Masked_action_V2 == 1 and self.Masked_action_V3 ==1 and self.Masked_action_V4 ==1:
            true_action = 15
        return true_action