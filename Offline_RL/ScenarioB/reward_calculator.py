# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# From Aalborg University

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
class RewardCalculator:
    def __init__(self, params):
        self.last_distance1 = 0
        self.last_distance2 = 0
        self.last_distance3 = 0
        self.last_distance4 = 0
        self.params = params
        self.T1 = 0
        self.T2 = 0
        self.ref=273.15+22
        self.T3 = 0
        self.T4 = 0
        self.Tmix = 0
        self.Rewards=[]
        self.plot_count = 0
        self.count = 0
        self.count2 = 0
        self.count3 = 0
        self.count4 = 0
        self.count_list = []
        self.Power_last = 0
        self.Power_last2 =0
        self.Power_last3=0
        self.Power_last4=0
        self.last_Supplytemp=273.15+40
        self.PP_c=0
        self.PP=0
        self.start=0
        self.scop= [2.5,3,3.3,4.1,4.8,5.1]#35
        self.temp = [-15,-7,2,7,12,20]
        self.cost= [0.3,0.28,0.7,0.4,0.8,0.3]#35
        self.Time = [0.0,4,8,12,19,24]
        self.PLF_list_1=[0.1,0.6,0.8,0.9,0.95,1]
        self.PLR_list_1=[0,0.2,0.4,0.6,0.8,1]
        self.PLF_fun= InterpolatedUnivariateSpline(self.PLR_list_1, self.PLF_list_1, k=2)
        self.Heatpump_cop = InterpolatedUnivariateSpline(self.temp, self.scop, k=2)
        self.price_pr_kw = InterpolatedUnivariateSpline(self.Time, self.cost, k=2)  
        self.PLR_list=[]
        self.PLR_list2=[]
        self.PLR_list3=[]
        self.PLR_list4=[]
        
    def calculate_reward_Zone1(self, out,action,actions_to_env,done,NN_world_bool,hard_v1,tod,NN_bool):


        T1 = out['RoomTemperature1']
        T_amb = out['Tamb1']-273.15
        distance1 = abs(self.params.goalT1 - T1)

        if action <= 0:
            action = 0
        elif action > 0:
            action = 1
        #if self.start==0:

        comfort = -(((distance1+1)**2)-1)

#############Power com with mass initial############################Power com with flow and temp###############
        if NN_bool==0:
            Power_c = (out['Powerwater1'])- self.Power_last
            self.Power_last = (out['Powerwater1'])

            #self.PP_c=Power_c+self.PP_c
            Power=Power_c
        else:
            try:
                dt=actions_to_env['SupplyTemperature']-out['Treturn1']
                Power = out['Flow1']*dt*4184*800
            except ValueError:
                dt=5
                Power=0
                print('Value Error recorded')
            

           # self.PP=Power+self.PP

##############Part load Factor###########################

        if out['Valveout1'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1 

        self.PLR_list.append(Valve1)

        if len(self.PLR_list)>10:
            self.PLR_list.pop(0)
        
        A=sum(self.PLR_list)

        PLR=A/len(self.PLR_list)
        PLF=self.PLF_fun(PLR)

        

############Price calculation###########################
        dts=actions_to_env['SupplyTemperature']-(35+273.15)
        #print(PLF)
        cop_correction = dts*-0.02+1
        cop=self.Heatpump_cop(T_amb)*cop_correction*PLF
        taxes=4
        Power_heatpump = Power/cop
        Price =  (Power_heatpump/(3600*1000))*self.price_pr_kw(tod)+taxes*((Power_heatpump/(3600*1000))*self.price_pr_kw(tod))
        
        if Price < 0:
            Price=0


        last_reward = comfort
        #print(price_reward)

############ soft constraint ############

        if distance1 < 0.4 and T1 < self.ref:
            last_reward=last_reward+hard_v1+2
        else:
            last_reward=last_reward+hard_v1

               
        if last_reward < -50:
            last_reward = -50
        return last_reward,Price,Power,hard_v1






    def calculate_reward_Zone2(self, out,action,actions_to_env,done,NN_world_bool,hard_v1,tod,NN_bool):
        #print('ssssssssssss',out)
        #print('kkkkkkkkkkkkkkkk',out['RoomTemperature2'])
        #print('ttttttttttttttttt',type(out['RoomTemperature2']))
        T2 = out['RoomTemperature2']
        
        T_amb = out['Tamb1']-273.15
        distance = abs(self.params.goalT2 - T2)

        if action <= 0:
            action = 0
        elif action > 0:
            action = 1
        #if self.start==0:

        comfort = -(((distance+1)**2)-1)

#############Power com with mass initial############################Power com with flow and temp###############
        if NN_bool==0:
            Power_c = (out['Powerwater2'])- self.Power_last2
            self.Power_last2 = (out['Powerwater2'])

            #self.PP_c=Power_c+self.PP_c
            Power=Power_c
        else:
            try:
                dt=actions_to_env['SupplyTemperature']-out['Treturn2']
                Power = out['Flow2']*dt*4184*800
            except ValueError:
                dt=5
                Power=0
                print('Value Error recorded2')

           # self.PP=Power+self.PP

##############Part load Factor###########################

        if out['Valveout2'] <0.5:
            Valve =  0 
        else:
            Valve =  1 

        self.PLR_list2.append(Valve)

        if len(self.PLR_list2)>10:
            self.PLR_list2.pop(0)
        
        A=sum(self.PLR_list2)

        PLR=A/len(self.PLR_list2)
        PLF=self.PLF_fun(PLR)

        

############Price calculation###########################

        dts=actions_to_env['SupplyTemperature']-(35+273.15)
        #print(PLF)
        cop_correction = dts*-0.02+1
        cop=self.Heatpump_cop(T_amb)*cop_correction*PLF
        taxes=4
        Power_heatpump = Power/cop
        Price =  (Power_heatpump/(3600*1000))*self.price_pr_kw(tod)+taxes*((Power_heatpump/(3600*1000))*self.price_pr_kw(tod))
        
        if Price < 0:
            Price=0


        last_reward = comfort
        #print(price_reward)

############ soft constraint ############

        if distance < 0.4 and T2 < self.ref:
            last_reward=last_reward+hard_v1+2
        else:
            last_reward=last_reward+hard_v1

               
        if last_reward < -50:
            last_reward = -50
        return last_reward,Price,Power,hard_v1



    def calculate_reward_Zone3(self, out,action,actions_to_env,done,NN_world_bool,hard_v1,tod,NN_bool):


        T = out['RoomTemperature3']
        T_amb = out['Tamb1']-273.15
        distance = abs(self.params.goalT3 - T)

        if action <= 0:
            action = 0
        elif action > 0:
            action = 1
        #if self.start==0:

        comfort = -(((distance+1)**2)-1)

#############Power com with mass initial############################Power com with flow and temp###############
        if NN_bool==0:
            Power_c = (out['Powerwater3'])- self.Power_last3
            self.Power_last3 = (out['Powerwater3'])

            #self.PP_c=Power_c+self.PP_c
            Power=Power_c
        else:
            try:
                dt=actions_to_env['SupplyTemperature']-out['Treturn3']
                Power = out['Flow3']*dt*4184*800
            except ValueError:
                dt=5
                Power=0
                print('Value Error recorded3')

           # self.PP=Power+self.PP

##############Part load Factor###########################

        if out['Valveout3'] <0.5:
            Valve =  0 
        else:
            Valve =  1 

        self.PLR_list3.append(Valve)

        if len(self.PLR_list3)>10:
            self.PLR_list3.pop(0)
        
        A=sum(self.PLR_list3)

        PLR=A/len(self.PLR_list3)
        PLF=self.PLF_fun(PLR)

        

############Price calculation###########################

        dts=actions_to_env['SupplyTemperature']-(35+273.15)
        #print(PLF)
        cop_correction = dts*-0.02+1
        cop=self.Heatpump_cop(T_amb)*cop_correction*PLF
        taxes=4
        Power_heatpump = Power/cop
        Price =  (Power_heatpump/(3600*1000))*self.price_pr_kw(tod)+taxes*((Power_heatpump/(3600*1000))*self.price_pr_kw(tod))
        
        if Price < 0:
            Price=0


        last_reward = comfort
        #print(price_reward)

############ soft constraint ############

        if distance < 0.4 and T < self.ref:
            last_reward=last_reward+hard_v1+2
        else:
            last_reward=last_reward+hard_v1

               
        if last_reward < -50:
            last_reward = -50
        return last_reward,Price,Power,hard_v1





    def calculate_reward_Zone4(self, out,action,actions_to_env,done,NN_world_bool,hard_v1,tod,NN_bool):


        T = out['RoomTemperature4']
        T_amb = out['Tamb1']-273.15
        distance = abs(self.params.goalT4 - T)

        if action <= 0:
            action = 0
        elif action > 0:
            action = 1
        #if self.start==0:

        comfort = -(((distance+1)**2)-1)

#############Power com with mass initial############################Power com with flow and temp###############
        if NN_bool==0:
            Power_c = (out['Powerwater4'])- self.Power_last4
            self.Power_last4 = (out['Powerwater4'])

            #self.PP_c=Power_c+self.PP_c
            Power=Power_c
        else:
            try:
                dt=actions_to_env['SupplyTemperature']-out['Treturn4']
                Power = out['Flow4']*dt*4184*800
            except ValueError:
                dt=5
                Power=0
                print('Value Error recorded4')

           # self.PP=Power+self.PP

##############Part load Factor###########################

        if out['Valveout4'] <0.5:
            Valve =  0 
        else:
            Valve =  1 

        self.PLR_list4.append(Valve)

        if len(self.PLR_list4)>10:
            self.PLR_list4.pop(0)
        
        A=sum(self.PLR_list4)

        PLR=A/len(self.PLR_list4)
        PLF=self.PLF_fun(PLR)

        

############Price calculation###########################

        dts=actions_to_env['SupplyTemperature']-(35+273.15)
        #print(PLF)
        cop_correction = dts*-0.02+1
        cop=self.Heatpump_cop(T_amb)*cop_correction*PLF
        taxes=4
        Power_heatpump = Power/cop
        Price =  (Power_heatpump/(3600*1000))*self.price_pr_kw(tod)+taxes*((Power_heatpump/(3600*1000))*self.price_pr_kw(tod))
        
        if Price < 0:
            Price=0


        last_reward = comfort
        #print(price_reward)

############ soft constraint ############

        if distance < 0.4 and T < self.ref:
            last_reward=last_reward+hard_v1+2
        else:
            last_reward=last_reward+hard_v1

               
        if last_reward < -50:
            last_reward = -50
        return last_reward,Price,Power,hard_v1





    def calculate_reward_mix(self, out,Hardconstraint,tref,actions_to_env,last_price1,last_price2,last_price3,last_price4):        
        T1 = out['RoomTemperature1']
        T2 = out['RoomTemperature2']
        T3 = out['RoomTemperature3']
        T4 = out['RoomTemperature4']
        distance1 = abs(tref - T1)
        distance1_no_abs = tref - T1

        distance2 = abs(tref - T2)
        distance2_no_abs = tref - T2

        distance3 = abs(tref - T3)
        distance3_no_abs = tref - T3

        distance4 = abs(tref - T4)
        distance4_no_abs = tref - T4

        last_reward1 =-(((distance1+1)**1.3)-1)# -distance1#
        last_reward2 =-(((distance2+1)**1.3)-1)# -distance1#
        last_reward3 =-(((distance3+1)**1.3)-1)# -distance1#
        last_reward4 =-(((distance4+1)**1.3)-1)# -distance1#

        if distance1_no_abs < 0.4 and distance1_no_abs > 0  and out['Valveout1'] > 0.1:
            mixing_reward1 = 2-last_reward1
        else:
            mixing_reward1 = last_reward1

        if distance2_no_abs < 0.4 and distance2_no_abs > 0  and out['Valveout2'] > 0.1:
            mixing_reward2 = 2-last_reward2
        else:
            mixing_reward2 = last_reward2

        if distance3_no_abs < 0.4 and distance3_no_abs > 0  and out['Valveout3'] > 0.1:
            mixing_reward3 = 2-last_reward3
        else:
            mixing_reward3 = last_reward3
        
        if distance4_no_abs < 0.4 and distance4_no_abs > 0  and out['Valveout4'] > 0.1:
            mixing_reward4 = 2-last_reward4
        else:
            mixing_reward4 = last_reward4



        change_in_Supply_temp =abs(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)**3))*-0.003
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        sum_price=last_price1+last_price2+last_price3+last_price4
        if sum_price < 0:
            sum_price=0.1     
        #(6-(sum_price)*2.5)+last_reward1+last_reward2
        price_reward=(3-(sum_price)*2)
        sum_mix_reward=mixing_reward1+mixing_reward2+mixing_reward3+mixing_reward4#(last_reward1+last_reward2+last_reward3+last_reward4)*2#mixing_reward1+mixing_reward2
        supply=((np.float64(actions_to_env['SupplyTemperature'])-(273.15+30))**3)*0.0008
        mixing_reward =sum_mix_reward-Hardconstraint+price_reward +change_in_Supply_temp-supply# -((np.float64(actions_to_env['SupplyTemperature'])-30)/30)**1.5#+(0.5-(np.float64(actions_to_env['SupplyTemperature'])/100))*2 #mixing_reward1#-change_in_Supply_temp
        #mixing_reward=mixing_reward1+mixing_reward2+mixing_reward3+mixing_reward4
        return mixing_reward, price_reward,sum_mix_reward,change_in_Supply_temp



