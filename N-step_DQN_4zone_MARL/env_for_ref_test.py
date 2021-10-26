#Master for collecting all actions before sending collected action to FMU(dymola model)
# 
import time
import numpy as np
from fmu_stepping import fmu_stepping
from fmpy.util import plot_result
import csv
from ai_input_provider import AiInputProvider
from reward_calculator import RewardCalculator
from parameters import Params
from scipy.interpolate import InterpolatedUnivariateSpline



class Envmasterscript: 
    def __init__(self, *args, **kwargs):
        self.power= np.array([0.15,0.15,0.15,0.15,0.15,0.17,0.31,0.69,0.76,0.95,0.35,0.15,0.15,0.15,0.19,0.28,0.35,0.51,0.75,0.90,0.80,0.45,0.38,0.15])
        self.KL=np.array([0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,3,3,3,3,3,0,0])*100+self.power*(500*0.4)
        self.BR1=np.array([2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2])*100+self.power*(500*0.2)
        self.BR2=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1])*100+self.power*(500*0.2)
        self.WC=np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*100+self.power*(500*0.2)
        self.iteration_number = 0
        self.iteration_number1 = 0
        self.iteration_number2 = 0
        self.last_Supplytemp=273.15+45
        self.iteration_number3 = 0
        self.iteration_number4 = 0
        self.iteration_number_mix = 0
        self.iteration_number1_fin = 0
        self.dd = 0
        self.temp = 10+35+273.15
        self.yoda =1
        self.zod =1
        self.pumba =1
        self.spock =1
        self.tref=273.15+22
        self.Power_last1 = 0
        self.Power_last2 = 0
        self.Power_last3 = 0
        self.Power_last4 = 0
        self.datalistz1 =[]
        self.datalistz2 =[]
        self.datalistz3 =[]
        self.datalistz4 =[]
        self.datalistmix =[]
        self.n=0
        self.nn=0
        self.n1=0
        self.nn1=0
        self.n2=0
        self.count=0
        self.nn2=0
        self.n3=0
        self.nn3=0
        self.n4=0
        self.nn4=0
        filename1 = "output_zone1_ref.csv"
        filename2 = "output_zone2_ref.csv"
        filename3 = "output_zone3_ref.csv"
        filename4 = "output_zone4_ref.csv"
        filename5 = "output_actions_ref.csv"
        f1 = open(filename1, "w+")
        f2 = open(filename2, "w+")
        f3 = open(filename3, "w+")
        f4 = open(filename4, "w+")
        f5 = open(filename5, "w+")
        f1.close()  
        f2.close()
        f3.close()  
        f4.close()
        f5.close() 
        self.supplyagent = 1 # if 0 supply=45DC if 1 supply=action +35 DC
        self.VS11= 1#switch 1 room 1 if 1 hyscontrol if 0 AI  ----- VS21 needs to be 1
        self.VS21= 1#switch 2 room 1 if 1 VS11 works if 0 AI/withhys
        self.VS12= 1#switch 1 room 2
        self.VS22= 1#switch 2 room 2
        self.VS13= 1#switch 1 room 2
        self.VS23= 1#switch 2 room 2
        self.VS14= 1#switch 1 room 2
        self.VS24= 1#switch 2 room 2
        #start_time = time.clock()
        filename = 'TwoElementHouse_04room_0FMU_0weatherforcast_Houses_testHouse_0hyscontrol.fmu'
        start_time=0.0
        stop_time=3155692600
        self.sample_time=800
        parameters={}
        input={'SupplyTemperature','Valve1','Valve2','Valve3','Valve4','FreeHeatRoom1','FreeHeatRoom2','FreeHeatRoom3','FreeHeatRoom4','ValveS1Room1','ValveS2Room1','ValveS1Room2','ValveS2Room2','ValveS1Room3','ValveS2Room3','ValveS1Room4','ValveS2Room4','TempRef1','TempRef2','TempRef3','TempRef4'}
        self.input_values ={'SupplyTemperature': 340,'Valve1': 1,'Valve2': 1,'Valve3': 1,'Valve4': 1,'FreeHeatRoom1': 0,'FreeHeatRoom2': 0,'FreeHeatRoom3': 0,'FreeHeatRoom4': 0,'ValveS1Room1': 0,'ValveS2Room1': 0,'ValveS1Room2': 0,'ValveS2Room2': 0,'ValveS1Room3': 0,'ValveS2Room3': 0,'ValveS1Room4': 0,'ValveS2Room4': 0,'TempRef1':273.15+22,'TempRef2':273.15+21.8,'TempRef3':273.15+21.9,'TempRef4':273.15+22}
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
        free2 = 0#self.BR1[TOD]
        return free2
    def userbehavior3(self,timeofday):

        TOD = round(timeofday)
        if TOD > 23:
            TOD = 23
        free3 = 0#self.BR2[TOD]
        return free3
    def userbehavior4(self,timeofday):

        TOD = round(timeofday)
        if TOD > 23:
            TOD = 23
        free4 = 0#self.WC[TOD]
        return free4

    def nightsetback(self,timeofday):
        TOD = round(timeofday)
        if TOD >22 or TOD < 6:
            self.tref = 273.15+18
        else:
            self.tref = 273.15+22
        return self.tref

    def Tsupply(self,action):
        if self.supplyagent == 1:
            self.temp = action+35+273.15# descret
        else:
            self.temp = 10+35+273.15
        return self.iteration_number_mix

    def sendaction(self,dd):
        self.iteration_number = self.iteration_number+1
        simulation_time = self.iteration_number*self.sample_time/86400
        timeofday = (simulation_time-(simulation_time//1))*24
        self.count=self.count+1
        if self.count == 108:
            print('time in simulation', simulation_time)
            self.count=0
        free1 = self.userbehavior1(timeofday)
        free2 = self.userbehavior2(timeofday)
        free3 = self.userbehavior1(timeofday)
        free4 = self.userbehavior2(timeofday)
        #self.tref = self.nightsetback(timeofday)
        self.input_values = {'SupplyTemperature': self.temp,'Valve1': self.yoda,'Valve2': self.zod,'Valve3': self.pumba,'Valve4': self.spock,'FreeHeatRoom1': free1,'FreeHeatRoom2': free2,'FreeHeatRoom3': free3,'FreeHeatRoom4': free4,'ValveS1Room1': self.VS11,'ValveS2Room1': self.VS21,'ValveS1Room2': self.VS12,'ValveS2Room2': self.VS22,'ValveS1Room3': self.VS13,'ValveS2Room3': self.VS23,'ValveS1Room4': self.VS14,'ValveS2Room4': self.VS24,'TempRef1':273.15+22,'TempRef2':273.15+21.8,'TempRef3':273.15+21.9,'TempRef4':273.15+22}
        #print('this is action', self.input_values)
        self.out = self.env.step(self.input_values)
        out = self.out
        actions_to_env=self.input_values
        self.temp = -1*(out['Tamb1']-273.15)*0.6+42+273.15
        scop= [2.5,3,3.3,4.1,4.8,5.1]#35
        temp = [-15,-7,2,7,12,20]
        Heatpump_cop = InterpolatedUnivariateSpline(temp, scop, k=1) 

        # T1 = out['RoomTemperature1']
        # T2 = out['RoomTemperature2']
        # T3 = out['RoomTemperature3']
        # T4 = out['RoomTemperature4']
        T_amb = out['Tamb1']-273.15

        # distance1 = abs(273.15+22 - T1)
        # distance2 = abs(273.15+22 - T2)
        # distance3 = abs(273.15+22 - T3)
        # distance4 = abs(273.15+22 - T4)

 

        Power1 = (out['Powerwater1'])- self.Power_last1
        self.Power_last1 = (out['Powerwater1'])
        
        Power2 = (out['Powerwater2'])- self.Power_last2
        self.Power_last2 = (out['Powerwater2'])

        Power3 = (out['Powerwater3'])- self.Power_last3
        self.Power_last3 = (out['Powerwater3'])

        Power4 = (out['Powerwater4'])- self.Power_last4
        self.Power_last4 = (out['Powerwater4'])

        dts=actions_to_env['SupplyTemperature']-(35+273.15)
        cop_correction = dts*-0.02+1

        cop=Heatpump_cop(T_amb)*cop_correction

        Power_heatpump1 = Power1/cop

        Power_heatpump2 = Power2/cop
        Power_heatpump3 = Power3/cop
        Power_heatpump4 = Power4/cop
        Price1 =  Power_heatpump1/550000
        Price2 =  Power_heatpump2/550000
        Price3 =  Power_heatpump3/550000
        Price4 =  Power_heatpump4/550000
        action =1





        T1 = out['RoomTemperature1']
        distance1 = abs(self.tref - T1)
        distance1_no_abs = self.tref - T1
        T2 = out['RoomTemperature2']
        distance2 = abs(self.tref - T2)
        distance2_no_abs = self.tref - T2
        T3 = out['RoomTemperature3']
        distance3 = abs(self.tref - T3)
        distance3_no_abs = self.tref - T3
        T4 = out['RoomTemperature4']
        distance4 = abs(self.tref - T4)
        distance4_no_abs = self.tref - T4

        comfort1 = -(((distance1+1)**2)-1)
        comfort2 = -(((distance2+1)**2)-1)
        comfort3 = -(((distance3+1)**2)-1)
        comfort4 = -(((distance4+1)**2)-1)

        if distance1 < 0.4 and T1 < 273.15+22:
            reward1=comfort1+2#+price_reward
        else:
            reward1=comfort1
               
        if reward1 < -50:
            reward1 = -50



        if distance2 < 0.4 and T2 < 273.15+22:
            reward2=comfort2+2#price_reward+
        else:
            reward2=comfort2
  
        if reward2 < -50:
            reward2 = -50


        if distance3 < 0.4 and T3 < 273.15+22:
            reward3=comfort3+2#price_reward+
        else:
            reward3=comfort3
  
        if reward3 < -50:
            reward3 = -50


        if distance4 < 0.4 and T4 < 273.15+22:
            reward4=comfort4+2#price_reward+
        else:
            reward4=comfort4
  
        if reward4 < -50:
            reward4 = -50

        last_reward1 =-(((distance1+1)**1.3)-1)# -distance1#
        last_reward2 =-(((distance2+1)**1.3)-1)# -distance2#
        last_reward3 =-(((distance3+1)**1.3)-1)# -distance1#
        last_reward4 =-(((distance4+1)**1.3)-1)# -distance2#

        if distance1_no_abs < 0.4 and distance1_no_abs > 0  and out['Valveout1'] > 0.1:
            mixing_reward1 = 2-last_reward1
        else:
            mixing_reward1 = last_reward1

        if distance2_no_abs < 0.4 and distance2_no_abs > 0 and out['Valveout2'] > 0.1:
            mixing_reward2 = 2-last_reward2
        else:
            mixing_reward2 = last_reward2

        if distance3_no_abs < 0.4 and distance3_no_abs > 0 and out['Valveout3'] > 0.1:
            mixing_reward3 = 2-last_reward3
        else:
            mixing_reward3 = last_reward3

        if distance4_no_abs < 0.4 and distance4_no_abs > 0 and out['Valveout4'] > 0.1:
            mixing_reward4 = 2-last_reward4
        else:
            mixing_reward4 = last_reward4


        change_in_Supply_temp =abs(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)**3))*-0.003
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        sum_price=Price1+Price2+Price3+Price4
        if sum_price < 0:
            sum_price=0.1     
        #(6-(sum_price)*2.5)+last_reward1+last_reward2
        price_reward=(3-(sum_price)*2)
        sum_mix_reward=mixing_reward1+mixing_reward2+mixing_reward3+mixing_reward4
        supply=((np.float64(actions_to_env['SupplyTemperature'])-(273.15+30))**3)*0.0008
        mixing_reward =price_reward +sum_mix_reward+change_in_Supply_temp-supply

        Hardconstraint=0
        data1 = out['RoomTemperature1'].tolist(),out['Tamb1'].tolist(),out['Valveout1'].tolist(),out['Power1'].tolist(), reward1,action,Price1,Power1,out['Powerwater1'].tolist(),simulation_time
        data2 = out['RoomTemperature2'].tolist(),out['Tamb1'].tolist(),out['Valveout2'].tolist(),out['Power2'].tolist(), reward2,action,Price2,Power2,out['Powerwater2'].tolist(),simulation_time
        data3 = out['RoomTemperature3'].tolist(),out['Tamb1'].tolist(),out['Valveout3'].tolist(),out['Power3'].tolist(), reward3,action,Price3,Power3,out['Powerwater3'].tolist(),simulation_time
        data4 = out['RoomTemperature4'].tolist(),out['Tamb1'].tolist(),out['Valveout4'].tolist(),out['Power4'].tolist(), reward4,action,Price4,Power4,out['Powerwater4'].tolist(),simulation_time
 
        data_mix = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),mixing_reward,Hardconstraint,price_reward,sum_mix_reward,change_in_Supply_temp
        self.savefun_Zone1(data1)
        self.savefun_Zone2(data2)
        self.savefun_Zone3(data3)
        self.savefun_Zone4(data4)
        self.savefun_actiondata(data_mix)
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])

    def savefun_Zone1(self,data):
            
        self.datalistz1.append(data)
        if self.nn1 == 200:
            with open("output_zone1_ref.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz1)
            resultFile.close()
            self.datalistz1 =[]
            self.nn1 =0
                
                
        self.n1 =self.n1 +1
        self.nn1 =self.nn1 +1

    def savefun_Zone2(self,data):
            
        self.datalistz2.append(data)

        if self.nn2 == 200:
            with open("output_zone2_ref.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz2)
            resultFile.close()
            self.datalistz2 =[]
            self.nn2 =0
                
                
        self.n2 =self.n2 +1
        self.nn2 =self.nn2 +1

    def savefun_Zone3(self,data):
            
        self.datalistz3.append(data)

        if self.nn3 == 200:
            with open("output_zone3_ref.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz3)
            resultFile.close()
            self.datalistz3 =[]
            self.nn3 =0
                
                
        self.n3 =self.n3 +1
        self.nn3 =self.nn3 +1

    def savefun_Zone4(self,data):
            
        self.datalistz4.append(data)

        if self.nn4 == 200:
            with open("output_zone4_ref.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz4)
            resultFile.close()
            self.datalistz4 =[]
            self.nn4 =0
                
                
        self.n4 =self.n4 +1
        self.nn4 =self.nn4 +1


    def savefun_actiondata(self,data):
        
        self.datalistmix.append(data)

        if self.nn == 200:
            with open("output_actions_ref.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistmix)
            resultFile.close()
            self.datalistmix =[]
            self.nn =0
            
            
        self.n =self.n +1
        self.nn =self.nn +1





