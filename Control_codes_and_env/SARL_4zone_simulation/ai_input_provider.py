# This code have been programmed by Christian Blad
import time
import numpy as np
import csv

# Environments
SHTL1, SHTL2, SHTL3, SETL1, SETL2, SETL3, ETL1, ETL2, ETL3 = ("shtl1", "shtl2", "shtl3", "setl1", "setl2", "setl3", "etl1", "etl2", "etl3")


class AiInputProvider:
    def __init__(self, params):
        self.params = params
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0
        self.count=0
        self.Tmix = 0
        self.last_T1 = 0
        self.last_T2 = 0
        self.last_T3 = 0
        self.last_T4 = 0
        self.last2_T1 = 0
        self.last2_T2=0
        self.last2_T3 = 0
        self.last2_T4=0
        self.last3_T1 =0 
        self.last3_T2 = 0
        self.last3_T3 =0 
        self.last3_T4 = 0

        self.T_amb = 0
        self.datalistz1 =[]
        self.datalistz2 =[]
        self.datalistz3 =[]
        self.datalistz4 =[]
        self.datalistmix =[]
        self.last_Supplytemp=273.15+40
        self.state_list=[]
        self.n=0
        self.nn=0
        self.n1=0
        self.nn1=0
        self.n2=0
        self.nn2=0
        self.n3=0
        self.nn3=0
        self.n4=0
        self.nn4=0
        filename1 = "output_zone1.csv"
        filename2 = "output_zone2.csv"
        filename3 = "output_zone3.csv"
        filename4 = "output_zone4.csv"
        filename5 = "output_actions.csv"
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
        
    def savefun_Zone1(self,data):
        
        self.datalistz1.append(data)
        if self.nn1 == 200:
            with open("output_zone1.csv",'a', newline='') as resultFile:
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
            with open("output_zone2.csv",'a', newline='') as resultFile:
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
            with open("output_zone3.csv",'a', newline='') as resultFile:
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
            with open("output_zone4.csv",'a', newline='') as resultFile:
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
            with open("output_actions.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistmix)
            resultFile.close()
            self.datalistmix =[]
            self.nn =0
            
            
        self.n =self.n +1
        self.nn =self.nn +1





    def calculate_ai_input_Zone1(self, out,actions_to_env,timeofday,price1,price2,price3,price4,Hard_V1,Hard_V2,Hard_V3,Hard_V4,Hard_mix):

        # Standadize input data
        sun = out['Sun1']
        sun_for3 = out['Sunforecast3']
        T1 = out['RoomTemperature1']
        T2 = out['RoomTemperature2']
        T3 = out['RoomTemperature3']
        T4 = out['RoomTemperature4']
        T_amb = out['Tamb1']
        T_amb_for1=out['Tambforecast2']
        T_amb_for2=out['Tambforecast3']
        if out['Valveout1'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        if out['Valveout2'] <0.5:
            Valve2 =  0 
        else:
            Valve2 =  1  
        if out['Valveout3'] <0.5:
            Valve3 =  0 
        else:
            Valve3 =  1  
        if out['Valveout4'] <0.5:
            Valve4 =  0 
        else:
            Valve4 =  1  
        distance1 = T1 - (273.15+22)
        distance2 = T2 - (273.15+22)
        distance3 = T3 - (273.15+22)
        distance4 = T4 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = round(timeofday)/24
        T_amb_std = (T_amb - 273.15)/10
        T_amb_for1_std = (T_amb_for1 - 273.15)/10  
        T_amb_for2_std = (T_amb_for2 - 273.15)/10 
        # Room Temperature
        T1_std = (T1-(273.15+22))/2
        if (T1-self.params.goalT1) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0


        T2_std = (T2-(273.15+22))/2
        if (T2-self.params.goalT2) <= 0:
            orientation2_std = 1
        else:
            orientation2_std = 0

        T3_std = (T3-(273.15+22))/2
        if (T3-self.params.goalT3) <= 0:
            orientation3_std = 1
        else:
            orientation3_std = 0

        T4_std = (T4-(273.15+22))/2
        if (T4-self.params.goalT4) <= 0:
            orientation4_std = 1
        else:
            orientation4_std = 0
        

        sun_std = sun/100
        sun_for3_std = out['Sunforecast3']/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp_std=(np.float64(actions_to_env['SupplyTemperature'])-(273.15+30))/15
        T1_std = 2+round(T1_std*100)/100
        T2_std = 2+round(T2_std*100)/100
        T3_std = 2+round(T3_std*100)/100
        T4_std = 2+round(T4_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        diff_Room_temp1 = T1_std-self.last_T1
        diff_Room_temp2 = T2_std-self.last3_T2
        diff_Room_temp3 = T3_std-self.last_T3
        diff_Room_temp4 = T4_std-self.last3_T4
        self.last3_T1 = self.last2_T1
        self.last3_T2 = self.last2_T2
        self.last3_T3 = self.last2_T3
        self.last3_T4 = self.last2_T4
        self.last2_T1 = self.last_T1
        self.last2_T2 = self.last_T2
        self.last2_T3 = self.last_T3
        self.last2_T4 = self.last_T4
        self.last_T1 = T1_std 
        self.last_T2 = T2_std
        self.last_T3 = T3_std 
        self.last_T4 = T4_std
        Hard_V1_std =Hard_V1/10
        Hard_V2_std =Hard_V2/10
        Hard_V3_std =Hard_V1/10
        Hard_V4_std =Hard_V2/10
        Hard_mix_std=Hard_mix/10

        state = np.array([T1_std,T2_std,T3_std,T4_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,Valve2,Valve3,Valve4,Hard_V1_std,Hard_V2_std,Hard_V3_std,Hard_V4_std,Hard_mix_std,diff_Room_temp1,diff_Room_temp2,diff_Room_temp3,diff_Room_temp4,T_mix_std,sun_round,sun_for3_std_round,timeofday_std,price1*3,price2*3,price3*3,price4*3,self.last_Supplytemp_std], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        self.state_list.append(state)
        if len(self.state_list) > 10:
            self.state_list.pop(0)
        if len(self.state_list) < 10:
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
        state=np.append(self.state_list[0],self.state_list[1])
        state=np.append(state,self.state_list[2])
        state=np.append(state,self.state_list[3])
        state=np.append(state,self.state_list[4])
        state=np.append(state,self.state_list[5])
        state=np.append(state,self.state_list[6])
        state=np.append(state,self.state_list[7])
        state=np.append(state,self.state_list[8])
        state=np.append(state,self.state_list[9])
        self.count=self.count+1
        if self.count > 50:
            self.count=0
            #self.state_list=[]
            #print('This is state',state)
        return state     



    def calculate_ai_input_Zone2(self, out,actions_to_env,timeofday):

        # Standadize input data
        sun = out['Sun1']
        sun_for3 = out['Sunforecast3']
        T1 = out['RoomTemperature1']
        T2 = out['RoomTemperature2']
        T_amb = out['Tamb1']
        T_amb_for1=out['Tambforecast2']
        T_amb_for2=out['Tambforecast3']
        if out['Valveout1'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        if out['Valveout2'] <0.5:
            Valve2 =  0 
        else:
            Valve2 =  1  
        distance1 = T1 - (273.15+22)
        distance2 = T2 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = timeofday/24
        T_amb_std = (T_amb - 273.15)/10
        T_amb_for1_std = (T_amb_for1 - 273.15)/10  
        T_amb_for2_std = (T_amb_for2 - 273.15)/10 
        # Room Temperature
        T1_std = (T1-(273.15+22))/2
        if (T1-self.params.goalT1) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0


        T2_std = (T2-(273.15+22))/2
        if (T2-self.params.goalT2) <= 0:
            orientation2_std = 1
        else:
            orientation2_std = 0

        

        sun_std = sun/100
        sun_for3_std = out['Sunforecast3']/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        T1_std = 2+round(T1_std*100)/100
        T2_std = 2+round(T2_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        diff_Room_temp1 = T1_std-self.last3_T1
        diff_Room_temp2 = T2_std-self.last3_T2
        self.last3_T1 = self.last2_T1
        self.last3_T2 = self.last2_T2
        self.last2_T1 = self.last_T1
        self.last2_T2 = self.last_T2
        self.last_T1 = T1_std 
        self.last_T2 = T2_std
        state = np.array([T1_std,T2_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,Valve2,diff_Room_temp1,diff_Room_temp2,T_mix_std,sun_round,sun_for3_std_round], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        self.state_list.append(state)
        if len(self.state_list) > 6:
            self.state_list.pop(0)
        if len(self.state_list) < 6:
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
        state=np.append(self.state_list[0],self.state_list[1])
        state=np.append(state,self.state_list[2])
        state=np.append(state,self.state_list[3])
        state=np.append(state,self.state_list[4])
        state=np.append(state,self.state_list[5])
        self.count=self.count+1
        if self.count > 50:
            self.count=0
            #self.state_list=[]
            #print('This is state',state)
        return state     


    def calculate_ai_input_mix(self, out,actions_to_env,timeofday,Hardconstraint,last_price1,last_price2):

        # Standadize input data
        sun = out['Sun1']
        sun_for3 = out['Sunforecast3']
        T1 = out['RoomTemperature1']
        T2 = out['RoomTemperature2']
        T_amb = out['Tamb1']
        T_amb_for1=out['Tambforecast2']
        T_amb_for2=out['Tambforecast3']
        if out['Valveout1'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        if out['Valveout2'] <0.5:
            Valve2 =  0 
        else:
            Valve2 =  1  
        distance1 = T1 - (273.15+22)
        distance2 = T2 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = timeofday/24
        T_amb_std = (T_amb - 273.15)/10
        T_amb_for1_std = (T_amb_for1 - 273.15)/10  
        T_amb_for2_std = (T_amb_for2 - 273.15)/10 
        # Room Temperature
        T1_std = (T1-(273.15+22))/2
        if (T1-self.params.goalT1) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0


        T2_std = (T2-(273.15+22))/2
        if (T2-self.params.goalT2) <= 0:
            orientation2_std = 1
        else:
            orientation2_std = 0
        sum_price=last_price1+last_price2
        if sum_price < 0:
            sum_price=0.1  
        

        sun_std = sun/100
        sun_for3_std = out['Sunforecast3']/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        T1_std = 2+round(T1_std*100)/100
        T2_std = 2+round(T2_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        sum_price = 2+round(sum_price*100)/100
        diff_Room_temp1 = T1_std-self.last3_T1
        diff_Room_temp2 = T2_std-self.last3_T2
        self.last3_T1 = self.last2_T1
        self.last3_T2 = self.last2_T2
        self.last2_T1 = self.last_T1
        self.last2_T2 = self.last_T2
        self.last_T1 = T1_std 
        self.last_T2 = T2_std
        state = np.array([T1_std,T2_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,Valve2,diff_Room_temp1,diff_Room_temp2,T_mix_std,Hardconstraint/10,sun_round,sun_for3_std_round], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        self.state_list.append(state)
        if len(self.state_list) > 6:
            self.state_list.pop(0)
        if len(self.state_list) < 6:
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
            self.state_list.append(state)
        state=np.append(self.state_list[0],self.state_list[1])
        state=np.append(state,self.state_list[2])
        state=np.append(state,self.state_list[3])
        state=np.append(state,self.state_list[4])
        state=np.append(state,self.state_list[5])
        self.count=self.count+1
        if self.count > 50:
            self.count=0
            #self.state_list=[]
            #print('This is state',state)
        return state        
            # def calculate_ai_input_Zone1(self, out,actions_to_env,timeofday):

    #     # Standadize input data
    #     T1 = out['RoomTemperature1']
    #     T_amb = out['Tamb1']
    #     if out['Valveout1'] <1:
    #         Valve1 =  0 
    #     else:
    #         Valve1 =  1 

    #     if (T1-self.params.goalT1) <= 0:
    #         orientation1_std = 0.5
    #     else:
    #         orientation1_std = -0.5
            
    #     #T_amb_std

    #     T_amb_std = (T_amb - 273.15)/10
    #     T_mix_std = (actions_to_env['SupplyTemperature'] - 273.15+40)/10   
    #     # Room Temperature
    #     T1_std = (T1 ) / 300
        
    #     # Diff
    #     diff1_std = abs((T1 - self.last_T1))
    #     distance1 = T1 - (273.15+22)
    #     timeofday_std = timeofday/24
    #     # Update
    #     self.last_T1 = T1
    #     action = actions_to_env['Valve1']

    #     state = np.array([T1_std, distance1, orientation1_std , T_mix_std, Valve1,T_amb_std,timeofday_std], dtype=float)#orientation1_std, diff1_std,
        
    #     return state
