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
        self.last_T1_m = 0
        self.last_T2 = 0
        self.last_T3 = 0
        self.last_T4 = 0
        self.last2_T1 = 0
        self.last2_T1_m = 0
        self.last2_T2=0
        self.last2_T3 = 0
        self.last2_T4=0
        self.last3_T1 =0 
        self.last3_T1_m =0 
        self.last3_T2 = 0
        self.last3_T3 =0 
        self.last3_T4 = 0

        self.T_amb = 0
        self.datalistout =[]
        self.datalistz1 =[]
        self.datalistz1_NN =[]
        self.datalistz2_NN =[]
        self.datalistz3_NN =[]
        self.datalistz4_NN =[]
        self.datalistz2 =[]
        self.datalistz3 =[]
        self.datalistz4 =[]
        self.datalistmix =[]
        self.datalistmix_NN =[]
        self.last_Supplytemp=273.15+40
        self.last_Supplytemp_m=273.15+40
        self.state_list=[]
        self.n=0
        self.nn=0


        self.state_tm13=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm12=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm11=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm10=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm9=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm8=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm7=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm6=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm5=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm4=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm3=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm2=[1,1,1,1,1,1,1,1,1,1]
        self.state_tm1=[1,1,1,1,1,1,1,1,1,1]
        self.state=[1,1,1,1,1,1,1,1,1,1]

        self.state2_tm5=[1,1,1,1,1,1,1,1,1,1]
        self.state2_tm4=[1,1,1,1,1,1,1,1,1,1]
        self.state2_tm3=[1,1,1,1,1,1,1,1,1,1]
        self.state2_tm2=[1,1,1,1,1,1,1,1,1,1]
        self.state2_tm1=[1,1,1,1,1,1,1,1,1,1]
        self.state2=[1,1,1,1,1,1,1,1,1,1]


        self.state3_tm5=[1,1,1,1,1,1,1,1,1,1]
        self.state3_tm4=[1,1,1,1,1,1,1,1,1,1]
        self.state3_tm3=[1,1,1,1,1,1,1,1,1,1]
        self.state3_tm2=[1,1,1,1,1,1,1,1,1,1]
        self.state3_tm1=[1,1,1,1,1,1,1,1,1,1]
        self.state3=[1,1,1,1,1,1,1,1,1,1]

        self.state4_tm5=[1,1,1,1,1,1,1,1,1,1]
        self.state4_tm4=[1,1,1,1,1,1,1,1,1,1]
        self.state4_tm3=[1,1,1,1,1,1,1,1,1,1]
        self.state4_tm2=[1,1,1,1,1,1,1,1,1,1]
        self.state4_tm1=[1,1,1,1,1,1,1,1,1,1]
        self.state4=[1,1,1,1,1,1,1,1,1,1]
        self.valve1_list=[]
        self.valve2_list=[]
        self.valve3_list=[]
        self.valve4_list=[]

        self.state_mix_tm13=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm12=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm11=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm10=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm9=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm8=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm7=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm6=[1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm5=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm4=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm3=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm2=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix_tm1=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.state_mix=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.n0=0
        self.nn0=0
        self.n1=0
        self.nn1=0
        self.n11=0
        self.nn11=0
        self.n11a=0
        self.nn11a=0
        self.n11b=0
        self.nn11b=0
        self.n11c=0
        self.nn11c=0
        self.D1=0
        self.n2=0
        self.nn2=0
        self.n3=0
        self.nn3=0
        self.n4=0
        self.nn4=0
        self.n_5=0
        self.nn_5=0
        self.D1_5=0
        filename0 = "output_out.csv"
        filename1 = "output_zone1.csv"
        filename2 = "output_zone2.csv"
        filename3 = "output_zone3.csv"
        filename4 = "output_zone4.csv"
        filename5 = "output_actions.csv"
        filename61 = "output_zone1_NN.csv"
        filename62 = "output_zone2_NN.csv"
        filename63 = "output_zone3_NN.csv"
        filename64 = "output_zone4_NN.csv"
        filename7 = "output_actions_NN.csv"
        f0 = open(filename0, "w+")
        f1 = open(filename1, "w+")
        f2 = open(filename2, "w+")
        f3 = open(filename3, "w+")
        f4 = open(filename4, "w+")
        f5 = open(filename5, "w+")
        f61 = open(filename61, "w+")
        f62 = open(filename62, "w+")
        f63 = open(filename63, "w+")
        f64 = open(filename64, "w+")
        f7 = open(filename7, "w+")
        f0.close()
        f1.close()  
        f2.close()
        f3.close()  
        f4.close()
        f5.close()
        f61.close()
        f62.close()
        f63.close()
        f64.close()
        f7.close()

    def savefun_out(self,data):
        
        self.datalistout.append(data)
        if self.nn0 == 200:
            with open("output_out.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistout)
            resultFile.close()
            self.datalistout =[]
            self.nn0 =0
            
            
        self.n0 =self.n0 +1
        self.nn0 =self.nn0 +1 
        


    def savefun_Zone1_NN(self,data):
        
        self.datalistz1_NN.append(data)
        if self.nn11 == 200:
            with open("output_zone1_NN.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz1_NN)
            resultFile.close()
            self.datalistz1_NN =[]
            self.nn11 =0
            
            
        self.n11 =self.n11 +1
        self.nn11 =self.nn11 +1


    def savefun_Zone2_NN(self,data):
        
        self.datalistz2_NN.append(data)
        if self.nn11a == 200:
            with open("output_zone2_NN.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz2_NN)
            resultFile.close()
            self.datalistz2_NN =[]
            self.nn11a =0
            
            
        self.n11a =self.n11a +1
        self.nn11a =self.nn11a +1


    def savefun_Zone3_NN(self,data):
        
        self.datalistz3_NN.append(data)
        if self.nn11b == 200:
            with open("output_zone3_NN.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz3_NN)
            resultFile.close()
            self.datalistz3_NN =[]
            self.nn11b =0
            
            
        self.n11b =self.n11b +1
        self.nn11b =self.nn11b +1

    def savefun_Zone4_NN(self,data):
        
        self.datalistz4_NN.append(data)
        if self.nn11c == 200:
            with open("output_zone4_NN.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistz4_NN)
            resultFile.close()
            self.datalistz4_NN =[]
            self.nn11c =0
            
            
        self.n11c =self.n11c +1
        self.nn11c =self.nn11c +1

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

    def savefun_actiondata_NN(self,data):
        
        self.datalistmix_NN.append(data)

        if self.nn_5 == 200:
            with open("output_actions_NN.csv",'a', newline='') as resultFile:
                wr = csv.writer(resultFile)
                wr.writerows(self.datalistmix_NN)
            resultFile.close()
            self.datalistmix_NN =[]
            self.nn_5 =0
            
            
        self.n_5 =self.n_5 +1
        self.nn_5 =self.nn_5 +1





    def calculate_ai_input_Zone1(self, out,actions_to_env,timeofday,Price1):

        # Standadize input data
        sun = float(out['Sun1'])
        sun_for3 = float(out['Sunforecast3'])
        T1 = float(out['RoomTemperature1'])
        T_amb = float(out['Tamb1'])
        T_amb_for1=float(out['Tambforecast2'])
        T_amb_for2=float(out['Tambforecast3'])
        if out['Valveout1'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        distance1 = T1 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = timeofday/24
        T_amb_std = (T_amb - 260)/10
        T_amb_for1_std = (T_amb_for1 - 260)/10  
        T_amb_for2_std = (T_amb_for2 - 260)/10 
        # Room Temperature
        T1_std = (T1-(273.15+22))/2
        if (T1-self.params.goalT1) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0 

        sun_std = sun/100
        sun_for3_std = float(out['Sunforecast3'])/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        T1_std = 2+round(T1_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        diff_Room_temp1 = T1_std-self.last3_T1
        self.last3_T1 = self.last2_T1
        self.last2_T1 = self.last_T1
        self.last_T1 = T1_std 
        Price1 = Price1

        self.state_tm5=self.state_tm4
        self.state_tm4=self.state_tm3
        self.state_tm3=self.state_tm2
        self.state_tm2=self.state_tm1
        self.state_tm1=self.state

        self.state = np.array([T1_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,diff_Room_temp1,T_mix_std,sun_round,sun_for3_std_round,timeofday_std], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        
        
        state_2d=[self.state,self.state_tm1,self.state_tm2,self.state_tm3,self.state_tm4,self.state_tm5]#,self.state_tm6,self.state_tm7,self.state_tm8,self.state_tm9]#,self.state_tm10,self.state_tm11,self.state_tm12,self.state_tm13]
        
        return state_2d    




    def calculate_ai_input_Zone2(self, out,actions_to_env,timeofday,Price2):

        # Standadize input data
        sun = float(out['Sun1'])
        sun_for3 = float(out['Sunforecast3'])
        T2 = float(out['RoomTemperature2'])
        T_amb = float(out['Tamb1'])
        T_amb_for1=float(out['Tambforecast2'])
        T_amb_for2=float(out['Tambforecast3'])
        if out['Valveout2'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        distance1 = T2 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = timeofday/24
        T_amb_std = (T_amb - 260)/10
        T_amb_for1_std = (T_amb_for1 - 260)/10  
        T_amb_for2_std = (T_amb_for2 - 260)/10 
        # Room Temperature
        T2_std = (T2-(273.15+22))/2
        if (T2-self.params.goalT2) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0 

        sun_std = sun/100
        sun_for3_std = float(out['Sunforecast3'])/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        T2_std = 2+round(T2_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        diff_Room_temp2 = T2_std-self.last3_T2
        self.last3_T2 = self.last2_T2
        self.last2_T2 = self.last_T2
        self.last_T2 = T2_std 
        #Price1 = Price2

        self.state2_tm5=self.state2_tm4
        self.state2_tm4=self.state2_tm3
        self.state2_tm3=self.state2_tm2
        self.state2_tm2=self.state2_tm1
        self.state2_tm1=self.state2

        self.state2 = np.array([T2_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,diff_Room_temp2,T_mix_std,sun_round,sun_for3_std_round,timeofday_std], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        
        
        state_2d=[self.state2,self.state2_tm1,self.state2_tm2,self.state2_tm3,self.state2_tm4,self.state2_tm5]#,self.state_tm6,self.state_tm7,self.state_tm8,self.state_tm9]#,self.state_tm10,self.state_tm11,self.state_tm12,self.state_tm13]
        
        return state_2d    




    def calculate_ai_input_Zone3(self, out,actions_to_env,timeofday,Price3):

        # Standadize input data
        sun = float(out['Sun1'])
        sun_for3 = float(out['Sunforecast3'])
        T3 = float(out['RoomTemperature3'])
        T_amb = float(out['Tamb1'])
        T_amb_for1=float(out['Tambforecast2'])
        T_amb_for2=float(out['Tambforecast3'])
        if out['Valveout3'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        distance1 = T3 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = timeofday/24
        T_amb_std = (T_amb - 260)/10
        T_amb_for1_std = (T_amb_for1 - 260)/10  
        T_amb_for2_std = (T_amb_for2 - 260)/10 
        # Room Temperature
        T3_std = (T3-(273.15+22))/2
        if (T3-self.params.goalT3) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0 

        sun_std = sun/100
        sun_for3_std = float(out['Sunforecast3'])/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        T3_std = 2+round(T3_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        diff_Room_temp3 = T3_std-self.last3_T3
        self.last3_T3 = self.last2_T3
        self.last2_T3 = self.last_T3
        self.last_T3 = T3_std 
        #Price1 = Price2

        self.state3_tm5=self.state3_tm4
        self.state3_tm4=self.state3_tm3
        self.state3_tm3=self.state3_tm2
        self.state3_tm2=self.state3_tm1
        self.state3_tm1=self.state3

        self.state3 = np.array([T3_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,diff_Room_temp3,T_mix_std,sun_round,sun_for3_std_round,timeofday_std], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        
        
        state_2d=[self.state3,self.state3_tm1,self.state3_tm2,self.state3_tm3,self.state3_tm4,self.state3_tm5]#,self.state_tm6,self.state_tm7,self.state_tm8,self.state_tm9]#,self.state_tm10,self.state_tm11,self.state_tm12,self.state_tm13]
        
        return state_2d    

    def calculate_ai_input_Zone4(self, out,actions_to_env,timeofday,Price4):

        # Standadize input data
        sun = float(out['Sun1'])
        sun_for3 = float(out['Sunforecast3'])
        T4 = float(out['RoomTemperature4'])
        T_amb = float(out['Tamb1'])
        T_amb_for1=float(out['Tambforecast2'])
        T_amb_for2=float(out['Tambforecast3'])
        if out['Valveout4'] <0.5:
            Valve1 =  0 
        else:
            Valve1 =  1
        distance1 = T4 - (273.15+22)
        diff_amb = T_amb-T_amb_for2   
        #T_amb_std
        timeofday_std = timeofday/24
        T_amb_std = (T_amb - 260)/10
        T_amb_for1_std = (T_amb_for1 - 260)/10  
        T_amb_for2_std = (T_amb_for2 - 260)/10 
        # Room Temperature
        T4_std = (T4-(273.15+22))/2
        if (T4-self.params.goalT4) <= 0:
            orientation1_std = 1
        else:
            orientation1_std = 0 

        sun_std = sun/100
        sun_for3_std = float(out['Sunforecast3'])/100
        sun_round=2+round(sun_std*100)/100
        sun_for3_std_round = 2+round(sun_for3_std*100)/100
        T_mix_std = 2+round(((np.float64(actions_to_env['SupplyTemperature']) - (273.15+46))/10)*100)/100  
        change_in_Supply_temp = round(((np.float64(actions_to_env['SupplyTemperature'])-self.last_Supplytemp)*-0.05)*100)/100
        self.last_Supplytemp=np.float64(actions_to_env['SupplyTemperature'])
        T4_std = 2+round(T4_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        diff_Room_temp4 = T4_std-self.last3_T4
        self.last3_T4 = self.last2_T4
        self.last2_T4 = self.last_T4
        self.last_T4 = T4_std 
        #Price1 = Price2

        self.state4_tm5=self.state4_tm4
        self.state4_tm4=self.state4_tm3
        self.state4_tm3=self.state4_tm2
        self.state4_tm2=self.state4_tm1
        self.state4_tm1=self.state4

        self.state4 = np.array([T4_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,diff_Room_temp4,T_mix_std,sun_round,sun_for3_std_round,timeofday_std], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        
        
        state_2d=[self.state4,self.state4_tm1,self.state4_tm2,self.state4_tm3,self.state4_tm4,self.state4_tm5]#,self.state_tm6,self.state_tm7,self.state_tm8,self.state_tm9]#,self.state_tm10,self.state_tm11,self.state_tm12,self.state_tm13]
        
        return state_2d    








    def calculate_ai_input_mix(self, out,actions_to_env,timeofday,Hardconstraint,last_price1,last_price2,last_price3,last_price4):

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



        sum_price=last_price1+last_price2+last_price3+last_price4
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
        T3_std = 2+round(T3_std*100)/100
        T4_std = 2+round(T4_std*100)/100
        T_amb_std = 2+round(T_amb_std*100)/100
        T_amb_for2_std = 2+round(T_amb_for2_std*100)/100
        diff_amb = 2+T_amb_for2_std-T_amb_std
        diff_sun = 2+sun_round-sun_for3_std_round
        sum_price = 2+round(sum_price*100)/100



        diff_Room_temp1 = T1_std-self.last3_T1
        diff_Room_temp2 = T2_std-self.last3_T2
        diff_Room_temp3 = T1_std-self.last3_T3
        diff_Room_temp4 = T2_std-self.last3_T4

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
        self.valve1_list.append(Valve1)
        self.valve2_list.append(Valve2)
        self.valve3_list.append(Valve3)
        self.valve4_list.append(Valve4)
        if len(self.valve1_list) > 5:
            self.valve1_list.pop(0)
            self.valve2_list.pop(0)
            self.valve3_list.pop(0)
            self.valve4_list.pop(0)
        #self.state_mix_tm13=self.state_mix_tm12
        #self.state_mix_tm12=self.state_mix_tm11
        #self.state_mix_tm11=self.state_mix_tm10
        #self.state_mix_tm10=self.state_mix_tm9
        #self.state_mix_tm9=self.state_mix_tm8
        #self.state_mix_tm8=self.state_mix_tm7
        #self.state_mix_tm7=self.state_mix_tm6
        #self.state_mix_tm6=self.state_mix_tm5
        self.state_mix_tm5=self.state_mix_tm4
        self.state_mix_tm4=self.state_mix_tm3
        self.state_mix_tm3=self.state_mix_tm2
        self.state_mix_tm2=self.state_mix_tm1
        self.state_mix_tm1=self.state_mix
        self.state_mix = np.array([T1_std,T2_std,T3_std,T4_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,Valve2,Valve3,Valve4,diff_Room_temp1,diff_Room_temp2,diff_Room_temp3,diff_Room_temp4,T_mix_std,Hardconstraint/10,sun_round,sun_for3_std_round,timeofday_std], dtype=float)
        #self.state_mix = np.array([T1_std, T_amb_std,T_amb_for2_std,diff_amb,Valve1,diff_Room_temp1,T_mix_std,Hardconstraint/10,sun_round,sun_for3_std_round,timeofday_std], dtype=float)#Valve1,Hardconstraint/10,T_mix_std,
        

        if len(self.state_mix_tm1) <21:
            self.state_mix_tm1=np.append(self.state_mix_tm1,self.valve1_list[0])
            self.state_mix_tm1=np.append(self.state_mix_tm1,self.valve2_list[0])
            self.state_mix_tm1=np.append(self.state_mix_tm1,self.valve3_list[0])
            self.state_mix_tm1=np.append(self.state_mix_tm1,self.valve4_list[0])
        # if len(self.state_mix_tm2) <21:
        #     self.state_mix_tm2=np.append(self.state_mix_tm2,self.valve1_list[1])
        #     self.state_mix_tm2=np.append(self.state_mix_tm2,self.valve2_list[1])
        #     self.state_mix_tm2=np.append(self.state_mix_tm2,self.valve3_list[1])
        #     self.state_mix_tm2=np.append(self.state_mix_tm2,self.valve4_list[1])
        # if len(self.state_mix_tm3) <21:
        #     self.state_mix_tm3=np.append(self.state_mix_tm3,self.valve1_list[2])
        #     self.state_mix_tm3=np.append(self.state_mix_tm3,self.valve2_list[2])
        #     self.state_mix_tm3=np.append(self.state_mix_tm3,self.valve3_list[2])
        #     self.state_mix_tm3=np.append(self.state_mix_tm3,self.valve4_list[2])
        # if len(self.state_mix_tm4) <21:
        #     self.state_mix_tm4=np.append(self.state_mix_tm4,self.valve1_list[3])
        #     self.state_mix_tm4=np.append(self.state_mix_tm4,self.valve2_list[3])
        #     self.state_mix_tm4=np.append(self.state_mix_tm4,self.valve3_list[3])
        #     self.state_mix_tm4=np.append(self.state_mix_tm4,self.valve4_list[3])
        # if len(self.state_mix_tm5) <21:
        #     self.state_mix_tm5=np.append(self.state_mix_tm5,self.valve1_list[4])
        #     self.state_mix_tm5=np.append(self.state_mix_tm5,self.valve2_list[4])
        #     self.state_mix_tm5=np.append(self.state_mix_tm5,self.valve3_list[4])
        #     self.state_mix_tm5=np.append(self.state_mix_tm5,self.valve4_list[4])



        state_2d=[self.state_mix,self.state_mix_tm1,self.state_mix_tm2,self.state_mix_tm3,self.state_mix_tm4,self.state_mix_tm5]#,self.state_mix_tm6,self.state_mix_tm7,self.state_mix_tm8,self.state_mix_tm9,self.state_mix_tm10]#,self.state_tm11,self.state_tm12,self.state_tm13]
     
        return state_2d      

