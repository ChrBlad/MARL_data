# plotting file 

import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
data=[]
data_actions=[]
data_actions_ref=[]
data_zone1=[]
data_zone1_com=[]
data_actions_com=[]
data_zone2=[]
data_zone2_com=[]
data_zone3=[]
data_zone3_com=[]
data_zone4=[]
data_zone4_com=[]
data_ref = []
error20_ref=[]
error20=[]
reward20_ref=[]
reward20=[]
Valves20_ref=[]
Valves20=[]
power20_ref=[]
power20=[]
power20_ref1=[]
power201=[]
temp20_ref=[]
temp20=[]
price20_ref=[]
price20=[]
ref=22
sum_number=15
supplytemp_ref=[]
supplytemp20=[]
supplytemp20_ref=[]

number=108#3240#12400#108#108#

n=0
m=number
n_ref=0
m_ref=number

n_p=0
m_p=number
n_p_ref=0
m_p_ref=number


n_pr=0
m_pr=number
n_pr_ref=0
m_pr_ref=number


n_t=0
m_t=number
n_t_ref=0
m_t_ref=number

with open('output_zone1.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone1.append(row)

csvFile.close()

with open('output_zone2.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone2.append(row)

csvFile.close()

with open('output_zone3.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone3.append(row)

csvFile.close()
with open('output_zone4.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone4.append(row)

csvFile.close()


with open('output_zone1_ref.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone1_com.append(row)

csvFile.close()


with open('output_zone2_ref.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone2_com.append(row)

csvFile.close()

with open('output_zone3_ref.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone3_com.append(row)

csvFile.close()

with open('output_zone4_ref.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_zone4_com.append(row)

csvFile.close()

# with open('output_zone4_ref.csv', 'r') as csvFile:
#     reader = csv.reader(csvFile)
#     for row in reader:
#         data_zone4_com.append(row)

# csvFile.close()

with open('output_actions.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_actions.append(row)

csvFile.close()


with open('output_actions_ref.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_actions_com.append(row)

csvFile.close()

print('type',type(data_actions_com))
action_len=len(data_actions)
zone1_len=len(data_zone1)


if action_len>zone1_len:
    del data_actions[zone1_len:action_len]
elif action_len<zone1_len:
    del data_zone1[action_len:zone1_len]




df_actions = pd.DataFrame(data_actions)
df_actions.columns = ["Valve1",  "Supplytemp","Reward","Hardcon","price_reward","sum_mix_reward","change_in_Supply_temp"]
df_actions_com = pd.DataFrame(data_actions_com)
df_actions_com.columns = ["Valve1","Valve2", "Supplytemp","Reward"]

# df_actions['Supplytemp'] = [eval(i) for i in df_actions['Supplytemp']]





# lenght_ac=len(df_actions)
# f=1
# while f < lenght_ac:
#     df_actions['Supplytemp'][f] = np.float64(df_actions['Supplytemp'][f][0])
#     f=f+1











SS_ref=len(data_zone1)
SS=len(data_zone1_com)
SS_action=len(data_actions)
SS_action_ref=len(data_actions_ref)
SS=min(SS,SS_ref)
SS_action=min(SS_action,SS_action_ref)
df_zone1 = pd.DataFrame(data_zone1)
df_zone1.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"time"]
df_zone2 = pd.DataFrame(data_zone2)
df_zone2.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"time"]
df_zone3 = pd.DataFrame(data_zone3)
df_zone3.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"time"]
df_zone4 = pd.DataFrame(data_zone4)
df_zone4.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"time"]
# df_zone4 = pd.DataFrame(data_zone4)
# df_zone4.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater','Hcv',"time"]
# df_zone1['action'] = [eval(i) for i in df_zone1['action']]
# lenght_ac=len(df_zone1)
# f=0
# while f < lenght_ac:
#     df_zone1['action'][f] = np.float64(df_zone1['action'][f][0])
#     f=f+1



df_zone1_com = pd.DataFrame(data_zone1_com)
df_zone1_com.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater',"time"]



df_zone2_com = pd.DataFrame(data_zone2_com)
df_zone2_com.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater',"time"]

df_zone3_com = pd.DataFrame(data_zone3_com)
df_zone3_com.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater',"time"]

df_zone4_com = pd.DataFrame(data_zone4_com)
df_zone4_com.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater',"time"]

# df_zone4_com = pd.DataFrame(data_zone4_com)
# df_zone4_com.columns = ["Troom", "Tamb", "Valve","Power1","Reward","action",'Price','Power','Powerwater',"time"]




df_zone1=df_zone1.astype(float)

df_zone2=df_zone2.astype(float)
df_zone3=df_zone3.astype(float)

df_zone4=df_zone4.astype(float)
# df_zone4=df_zone4.astype(float)


df_zone1_com=df_zone1_com.astype(float)

df_zone2_com=df_zone2_com.astype(float)
df_zone3_com=df_zone3_com.astype(float)

df_zone4_com=df_zone4_com.astype(float)
# df_zone4_com=df_zone4_com.astype(float)


df_actions = df_actions.astype(float)
df_actions_com = df_actions_com.astype(float)



#df_zone1['Valve']=df_zone1['Valve']+13
df_zone1['action']=df_zone1['action']+20.1
df_zone1['Troom']=df_zone1['Troom']-273.15
df_zone1['Tamb']=df_zone1['Tamb']-273.15

df_zone2['action']=df_zone2['action']+20.1
df_zone2['Troom']=df_zone2['Troom']-273.15
df_zone2['Tamb']=df_zone2['Tamb']-273.15

df_zone3['action']=df_zone3['action']+20.1
df_zone3['Troom']=df_zone3['Troom']-273.15
df_zone3['Tamb']=df_zone3['Tamb']-273.15


df_zone4['action']=df_zone4['action']+20.1
df_zone4['Troom']=df_zone4['Troom']-273.15
df_zone4['Tamb']=df_zone4['Tamb']-273.15
# df_zone4['action']=df_zone4['action']+20.1
# df_zone4['Troom']=df_zone4['Troom']-273.15
# df_zone4['Tamb']=df_zone4['Tamb']-273.15
#df_actions['Reward']=df_actions['Reward']+df_actions['Hardcon']
df_actions['Reward']=df_actions['Reward']-df_zone1['Reward']-df_zone2['Reward']-df_zone3['Reward']-df_zone4['Reward']+df_actions['Hardcon']##+df_zone2['Reward']

#df_zone1_com['Valve']=df_zone1_com['Valve']+16
df_zone1_com['Troom']=df_zone1_com['Troom']-273.15
df_zone1_com['Tamb']=df_zone1_com['Tamb']-273.15

df_zone2_com['Troom']=df_zone2_com['Troom']-273.15
df_zone2_com['Tamb']=df_zone2_com['Tamb']-273.15

df_zone3_com['Troom']=df_zone3_com['Troom']-273.15
df_zone3_com['Tamb']=df_zone3_com['Tamb']-273.15


df_zone4_com['Troom']=df_zone4_com['Troom']-273.15
df_zone4_com['Tamb']=df_zone4_com['Tamb']-273.15
# df_zone4_com['Troom']=df_zone4_com['Troom']-273.15
# df_zone4_com['Tamb']=df_zone4_com['Tamb']-273.15

#df_actions['Hardcon']=df_actions['Hardcon']+15
df_actions['Supplytemp']=df_actions['Supplytemp']-273.15
df_actions_com['Supplytemp']=df_actions_com['Supplytemp']-272.15
#supply_corrected=list(range(SS_action))
#supply_corrected[0]=45
cc=0
Valve = df_actions['Valve1']
Stemp = df_actions['Supplytemp']
#SS_action_ref


# for c in range(SS_action):
#     if Valve[c] > 0.1:
#         supply_corrected[c] = Stemp[c]
#         cc=c
#     else: 
#         supply_corrected[c] = supply_corrected[cc]

# df_actions['supply_corrected'] = supply_corrected

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


print(df_actions.head())
#print(df_actions_com.head())
print(df_zone1.head())

#print(df_zone1_com.head())
def Evaluation_of_power1(m_p,n_p,SS,n_p_ref,m_p_ref,df_zone1,df_zone1_com,power20,power20_ref,number):
    n_p=0
    m_p=number
    n_p_ref=0
    m_p_ref=number
    '''
    Function to evauate the results of a plot
    '''
    power = df_zone1['Power1']
    
    while m_p<SS: 
        power20.append(sum(abs(power[n_p:m_p]))/number)
        n_p=n_p+number
        m_p=m_p+number

    power_ref = df_zone1_com['Power1']
    while m_p_ref<SS: 
        power20_ref.append(sum(abs(power_ref[n_p_ref:m_p_ref]))/number)
        n_p_ref=n_p_ref+number
        m_p_ref=m_p_ref+number
    f2 = plt.figure(2)
    X = list(range(len(power20)))
    X_ref = list(range(len(power20_ref)))
    p20=np.asarray(power20, dtype=np.float32)
    pr20=np.asarray(power20_ref, dtype=np.float32)
    savings = p20-pr20
    X_sav = list(range(len(savings)))
    plt.plot(X,power20,label='RL', alpha=0.7)
    plt.plot(X_ref,power20_ref,label='hys control ref', alpha=0.7)
    plt.plot(X_sav,savings,label='savings', alpha=0.7)
    plt.ylabel('Power[W]')
    plt.xlabel('Time')    
    plt.legend()
    plt.grid(True)

def Evaluation_of_power(m_p,n_p,SS,n_p_ref,m_p_ref,df_zone1,df_zone1_com,power20,power20_ref,number):
    '''
    Function to evauate the results of a plot
    '''
    n_p=0
    m_p=number
    n_p_ref=0
    m_p_ref=number
    power=[]
    for i in range(SS-2):
        power.append(df_zone1['Powerwater'][i+1]-df_zone1['Powerwater'][i])
    
    while m_p<SS: 
        power201.append(sum(np.abs(power[n_p:m_p]))/number)
        n_p=n_p+number
        m_p=m_p+number
    power_ref=[]
    for i in range(SS-2):
        power_ref.append(df_zone1_com['Powerwater'][i+1]-df_zone1_com['Powerwater'][i])

    while m_p_ref<SS: 
        power20_ref1.append(sum(np.abs(power_ref[n_p_ref:m_p_ref]))/number)
        n_p_ref=n_p_ref+number
        m_p_ref=m_p_ref+number
    f2 = plt.figure(2)
    X = list(range(len(power201)))
    X_ref = list(range(len(power20_ref1)))
    p20=np.asarray(power201, dtype=np.float32)
    pr20=np.asarray(power20_ref1, dtype=np.float32)
    savings = p20-pr20
    print(sum(savings[sum_number:SS]))
    print(sum(pr20[sum_number:SS]))
    print((sum(savings[sum_number:SS])*-1)/sum(pr20[sum_number:SS]))
    X_sav = list(range(len(savings)))
    plt.plot(X,power201,label='RL water', alpha=0.7)
    plt.plot(X_ref,power20_ref1,label='hys control ref water', alpha=0.7)
    plt.plot(X_sav,savings,label='savings', alpha=0.7)
    plt.ylabel('Power[W]')
    plt.xlabel('Time')    
    plt.legend()
    plt.grid(True)



def Evaluation_of_Roomtemp(m_t,n_t,SS,n_t_ref,m_t_ref,df_zone1,df_zone1_com,temp20,temp20_ref,number,fig_num):
    '''
    Function to evauate the results of a plot
    '''
    temp = df_zone1['Troom']
    
    while m_t<SS: 
        temp20.append(sum(abs(temp[n_t:m_t]))/number)
        n_t=n_t+number
        m_t=m_t+number
    
    temp_ref = df_zone1_com['Troom']
    while m_t_ref<SS: 
        temp20_ref.append(sum(abs(temp_ref[n_t_ref:m_t_ref]))/number)
        n_t_ref=n_t_ref+number
        m_t_ref=m_t_ref+number
    f3 = plt.figure(fig_num)
    X = list(range(len(temp20)))
    X_ref = list(range(len(temp20_ref)))
    plt.plot(X,temp20,label='RL', alpha=0.7)
    plt.plot(X_ref,temp20_ref,label='hys control ref', alpha=0.7)
    plt.ylabel('Avrage temperature[C]')
    plt.xlabel('Time')  
    plt.legend()
    plt.grid(True)




def Evaluation_of_error(m,n,SS,n_ref,m_ref,df_zone1,df_zone1_com,error20,error20_ref,number,fig_num):
    '''
    Function to evauate the results of a plot
    '''
    del error20
    del error20_ref
    error20=[]
    error20_ref=[]

    error = (df_zone1['Troom']-22)#((df_zone1['Troom']-22)+1)**2
    
    while m<SS: 
        error20.append(sum(abs(error[n:m])))
        n=n+number
        m=m+number
    
    error_ref =(df_zone1_com['Troom']-22) #((df_zone1_com['Troom']-22)+1)**2
    while m_ref<SS: 
        error20_ref.append(sum(abs(error_ref[n_ref:m_ref])))
        n_ref=n_ref+number
        m_ref=m_ref+number
    f4 = plt.figure(fig_num)
    X = list(range(len(error20)))
    X_ref = list(range(len(error20_ref)))
    plt.plot(X,error20,label='RL', alpha=0.7)
    plt.plot(X_ref,error20_ref,label='hys control ref', alpha=0.7)
    plt.legend()
    plt.ylabel('Avrage error')
    plt.xlabel('Time')  
    plt.grid(True)



def Evaluation_of_reward(m,n,SS,n_ref,m_ref,df_zone1,df_zone1_com,reward20,reward20_ref,number,fig_num):

    del reward20
    del reward20_ref
    reward20=[]
    reward20_ref=[]

    reward = df_zone1['Reward']#-df_zone1['Hcv']
    reward_com = df_zone1_com['Reward']
    
    while m<SS: 
        reward20.append(sum(reward[n:m]))
        n=n+number
        m=m+number

    while m_ref<SS: 
        reward20_ref.append(sum(reward_com[n_ref:m_ref]))
        n_ref=n_ref+number
        m_ref=m_ref+number

    f6= plt.figure(fig_num)
    X = list(range(len(reward20)))
    X_ref = list(range(len(reward20_ref)))
    plt.plot(X,reward20,label='RL', alpha=0.7)
    plt.plot(X_ref,reward20_ref,label='Outside compensation', alpha=0.7)
    plt.legend()
    plt.ylabel('Avrage reward')
    plt.xlabel('Time')  
    plt.grid(True)


def Evaluation_of_duty_cycle(m,n,SS,n_ref,m_ref,df_zone1,df_zone1_com,Valves20,Valves20_ref,number,fig_num):
    print(SS)
    
    Valves = df_zone1['Valve']
    Valves_com = df_zone1_com['Valve']
    
    while m<SS: 
        Valves20.append(sum(Valves[n:m]))
        n=n+number
        m=m+number

    while m_ref<SS: 
        Valves20_ref.append(sum(Valves_com[n_ref:m_ref]))
        n_ref=n_ref+number
        m_ref=m_ref+number

    f6= plt.figure(fig_num)
    X = list(range(len(Valves20)))
    X_ref = list(range(len(Valves20_ref)))
    plt.plot(X,Valves20,label='RL', alpha=0.7)
    plt.plot(X_ref,Valves20_ref,label='Outside compensation', alpha=0.7)
    plt.legend()
    plt.ylabel('Avrage duty cycle')
    plt.xlabel('Time')  
    plt.grid(True)



def Evaluation_of_supplytemp(m,n,SS_action,n_ref,m_ref,df_actions,df_actions_com,supplytemp20,supplytemp20_ref,number,fig_num):

    supplytemp = df_actions['supply_corrected'] #  Supplytemp 
    
    while m<SS_action: 
        supplytemp20.append(sum(supplytemp[n:m])/number)
        n=n+number
        m=m+number

    supplytemp_ref = df_actions_com['Supplytemp']
        
    while m_ref<SS_action: 
        supplytemp20_ref.append(sum(supplytemp_ref[n_ref:m_ref])/number)
        n_ref=n_ref+number
        m_ref=m_ref+number

    f6= plt.figure(fig_num)
    X = list(range(len(supplytemp20)))
    X_ref = list(range(len(supplytemp20_ref)))
    plt.plot(X,supplytemp20,label='RL', alpha=0.7)       
    plt.plot(X_ref,supplytemp20_ref,label='referance', alpha=0.7)     
    #print(X)
    #print(X_ref)
    plt.legend()
    plt.ylabel('Avrage supplytemp')
    plt.xlabel('Time')  
    plt.grid(True)



def Evaluation_of_price(m_pr,n_pr,SS,n_pr_ref,m_pr_ref,df_zone1,df_zone1_com,price20,price20_ref,number,fig_num):
    '''
    Function to evauate the results of a plot
    '''
    del price20
    del price20_ref
    price20=[]
    price20_ref=[]
    #print('len price',len(price))
    price = df_zone1['Price']
    while m_pr<SS: 
        price20.append(sum(price[n_pr:m_pr])/number)
        n_pr=n_pr+number
        m_pr=m_pr+number
    
    price_ref = df_zone1_com['Price']
    while m_pr_ref<SS: 
        price20_ref.append(sum(price_ref[n_pr_ref:m_pr_ref])/number)
        n_pr_ref=n_pr_ref+number
        m_pr_ref=m_pr_ref+number
    plt.figure(fig_num)
    X = list(range(len(price20)))
    X_ref = list(range(len(price20_ref)))
    pr20=np.asarray(price20, dtype=np.float32)
    prr20=np.asarray(price20_ref, dtype=np.float32)
    savings_price = pr20-prr20
    #print(sum(savings_price[sum_number:SS]))
    #print(sum(prr20[sum_number:SS]))
    #print((sum(savings_price[sum_number:SS])*-1)/sum(prr20[sum_number:SS]))
    print((sum(prr20)-sum(pr20))/sum(prr20))
    print('price of heating Traditional',sum(price_ref[0:SS]))
    print('price of heating RL',sum(price[0:SS]))
    X_sav = list(range(len(savings_price)))
    plt.plot(X,price20,label='RL', alpha=0.7)
    plt.plot(X_ref,price20_ref,label='hys control ref', alpha=0.7)
    plt.plot(X_sav,savings_price,label='savings_price', alpha=0.7)
    plt.ylabel('price[]')
    plt.xlabel('Time')    
    plt.legend()
    plt.grid(True)





f1 = plt.figure(1)
plt.style.use('ggplot')
plt.plot(df_zone1['time'],df_zone1['Troom'],label='RL', alpha=0.7)
plt.plot(df_zone1['time'],df_zone1['Tamb'],label='action', alpha=0.7)
plt.plot(df_zone1_com['time'],df_zone1_com['Troom'],label='hys control ref', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Reward'],label='Hardconstraint', alpha=0.7)
plt.plot(df_zone1['time'],df_zone1['Valve']+15,label='Valve pos after hard', alpha=0.7)
plt.plot(df_zone1_com['time'],df_zone1_com['Tamb'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

f2 = plt.figure(2)
plt.style.use('ggplot')
plt.plot(df_zone2['time'],df_zone2['Troom'],label='RL', alpha=0.7)
#plt.plot(df_zone2['time'],df_zone2['action']-5,label='action', alpha=0.7)
plt.plot(df_zone2_com['time'],df_zone2_com['Troom'],label='hys control ref', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Reward'],label='Hardconstraint', alpha=0.7)
plt.plot(df_zone2['time'],df_zone2['Valve']+15,label='Valve pos after hard', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

f3 = plt.figure(3)
plt.style.use('ggplot')
plt.plot(df_zone3['time'],df_zone3['Troom'],label='RL', alpha=0.7)
#plt.plot(df_zone2['time'],df_zone2['action']-5,label='action', alpha=0.7)
plt.plot(df_zone3_com['time'],df_zone3_com['Troom'],label='hys control ref', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Reward'],label='Hardconstraint', alpha=0.7)
plt.plot(df_zone3['time'],df_zone3['Valve']+15,label='Valve pos after hard', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

f4 = plt.figure(4)
plt.style.use('ggplot')
plt.plot(df_zone4['time'],df_zone4['Troom'],label='RL', alpha=0.7)
#plt.plot(df_zone2['time'],df_zone2['action']-5,label='action', alpha=0.7)
plt.plot(df_zone4_com['time'],df_zone4_com['Troom'],label='hys control ref', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Reward'],label='Hardconstraint', alpha=0.7)
plt.plot(df_zone4['time'],df_zone4['Valve']+15,label='Valve pos after hard', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()



# f3 = plt.figure(3)
# plt.style.use('ggplot')
# plt.plot(df_zone3['time'],df_zone3['Troom'],label='RL', alpha=0.7)
# plt.plot(df_zone3_com['time'],df_zone3_com['Troom'],label='hys control ref', alpha=0.7)
# # plt.plot(df_zone1['time'],df_actions['Hardcon'],label='Hardconstraint', alpha=0.7)
# # plt.plot(df_zone1['time'],df_zone1['Valve'],label='Valve pos', alpha=0.7)
# # plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
# plt.ylabel('Temperature[C]')
# plt.xlabel('Time[Days]')
# plt.legend()

# f4 = plt.figure(4)
# plt.style.use('ggplot')
# plt.plot(df_zone4['time'],df_zone4['Troom'],label='RL', alpha=0.7)
# plt.plot(df_zone4_com['time'],df_zone4_com['Troom'],label='hys control ref', alpha=0.7)
# # plt.plot(df_zone1['time'],df_actions['Hardcon'],label='Hardconstraint', alpha=0.7)
# # plt.plot(df_zone1['time'],df_zone1['Valve'],label='Valve pos', alpha=0.7)
# # plt.plot(df_zone1_com['time'],df_zone1_com['Valve'],label='Valve ref pos', alpha=0.7)
# plt.ylabel('Temperature[C]')
# plt.xlabel('Time[Days]')
# plt.legend()



f5 = plt.figure(5)
plt.style.use('ggplot')
plt.plot(df_zone1['time'],df_actions['Supplytemp'],label='Supplytemp', alpha=0.7)
#plt.plot(df_zone1['time'],df_zone1['Tamb'],label='RL', alpha=0.7)
#plt.plot(df_zone1['time'],df_actions['Hardcon'],label='RL', alpha=0.7)
plt.plot(df_zone1['time'],df_zone1['Hcv'],label='HCV1', alpha=0.7)
plt.plot(df_zone2['time'],df_zone2['Hcv'],label='HCV2', alpha=0.7)
#plt.plot(df_zone1_com['time'],df_zone2_com['Troom'],label='hys control ref', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()



# plt.legend()

# f4 = plt.figure(4)
# plt.style.use('ggplot')
# plt.plot(df_zone1['time'],df_actions['Reward'],label='RL mix', alpha=0.7)
# plt.plot(df_zone1['time'],df_actions['Reward']-df_actions['sum_mix_reward']-df_actions['change_in_Supply_temp']+df_actions['Hardcon']-15,label='supply', alpha=0.7)
# plt.plot(df_zone1['time'],df_actions['Hardcon']-15,label='Hard_con', alpha=0.7)
# #plt.plot(df_zone1['time'],df_actions['price_reward'],label='Price_reward', alpha=0.7)
# plt.plot(df_zone1['time'],df_actions['sum_mix_reward'],label='sum mix', alpha=0.7)
# plt.plot(df_zone1['time'],df_actions['change_in_Supply_temp'],label='change in supply', alpha=0.7)
# # plt.plot(df_zone1_com['time'],df_zone1_com['Power'],label='RL zone 1', alpha=0.7)
# plt.ylabel('Reward[C]')
# plt.xlabel('Time[Days]')

# plt.legend()

#Evaluation_of_power1(m_p,n_p,SS,n_p_ref,m_p_ref,df_zone1,df_zone1_com,power20,power20_ref,number)
# Evaluation_of_power(m_p,n_p,SS,n_p_ref,m_p_ref,df_zone1,df_zone1_com,power20,power20_ref,number)
#Evaluation_of_reward(m,n,SS,n_ref,m_ref,df_zone2,df_zone2_com,reward20,reward20_ref,number,4)
#(self,m,n,SS,n_ref,m_ref,df_actions,df_actions_com,supplytemp20,reward20_ref,number,fig_num):
#Evaluation_of_supplytemp(m,n,SS,n_ref,m_ref,df_actions_com,df_actions_com,supplytemp20,number=number,fig_num=7)
Evaluation_of_error(m,n,SS,n_ref,m_ref,df_zone1,df_zone1_com,error20,error20_ref,number,15)
#Evaluation_of_supplytemp(m,n,SS_action,n_ref,m_ref,df_actions,df_actions_com,supplytemp20,supplytemp20_ref,number=number,fig_num=7)
#Evaluation_of_reward(m,n,SS,n_ref,m_ref,df_actions,df_actions_com,reward20,reward20_ref,number,6)
Evaluation_of_duty_cycle(m,n,SS,n_ref,m_ref,df_zone1,df_zone1_com,Valves20,Valves20_ref,number,10)
#Evaluation_of_duty_cycle(m,n,SS,n_ref,m_ref,df_zone2,df_zone2_com,Valves20,Valves20_ref,number,14)

#Evaluation_of_duty_cycle(m,n,SS,n_ref,m_ref,df_actions,df_zone2_com,Valves20,Valves20_ref,number,10)
#Evaluation_of_Roomtemp(m_t,n_t,SS,n_t_ref,m_t_ref,df_zone1,df_zone1_com,temp20,temp20_ref,number,9)
#Evaluation_of_price(m_pr,n_pr,SS,n_pr_ref,m_pr_ref,df_zone1,df_zone1_com,price20,price20_ref,number,30)

Evaluation_of_price(m_pr,n_pr,SS,n_pr_ref,m_pr_ref,df_zone1,df_zone1_com,price20,price20_ref,number,32)
Evaluation_of_price(m_pr,n_pr,SS,n_pr_ref,m_pr_ref,df_zone2,df_zone2_com,price20,price20_ref,number,33)

Evaluation_of_reward(m,n,SS,n_ref,m_ref,df_zone1,df_zone1_com,reward20,reward20_ref,number,22)
Evaluation_of_reward(m,n,SS,n_ref,m_ref,df_zone2,df_zone2_com,reward20,reward20_ref,number,23)
Evaluation_of_reward(m,n,SS,n_ref,m_ref,df_actions,df_actions_com,reward20,reward20_ref,number,24)

# plt.figure(44)
# df_zone1['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# df_zone1_com['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# plt.figure(55)
# df_zone2['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# df_zone2_com['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# plt.figure(44)
# df_zone1['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# df_zone1_com['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# plt.figure(55)
# df_zone2['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# df_zone2_com['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# plt.figure(66)
# df_zone3['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# df_zone3_com['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# plt.figure(77)
# df_zone4['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
# df_zone4_com['Troom'][30000:SS].plot.hist(bins=50, alpha=0.5)
plt.show()















