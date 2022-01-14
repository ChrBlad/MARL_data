from functions import fun
import matplotlib.pyplot as plt
fun=fun()
dataset,df_out,df_zone1,df_actions=fun.load_data()
#
#self.df_out.columns = ['RoomTemperature1','Valveout1','Power1','Tamb1','Sun1','Tambforecast1','Tambforecast2','Tambforecast3','Sunforecast1','Sunforecast2','Sunforecast3','Powerwater']
#self.dataset = data[['Troom', 'Tamb', 'Valveout1','Supplytemp','TOD','Sun1','Sunforecast1','Tambforecast1']].copy()
#constants


rho_water=1000
C_water=4186
C_room=1005
C_floor=750
m_water=28.35#asumtions about lengt of pip - 100m diameter 0.019
m_room=63.8 #asumtions about room - 20m2 and 2.5m in hight
m_floor=3000 # concret 0.2m over 20m24000#
flow=100/3600000
count=0
dt=10
Bw=500
Br=343
Ba=37
Bs=0.4
#initial guess
T_room=22+273.15
T_floor=22+273.15
T_supply=22+273.15
T_return=22+273.15


print(1/(m_water*C_water))
print(1/(m_floor*C_floor))
print(1/(m_room*C_room))



# heatflux equations

def temp_fun(T_room,T_floor,T_return,T_amb,flow,T_supply,Power_sun,Br,Bw,Ba,Bs):
    for n in range(80):
        HF_return=(rho_water*C_water*flow*(T_supply-T_return)-Bw*(T_return-T_floor))/(m_water*C_water)#*1.52047317e-5#
        HF_floor=(Bw*(T_return-T_floor)-Br*(T_floor-T_room))/(m_floor*C_floor)#*1.33333333e-6#
        HF_room=(Br*(T_floor-T_room)-Ba*(T_room-T_amb)+Bs*Power_sun)/(m_room*C_room)#+Bf*(T_free-T_room)+Bsun*(T_sun-T_room)*0.00002#


        T_room=HF_room*dt+T_room
        T_floor=HF_floor*dt+T_floor
        T_return=HF_return*dt+T_return
    return T_room,T_floor,T_return

T_room_list=[]
T_room_dymola_list=[]
Time_list=[]
time=0
for i in range(5000):
    
    valve=dataset['Valveout1'][i]
    T_supply=dataset['Supplytemp_t'][i]
    Power_sun=dataset['Sun1_t'][i]
    T_amb=dataset['Tamb_t'][i]
    T_room_dymola=dataset['Troom_t'][i]
    if valve ==1:
        flow=80/3600000
    else:
        flow=0
    
    T_room,T_floor,T_return=temp_fun(T_room,T_floor,T_return,T_amb,flow,T_supply,Power_sun,Br,Bw,Ba,Bs)
    time=(i*800)/(24*3600)
    T_room_list.append(T_room)
    Time_list.append(time)
    T_room_dymola_list.append(T_room_dymola)
    # if count==500:
    #     T_room=20+273.15
    #     T_floor=23+273.15
    #     T_supply=40+273.15
    #     T_return=24+273.15
    #     count=0
    count=count+1

plt.plot(Time_list,T_room_list,label='1D', alpha=0.7)
plt.plot(Time_list,T_room_dymola_list,label='Dymola', alpha=0.7)
plt.ylabel('Temperature[C]')
plt.xlabel('Time[Days]')
plt.legend()

plt.show()