import Rainbow_1
import Rainbow_2
import Rainbow_3
import Rainbow_4
import Rainbow_mix
import gym
import torch
import numpy as np
import random
from reacher_gym_env import CustomReacherEnv
from reacher_gym_env_zone2 import CustomReacherEnv_zone2
from reacher_gym_env_zone3 import CustomReacherEnv_zone3
from reacher_gym_env_zone4 import CustomReacherEnv_zone4
from reacher_gym_env_mixing_agent import CustomReacherEnv as CustomReacherEnv_mixing
#from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_mixing 
from Env_master_script import Envmasterscript
from threading import Thread


#hh=8
Env_master_script=Envmasterscript()
model1,model2,model3,model4=Env_master_script.load()
a=np.zeros((6, 8))
Point_in_time_data=np.expand_dims(a, axis=0)

for i in range(30):
    prediction=Env_master_script.serve1(Point_in_time_data,model1[i])#model.predict(Point_in_time_data)

    prediction=Env_master_script.serve2(Point_in_time_data,model2[i])

    prediction=Env_master_script.serve3(Point_in_time_data,model3[i])

    prediction=Env_master_script.serve4(Point_in_time_data,model4[i])        

initial=Env_master_script.build_NN_world()
# environment
env_id = "CartPole-v0"
env = gym.make(env_id)

env_mixing = CustomReacherEnv_mixing(Env_master_script)
env_Z1 = CustomReacherEnv(Env_master_script)
env_Z2 = CustomReacherEnv_zone2(Env_master_script)
env_Z3 = CustomReacherEnv_zone3(Env_master_script)
env_Z4 = CustomReacherEnv_zone4(Env_master_script)
#env_mixing = DummyVecEnv_mixing([lambda: env_mixing])


seed = 1

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
env_mixing.action_space.seed(seed)
env_Z1.action_space.seed(seed)
env_Z2.action_space.seed(seed)
env_Z3.action_space.seed(seed)
env_Z4.action_space.seed(seed)
env_mixing.seed(seed)
env_Z1.seed(seed)
env_Z2.seed(seed)
env_Z3.seed(seed)
env_Z4.seed(seed)


num_frames = 218000
memory_size = 100000
batch_size = 428
target_update = 540
epsilon_decay = 1 / 5000

# train

if __name__ == "__main__":
    agent_Z1 = Rainbow_1.DQNAgent1(env_Z1,Env_master_script, memory_size, batch_size, target_update, epsilon_decay,n_step=1)
    thread1 = Thread(target=agent_Z1.train, args=(num_frames,))
    agent_Z2 = Rainbow_2.DQNAgent1(env_Z2,agent_Z1,Env_master_script, memory_size, batch_size, target_update, epsilon_decay,n_step=1)
    thread2 = Thread(target=agent_Z2.train, args=(num_frames,))
    agent_Z3 = Rainbow_3.DQNAgent1(env_Z3,agent_Z1,Env_master_script, memory_size, batch_size, target_update, epsilon_decay,n_step=1)
    thread3 = Thread(target=agent_Z3.train, args=(num_frames,))
    agent_Z4 = Rainbow_4.DQNAgent1(env_Z4,agent_Z1,Env_master_script, memory_size, batch_size, target_update, epsilon_decay,n_step=1)
    thread4 = Thread(target=agent_Z4.train, args=(num_frames,))
    agent_mix = Rainbow_mix.DQNAgentmix(env_mixing,agent_Z1,agent_Z2,agent_Z3,agent_Z4,Env_master_script, memory_size, batch_size, target_update, epsilon_decay,n_step=1)
    thread_mix = Thread(target=agent_mix.train, args=(num_frames,))
    thread_mix.start()
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread_mix.join()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

