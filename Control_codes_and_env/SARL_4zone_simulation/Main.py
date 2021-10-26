import Rainbow

import gym
import torch
import numpy as np
import random
from reacher_gym_env import CustomReacherEnv
#from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_mixing 
from Env_master_script import Envmasterscript
from threading import Thread
Env_master_script=Envmasterscript()
# environment
env_id = "CartPole-v0"
env = gym.make(env_id)

env_UFH = CustomReacherEnv(Env_master_script)
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
env_UFH.action_space.seed(seed)

env_UFH.seed(seed)



num_frames = 218000
memory_size = 100000
batch_size = 428
target_update = 540
epsilon_decay = 1 / 5000

# train

if __name__ == "__main__":
    agent=Rainbow.DQNAgent(env_UFH,Env_master_script, memory_size, batch_size, target_update, epsilon_decay,n_step=45)
    agent.train(num_frames)
