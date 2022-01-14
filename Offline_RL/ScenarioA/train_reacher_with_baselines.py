# Workaround for the following error
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized
import os
from Env_master_script import Envmasterscript
from multiprocessing import Process, freeze_support
from threading import Thread
import tensorflow as tf
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Imports
import numpy as np

Env_master_script=Envmasterscript()
########################################
# Import custom environment
#sfrom reacher_gym_env import CustomReacherEnv
from reacher_gym_env_mixing_agent import CustomReacherEnv as CustomReacherEnv_mixing
########################################
# Import Stable Baselines vector wrapper for environments
from stable_baselines.common.vec_env import DummyVecEnv#, SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_mixing 
########################################
# Import Reinforcement Learning libraries from Stable Baselines
### DDPG
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy as ddpgLnMlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
### HER
# from stable_baselines.her import HER
### SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines import SAC

from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC2
from stable_baselines import SAC as SAC2
### PPO1, TRPO


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy as MlpPolicy1
from stable_baselines.common.policies import MlpPolicy as MlpPolicymix#, MlpLstmPolicy, MlpLnLstmPolicy

from stable_baselines import PPO1
from stable_baselines import PPO2 
from stable_baselines import PPO2 as PPO2_mix
from stable_baselines import PPO2 as PPO2_2

########################################
# Create custom environment
# env1 = CustomReacherEnv(Env_master_script)
# env1 = DummyVecEnv([lambda: env1])


env_mixing = CustomReacherEnv_mixing(Env_master_script)
env_mixing = DummyVecEnv_mixing([lambda: env_mixing])

policy_kwargs = {
    'layers' : [20,20,20,20]
}
policy_kwargs2 = {
    'layers' : [20,20,20,20]
}

policy_kwargs3 = {
    'layers' : [1]
}

policy_kwargs4 = {
    'layers' : [50,50,50,50,50,50,50,50,30]
}

policy_kwargs3SAC = {
    'layers' : [500,500]
}
policy_kwargsa = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
#policy_kwargs5  = dict(act_fun=tf.nn.softmax, net_arch=[100,100])

#policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
#tensorboard --logdir=log_bord --port 6006
from stable_baselines.deepq.policies import LnMlpPolicy as MlpPolicydqn
from stable_baselines import DQN
from stable_baselines import TRPO


from stable_baselines.common.policies import MlpLstmPolicy, LstmPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy as MlpPolicy_mix
#from stable_baselines.common import make_vec_env
#from stable_baselines.common import make_vec_env as make_vec_env_mix
from stable_baselines import A2C
from stable_baselines import A2C as A2C_mix
from stable_baselines import TD3
#act_fun=tf.nn.softmax,
class CustomLSTMPolicy1(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch=1, n_lstm=10, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, net_arch=[10, 'lstm', dict(vf=[10, 10,10,10,10], pi=[10, 10,10,10,10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

class CustomLSTMPolicy2(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=80, reuse=True, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, net_arch=[80, 'lstm', dict(vf=[50,50,50,50,50,50], pi=[50,50,50,50,50,50])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)
from stable_baselines import ACKTR
from stable_baselines import ACER
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.td3.policies import MlpPolicy as MlpPolicyTD3
register_policy('CustomLSTMPolicy2', CustomLSTMPolicy2)
#CustomLSTMPolicy2

env_id = "CartPole-v0"
#env = gym.make(env_id)
Training = 1
if Training > 0.1:
    #LR change from 0.03 to 0.01
    model5 = DQN(MlpPolicydqn,env_mixing,gamma=0.9,verbose=1,seed=150,learning_rate=0.03,n_cpu_tf_sess=1,target_network_update_freq=540, learning_starts=1080, exploration_fraction=0.02,buffer_size=100000,prioritized_replay=False, batch_size=2160)#,,policy_kwargs=policy_kwargs3policy_kwargs=policy_kwargs31728)#,policy_kwargs=policy_kwargs5,exploration_final_eps=0.1
    #model5 = PPO2(MlpPolicy,env_mixing,gamma=0.9,verbose=1,seed=100,n_cpu_tf_sess=1,nminibatches=20,n_steps=100,noptepochs=4,learning_rate=0.001,cliprange_vf=-1)80#,n_steps=50,,noptepochs=1,ent_coef=0.15,nminibatches=10,policy_kwargs=policy_kwargs4,learning_rate=0.0001,learning_rate=0.0007,n_steps=20,,n_steps=216,policy_kwargs=policy_kwargs3
    #model5 = PPO(MlpPolicy,env_mixing,gamma=0.9,verbose=1,seed=1001085,n_cpu_tf_sess=1,nminibatches=20,n_steps=100,noptepochs=4,learning_rate=0.001,cliprange_vf=-1)#,n_steps=50,,noptepochs=1,ent_coef=0.15,nminibatches=10,policy_kwargs=policy_kwargs4,learning_rate=0.0001,learning_rate=0.0007,n_steps=20,,n_steps=216,policy_kwargs=policy_kwargs3
    model5.learn(108500,log_interval=50,tb_log_name="first_run_LSTM")
    model5.save("Mixing_agent")
else:
    model = DQN.load("Mixing_agent")

    obs = env_mixing.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_mixing.step(action)
print("########################################")
print("Done")