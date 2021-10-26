'''Custom wrapper for unity ml agents Reacher environment
'''
import time
import gym
from gym import spaces
import numpy as np
from fmu_stepping import fmu_stepping
from fmpy.util import plot_result
from parameters import Params
from reward_calculator import RewardCalculator
from ai_input_provider import AiInputProvider
#from Env_master_script import Envmasterscript
#from parameters import Params
######from unityagents import UnityEnvironment
# from gym_unity.envs.unity_env import UnityEnv

class CustomReacherEnv(gym.Env):
    '''Custom wrapper
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self,Env_master_script, *args, **kwargs):
        super(CustomReacherEnv, self).__init__()
        print("Initializing Reacher environment ...")
        # Define action and observation space
        # They must be gym.spaces
        # Example when using discrete actions:
        # Run without rendering: no_graphics=True
        
        params = Params
        #fmu_stepping  =  fmu_stepping 
        self.name = 'Underfloorheating'
        self.current_state = None
        self._num_agents = 1
        self.sample_time = 800
        self.reward_calculator = RewardCalculator(params)
        self.ai_input_provider = AiInputProvider(params)
        self.Env_master_script = Env_master_script
        self.last_reward = 0
        self.days_pased = 0
        self.scores = []
        self.action = 0
        self.last_temp = 273.15 + 40
        self.last_alfa = 0
        self.iteration_number = 0
        self.params = params
        self.Old_Valve1 = 0
        self.Old_Valve2 = 0
        self.T1 = 273.15+22
        self.datalist =[]
        self.T2 = 273.15+22
        self.T3 = 273.15+22
        self.T4 = 273.15+22
        self.params.goalT1 = 273.15+22
        self.params.goalT2 = 273.15+22
        self.params.goalT3 = 273.15+22
        self.params.goalT4 = 273.15+22
#         self.Old_SupplyTemperature = 340
#         filename = 'TwoElementHouse_01room_0FMU_0nohys_0onlywinter_Houses_testHouse.fmu'
#         start_time=0.0
#         stop_time=3155692600#100Years#31556926oneyearinSecounds
#         self.sample_time=800
#         parameters={}
#         input={'SupplyTemperature','Valve1'}
#         self.input_values ={'SupplyTemperature': 340,'Valve1': 1}
#         output={'RoomTemperature1','Tamb','Valve1out'}
#         # Setup the FMU
#         FMU_stepping = fmu_stepping(filename = filename,
#                             start_time=start_time,
#                             stop_time=stop_time,
#                             sample_time=self.sample_time,
#                             parameters=parameters,
#                             input=input,
#                             output=output
# )
#         self.env = FMU_stepping
        # Environments contain **_brains_** which are responsible for deciding the
        # actions of their associated agents. Here we check for the first brain
        # available, and set it as the default brain we will be controlling from Python.
        # Get the default brain
        self.brain_name = 'Underfloorheating_brain'##self.env.brain_names[0]
        brain = self.brain_name##self.env.brains[self.brain_name]
        print("Brain name:", self.brain_name)
        print("Brain:", brain)
        # Print out information about the state and action spaces
        # Reset the environment
        #env_info = self.env.reset()[self.brain_name]
        # Number of agents in the environment
        self._num_agents = 1
        #####print('Number of agents:', self._num_agents)
        # Size of each action
        action_size =31#30#brain.vector_action_space_size
        print('Size of action:', action_size)
        # State space information
        ######states = env_info.vector_observations
        state_size = 29*10#states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(1, state_size))
        ####print('The state for the first agent looks like:', states[0])

        ##################################
        ########## ACTION SPACE ##########
        ##################################
        print("Action space")
        #low = np.array([-1],dtype='f')
        #high = np.array([1],dtype='f')
        self._action_space = spaces.Discrete(action_size)
        #self._action_space = spaces.Box(low, high, dtype=np.float32)

        #######################################
        ########## OBSERVATION SPACE ##########
        #######################################
        #state_size=12*10
        print("Observation space")
        high = np.array([np.inf] *state_size)
        #self._observation_space = spaces.Discrete(state_size)
        self._observation_space = spaces.Box(-high, high, dtype=np.float32)
        print("Done setting up")

    def step(self, action):
        #print('Step reacher 1')
        self.iteration_number = 1+self.iteration_number 
        info = 0
        simulation_time = self.iteration_number*self.sample_time/86400
        timeofday = (simulation_time-(simulation_time//1))*24
        action = np.array(action)
        Hardconstraint=self.Env_master_script.Hardconstraint_for_mixer()
        Hard_v1=self.Env_master_script.Hardconstraint_send_V1()
        Hard_v2=self.Env_master_script.Hardconstraint_send_V2()
        Hard_v3=self.Env_master_script.Hardconstraint_send_V3()
        Hard_v4=self.Env_master_script.Hardconstraint_send_V4()
        out,actions_to_env,ref = self.Env_master_script.sendaction()
        last_reward1,Price1,Power1 = self.reward_calculator.calculate_reward_Zone1(out,actions_to_env,Hard_v1,timeofday)
        last_reward2,Price2,Power2 = self.reward_calculator.calculate_reward_Zone2(out,actions_to_env,Hard_v2,timeofday)  
        last_reward3,Price3,Power3 = self.reward_calculator.calculate_reward_Zone3(out,actions_to_env,Hard_v3,timeofday)
        last_reward4,Price4,Power4 = self.reward_calculator.calculate_reward_Zone4(out,actions_to_env,Hard_v4,timeofday)  
        mixing_reward, price_reward,sum_mix_reward,change_in_Supply_temp = self.reward_calculator.calculate_reward_mix( out,Hardconstraint,ref,actions_to_env,Price1,Price2,Price3,Price4)
        obs = self.ai_input_provider.calculate_ai_input_Zone1(out,actions_to_env,timeofday,Price1,Price2,Price3,Price4,Hard_v1,Hard_v2,Hard_v3,Hard_v4,Hardconstraint)       
        reward=mixing_reward+last_reward1+last_reward2+last_reward3+last_reward4
        self.days_pased =self.days_pased+1 
        if self.days_pased > 108: 
            done =1#0# 
            self.days_pased = 0
        else:
            done = 0
        #print('Time in simulationsssssssss',simulation_time)
        info = {"text_observation": 'allgood', "brain_info": 0}
        data_action = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),reward,Hardconstraint,price_reward,sum_mix_reward,change_in_Supply_temp
        data1 = out['RoomTemperature1'].tolist(),out['Tamb1'].tolist(),out['Valveout1'].tolist(),out['Power1'].tolist(), last_reward1,np.float64(action),np.float64(Price1),Power1,out['Powerwater1'].tolist(),Hard_v1,simulation_time
        data2 = out['RoomTemperature2'].tolist(),out['Tamb1'].tolist(),out['Valveout2'].tolist(),out['Power2'].tolist(), last_reward2,np.float64(action),np.float64(Price2),Power2,out['Powerwater2'].tolist(),Hard_v2,simulation_time
        data3 = out['RoomTemperature3'].tolist(),out['Tamb1'].tolist(),out['Valveout3'].tolist(),out['Power3'].tolist(), last_reward3,np.float64(action),np.float64(Price3),Power3,out['Powerwater3'].tolist(),Hard_v3,simulation_time
        data4 = out['RoomTemperature4'].tolist(),out['Tamb1'].tolist(),out['Valveout4'].tolist(),out['Power4'].tolist(), last_reward4,np.float64(action),np.float64(Price4),Power4,out['Powerwater4'].tolist(),Hard_v4,simulation_time
        self.ai_input_provider.savefun_actiondata(data_action)
        self.ai_input_provider.savefun_Zone1(data1)
        self.ai_input_provider.savefun_Zone2(data2)
        self.ai_input_provider.savefun_Zone3(data3)
        self.ai_input_provider.savefun_Zone4(data4)
    
        #print('return reacher1')
        return obs, reward, done, info


    def reset(self):
        self.iteration_number = 1+self.iteration_number 
        info = 0
        action=0
        simulation_time = self.iteration_number*self.sample_time/86400
        timeofday = (simulation_time-(simulation_time//1))*24
        Hardconstraint=self.Env_master_script.Hardconstraint_for_mixer()
        Hard_v1=self.Env_master_script.Hardconstraint_send_V1()
        Hard_v2=self.Env_master_script.Hardconstraint_send_V2()
        Hard_v3=self.Env_master_script.Hardconstraint_send_V3()
        Hard_v4=self.Env_master_script.Hardconstraint_send_V4()
        out,actions_to_env,ref = self.Env_master_script.sendaction()
        last_reward1,Price1,Power1 = self.reward_calculator.calculate_reward_Zone1(out,actions_to_env,Hard_v1,timeofday)
        last_reward2,Price2,Power2 = self.reward_calculator.calculate_reward_Zone2(out,actions_to_env,Hard_v2,timeofday)  
        last_reward3,Price3,Power3 = self.reward_calculator.calculate_reward_Zone3(out,actions_to_env,Hard_v3,timeofday)
        last_reward4,Price4,Power4 = self.reward_calculator.calculate_reward_Zone4(out,actions_to_env,Hard_v4,timeofday)  
        mixing_reward, price_reward,sum_mix_reward,change_in_Supply_temp = self.reward_calculator.calculate_reward_mix( out,Hardconstraint,ref,actions_to_env,Price1,Price2,Price3,Price4)
        obs = self.ai_input_provider.calculate_ai_input_Zone1(out,actions_to_env,timeofday,Price1,Price2,Price3,Price4,Hard_v1,Hard_v2,Hard_v3,Hard_v4,Hardconstraint)       
        reward=mixing_reward+last_reward1+last_reward2+last_reward3+last_reward4
        #print('Time in simulationsssssssss',simulation_time)
        info = {"text_observation": 'allgood', "brain_info": 0}
        data_action = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),reward,Hardconstraint,price_reward,sum_mix_reward,change_in_Supply_temp
        data1 = out['RoomTemperature1'].tolist(),out['Tamb1'].tolist(),out['Valveout1'].tolist(),out['Power1'].tolist(), last_reward1,np.float64(action),np.float64(Price1),Power1,out['Powerwater1'].tolist(),Hard_v1,simulation_time
        data2 = out['RoomTemperature2'].tolist(),out['Tamb1'].tolist(),out['Valveout2'].tolist(),out['Power2'].tolist(), last_reward2,np.float64(action),np.float64(Price2),Power2,out['Powerwater2'].tolist(),Hard_v2,simulation_time
        data3 = out['RoomTemperature3'].tolist(),out['Tamb1'].tolist(),out['Valveout3'].tolist(),out['Power3'].tolist(), last_reward3,np.float64(action),np.float64(Price3),Power3,out['Powerwater3'].tolist(),Hard_v3,simulation_time
        data4 = out['RoomTemperature4'].tolist(),out['Tamb1'].tolist(),out['Valveout4'].tolist(),out['Power4'].tolist(), last_reward4,np.float64(action),np.float64(Price4),Power4,out['Powerwater4'].tolist(),Hard_v4,simulation_time
        self.ai_input_provider.savefun_actiondata(data_action)
        self.ai_input_provider.savefun_Zone1(data1)
        self.ai_input_provider.savefun_Zone2(data2)
        self.ai_input_provider.savefun_Zone3(data3)
        self.ai_input_provider.savefun_Zone4(data4)
        return obs#, reward, done, info

    def render(self, mode='human', close=False):
        pass



    def close(self):
        """Override _close in your subclass to perform any necessary cleanup..transpose()
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        print("Closing Reacher environment")
        #result = FMU_stepping.cleanup()
        #self.env.close()
    
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._num_agents


# def main():
    # # env = UnityEnv('./unity-ml-agents/Reacher_20_agents.app', worker_id=0, use_visual=False, multiagent=True)
    # print('main()')
    # reacher_env = CustomReacherEnv()
    # obs = reacher_env.observation_space
    # reacher_env.reset()
    # reacher_env.step([1,1,1,1])
    # # reacher_env.step([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    # #reacher_env.close()


# if __name__ == '__main__':
#     main()
def main():
    # env = UnityEnv('./unity-ml-agents/Reacher_20_agents.app', worker_id=0, use_visual=False, multiagent=True)
    print('main()')
    reacher_env = CustomReacherEnv()
    obs = reacher_env.observation_space
    reacher_env.reset()
    reacher_env.step([1,1,1,1])
    # reacher_env.step([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    reacher_env.close()


if __name__ == '__main__':
    main()