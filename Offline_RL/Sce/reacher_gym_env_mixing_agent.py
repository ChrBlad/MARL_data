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
        self.name = 'Underfloorheating'
        self.current_state = None
        self._num_agents = 1
        self.sample_time = 800
        self.days_pased = 0
        self.last_price1=0
        self.last_price2=0
        self.action=0
        self.D1_world_bool=0
        self.lenght_of_epi=30
        self.NN_world_count=0
        self.reward_calculator = RewardCalculator(params)
        self.ai_input_provider = AiInputProvider(params)
        self.Env_master_script = Env_master_script
        self.last_reward = 0
        self.NN_world_bool=0
        self.print_number_NN=0
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
        
        #print('Size of action:', action_size)
        # State space information
        ######states = env_info.vector_observations
        
        #print('There are {} agents. Each observes a state with length: {}'.format(1, state_size))
        ####print('The state for the first agent looks like:', states[0])

        ##################################
        ########## ACTION SPACE ##########
        ##################################
        print("Action space")
        #low = np.array([-1],dtype='f')
        #high = np.array([1],dtype='f')
        self._action_space = spaces.Discrete(15)
        #self._action_space = spaces.Box(low, high, dtype=np.float32)

        #######################################
        ########## OBSERVATION SPACE ##########
        #######################################
        state_size = 24#+1
        print("Observation space")
        high = np.array([np.inf] * state_size)
        self._observation_space = spaces.Box(-high, high, dtype=np.float32)
        print("Done setting up")

    def step(self, action):
        itnum1,itnum2,itnum3,itnum4,itnum_mix,self.NN_world_bool,itnum_fin = self.Env_master_script.check_mix()
        while itnum_mix > itnum_fin:
            itnum1,itnum2,itnum3,itnum4,itnum_mix,self.NN_world_bool,itnum_fin = self.Env_master_script.check_mix()
            #print('tt1')
        if self.NN_world_bool == 0: 
            action = np.array(action)
            #action = self.Env_master_script.Hardconstraint_masked_action(action)
            simulation_time = self.iteration_number*self.sample_time/86400
            timeofday = (simulation_time-(simulation_time//1))*24
            itnum_mix = self.Env_master_script.Tsupply(action,self.NN_world_bool,type=1)
            Hardconstraint=self.Env_master_script.Hardconstraint_for_mixer(action)
            out,actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
            outw=out.tolist()
            while move<1: 
                out, actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
                outw=out.tolist()
                #print("13")
            self.days_pased =self.days_pased+1
            if self.days_pased > self.lenght_of_epi: 
                done =1
                self.days_pased = 0
            else:
                done = 0
            action_v1 =out['Valveout1']
            action_v2 =out['Valveout2']
            action_v3 =out['Valveout3']
            action_v4 =out['Valveout4']
            hard_v1=0
            hard_v2=0
            hard_v3=0
            hard_v4=0       
            last_reward_valve1,Price1,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone1(out,action_v1,actions_to_env,done,self.NN_world_bool,hard_v1,timeofday,self.NN_world_bool)
            last_reward_valve1,Price2,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone2(out,action_v2,actions_to_env,done,self.NN_world_bool,hard_v2,timeofday,self.NN_world_bool)
            last_reward_valve1,Price3,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone3(out,action_v3,actions_to_env,done,self.NN_world_bool,hard_v3,timeofday,self.NN_world_bool)
            last_reward_valve1,Price4,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone4(out,action_v4,actions_to_env,done,self.NN_world_bool,hard_v4,timeofday,self.NN_world_bool)
            self.last_price1=np.float64(Price1)
            self.last_price2=np.float64(Price2)
            self.last_price3=np.float64(Price3)
            self.last_price4=np.float64(Price4)
            obs = self.ai_input_provider.calculate_ai_input_mix(out,actions_to_env,timeofday,Hardconstraint,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            reward,price_reward,sum_mix_reward,change_in_Supply_temp = self.reward_calculator.calculate_reward_mix(out,Hardconstraint,tref,actions_to_env,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            

            info = {"text_observation": 'allgood', "brain_info": 0}
            data = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),reward,Hardconstraint,price_reward,sum_mix_reward,change_in_Supply_temp
            data_out=out['RoomTemperature1'],out['RoomTemperature2'],out['RoomTemperature3'],out['RoomTemperature4'],out['Valveout1'],out['Valveout2'],out['Valveout3'],out['Valveout4'],out['Power1'],out['Power2'],out['Power3'],out['Power4'],out['Tamb1'],out['Sun1'],out['Tambforecast1'],out['Tambforecast2'],out['Tambforecast3'],out['Sunforecast1'],out['Sunforecast2'],out['Sunforecast3'],out['Powerwater1'],out['Powerwater2'],out['Powerwater3'],out['Powerwater4'],float(out['Treturn1']),float(out['Treturn2']),float(out['Treturn3']),float(out['Treturn4']),float(out['Flow1']),float(out['Flow2']),float(out['Flow3']),float(out['Flow4'])
            self.ai_input_provider.savefun_out(data_out)              
            #data_out=out['RoomTemperature1'].tolist(),out['Valveout1'].tolist(),out['Power1'].tolist(),out['Tamb1'].tolist(),out['Sun1'].tolist(),out['Tambforecast1'].tolist(),out['Tambforecast2'].tolist(),out['Tambforecast3'].tolist(),out['Sunforecast1'].tolist(),out['Sunforecast2'].tolist(),out['Sunforecast3'].tolist(),out['Powerwater'].tolist()
            #data_out=out['RoomTemperature1'],out['Valveout1'],out['Power1'],out['Tamb1'],out['Sun1'],out['Tambforecast1'],out['Tambforecast2'],out['Tambforecast3'],out['Sunforecast1'],out['Sunforecast2'],out['Sunforecast3'],out['Powerwater']
            #print(data_out)
            self.ai_input_provider.savefun_actiondata(data)
            #self.ai_input_provider.savefun_out(data_out)
            self.iteration_number = 1+self.iteration_number
        else:
            NN_world_simulation_time = self.NN_world_count*self.sample_time/86400
            action = np.array(action)
            #print('enterNN_Mix')
            #action = self.Env_master_script.Hardconstraint_masked_action(action)
            itnum_mix = self.Env_master_script.Tsupply(action,self.NN_world_bool,type=1)
            Hardconstraint=self.Env_master_script.Hardconstraint_for_mixer(action)
            out,actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
            outw=out
            while move<1: 
                out, actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
                outw=out
                #print("15")
                #print('stuck in mix_NN')
            done=done_NN
            action_v1 =out['Valveout1']
            action_v2 =out['Valveout2']
            action_v3 =out['Valveout3']
            action_v4 =out['Valveout4']
            hard_v1=0
            hard_v2=0
            hard_v3=0
            hard_v4=0       
            last_reward_valve1,Price1,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone1(out,action_v1,actions_to_env,done,self.NN_world_bool,hard_v1,timeofday_NN,self.NN_world_bool)
            last_reward_valve1,Price2,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone2(out,action_v2,actions_to_env,done,self.NN_world_bool,hard_v2,timeofday_NN,self.NN_world_bool)
            last_reward_valve1,Price3,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone3(out,action_v3,actions_to_env,done,self.NN_world_bool,hard_v3,timeofday_NN,self.NN_world_bool)
            last_reward_valve1,Price4,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone4(out,action_v4,actions_to_env,done,self.NN_world_bool,hard_v4,timeofday_NN,self.NN_world_bool)
            self.last_price1=np.float64(Price1)
            self.last_price2=np.float64(Price2)
            self.last_price3=np.float64(Price3)
            self.last_price4=np.float64(Price4)
            obs = self.ai_input_provider.calculate_ai_input_mix(out,actions_to_env,timeofday_NN,Hardconstraint,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            reward,price_reward,sum_mix_reward,change_in_Supply_temp = self.reward_calculator.calculate_reward_mix(out,Hardconstraint,tref,actions_to_env,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            info = {"text_observation": 'allgood', "brain_info": 0}
            data = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),float(reward),Hardconstraint,price_reward,float(sum_mix_reward),change_in_Supply_temp
            self.ai_input_provider.savefun_actiondata_NN(data)
        return obs, reward, done, info


    def reset(self):
        itnum1,itnum2,itnum3,itnum4,itnum_mix,self.NN_world_bool,itnum_fin = self.Env_master_script.check_mix()
        while itnum_mix > itnum_fin:
            itnum1,itnum2,itnum3,itnum4,itnum_mix,self.NN_world_bool,itnum_fin = self.Env_master_script.check_mix()
            #print('tt2')
        if self.NN_world_bool == 0: 
            action = 7#np.array(action)
            simulation_time = self.iteration_number*self.sample_time/86400
            timeofday = (simulation_time-(simulation_time//1))*24
            itnum_mix = self.Env_master_script.Tsupply(action,self.NN_world_bool,type=1)
            Hardconstraint=0#self.Env_master_script.Hardconstraint_for_mixer(action)
            out,actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
            outw=out.tolist()
            while move<1: 
                out, actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
                outw=out.tolist()
                #print("13")
            self.days_pased =self.days_pased+1
            if self.days_pased > self.lenght_of_epi: 
                done =1
                self.days_pased = 0
            else:
                done = 0
            action_v1 =out['Valveout1']
            action_v2 =out['Valveout2']
            action_v3 =out['Valveout3']
            action_v4 =out['Valveout4']
            hard_v1=0
            hard_v2=0
            hard_v3=0
            hard_v4=0       
            last_reward_valve1,Price1,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone1(out,action_v1,actions_to_env,done,self.NN_world_bool,hard_v1,timeofday,self.NN_world_bool)
            last_reward_valve1,Price2,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone2(out,action_v2,actions_to_env,done,self.NN_world_bool,hard_v2,timeofday,self.NN_world_bool)
            last_reward_valve1,Price3,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone3(out,action_v3,actions_to_env,done,self.NN_world_bool,hard_v3,timeofday,self.NN_world_bool)
            last_reward_valve1,Price4,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone4(out,action_v4,actions_to_env,done,self.NN_world_bool,hard_v4,timeofday,self.NN_world_bool)
            self.last_price1=np.float64(Price1)
            self.last_price2=np.float64(Price2)
            self.last_price3=np.float64(Price3)
            self.last_price4=np.float64(Price4)
            obs = self.ai_input_provider.calculate_ai_input_mix(out,actions_to_env,timeofday,Hardconstraint,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            reward,price_reward,sum_mix_reward,change_in_Supply_temp = self.reward_calculator.calculate_reward_mix(out,Hardconstraint,tref,actions_to_env,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            
            data_out=out['RoomTemperature1'],out['RoomTemperature2'],out['RoomTemperature3'],out['RoomTemperature4'],out['Valveout1'],out['Valveout2'],out['Valveout3'],out['Valveout4'],out['Power1'],out['Power2'],out['Power3'],out['Power4'],out['Tamb1'],out['Sun1'],out['Tambforecast1'],out['Tambforecast2'],out['Tambforecast3'],out['Sunforecast1'],out['Sunforecast2'],out['Sunforecast3'],out['Powerwater1'],out['Powerwater2'],out['Powerwater3'],out['Powerwater4'],float(out['Treturn1']),float(out['Treturn2']),float(out['Treturn3']),float(out['Treturn4']),float(out['Flow1']),float(out['Flow2']),float(out['Flow3']),float(out['Flow4'])
            self.ai_input_provider.savefun_out(data_out)   
            info = {"text_observation": 'allgood', "brain_info": 0}
            data = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),reward,Hardconstraint,price_reward,sum_mix_reward,change_in_Supply_temp
            #data_out=out['RoomTemperature1'],out['Valveout1'],out['Power1'],out['Tamb1'],out['Sun1'],out['Tambforecast1'],out['Tambforecast2'],out['Tambforecast3'],out['Sunforecast1'],out['Sunforecast2'],out['Sunforecast3'],out['Powerwater']
            self.ai_input_provider.savefun_actiondata(data)
            #self.ai_input_provider.savefun_out(data_out)
            self.iteration_number = 1+self.iteration_number
        else:
            action = 7#np.array(action)
            itnum_mix = self.Env_master_script.Tsupply(action,self.NN_world_bool,type=1)
            Hardconstraint=0#self.Env_master_script.Hardconstraint_for_mixer(action)
            out,actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
            outw=out
            while move<1: 
                out, actions_to_env,tref,self.NN_world_bool,timeofday_NN,done_NN,move = self.Env_master_script.outmixing(itnum_mix)
                outw=out
                #print("15")
                #print('stuck in mix_NN')
            done=done_NN
            action_v1 =out['Valveout1']
            action_v2 =out['Valveout2']
            action_v3 =out['Valveout3']
            action_v4 =out['Valveout4']
            hard_v1=0
            hard_v2=0
            hard_v3=0
            hard_v4=0       
            last_reward_valve1,Price1,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone1(out,action_v1,actions_to_env,done,self.NN_world_bool,hard_v1,timeofday_NN,self.NN_world_bool)
            last_reward_valve1,Price2,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone2(out,action_v2,actions_to_env,done,self.NN_world_bool,hard_v2,timeofday_NN,self.NN_world_bool)
            last_reward_valve1,Price3,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone3(out,action_v3,actions_to_env,done,self.NN_world_bool,hard_v3,timeofday_NN,self.NN_world_bool)
            last_reward_valve1,Price4,Power1,hard_conV1 = self.reward_calculator.calculate_reward_Zone4(out,action_v4,actions_to_env,done,self.NN_world_bool,hard_v4,timeofday_NN,self.NN_world_bool)
            self.last_price1=np.float64(Price1)
            self.last_price2=np.float64(Price2)
            self.last_price3=np.float64(Price3)
            self.last_price4=np.float64(Price4)
            obs = self.ai_input_provider.calculate_ai_input_mix(out,actions_to_env,timeofday_NN,Hardconstraint,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            reward,price_reward,sum_mix_reward,change_in_Supply_temp = self.reward_calculator.calculate_reward_mix(out,Hardconstraint,tref,actions_to_env,self.last_price1,self.last_price2,self.last_price3,self.last_price4)
            info = {"text_observation": 'allgood', "brain_info": 0}
            data = out['Valveout1'],np.float64(actions_to_env['SupplyTemperature']),float(reward),Hardconstraint,price_reward,float(sum_mix_reward),change_in_Supply_temp
            #data_out = out
            self.ai_input_provider.savefun_actiondata_NN(data)
        #print('call reset mix',obs)
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