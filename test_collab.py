import os
from citylearn import  CityLearn
from pathlib import Path
from agent import Agent
import numpy as np                                                
import torch
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold




# Load environment
climate_zone = 5
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_ids = ["Building_"+str(i) for i in [1]]
sim_period = (0, 8760)
params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':building_ids,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': sim_period, 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': True,
        'save_memory': False }

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
print(env.step(env.action_space.sample()))



# import os
from stable_baselines3 import DDPG , A2C
from stable_baselines3 import PPO

# models_dir = "models/A2C"
# logdir="logs"

# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
  
# if not os.path.exists(logdir):
#     os.makedirs(logdir)

# env.reset()

# model = A2C('MlpPolicy', env, verbose=1)

# TIMESTEPS = 8760
# for i in range(1,30): #run sur 30 epochs
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#     print("a")


# obs=env.reset()
# from stable_baselines3 import A2C
# dones=False
# list_rewards=[]
# model=A2C.load("models/A2C/96360")
# while not dones:
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   list_rewards.append(rewards)

# print(env.cost())
