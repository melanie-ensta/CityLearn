import os
import sys
sys.path.insert(0,'..')
from citylearn import  CityLearn
from pathlib import Path
import numpy as np                                   
import torch
import matplotlib.pyplot as plt
from agents.sac import SAC
import gym
from stable_baselines3 import A2C

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



##################################################################################################################
# Load environment


climate_zone = 5
data_path = Path("data/Climate_Zone_"+str(climate_zone))
# sim_period = (0, 8760*4-1)
sim_period = (0, 8760) #1 an 
building_ids = ["Building_"+str(i) for i in [1]]
params = {'data_path':data_path, 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':building_ids,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': sim_period, 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': True, #à choisir en fonction de si on veut single or MARL
        'save_memory': False }

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
#############################################################################################################



# # ######DDPG###########


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=8760*2, log_interval=10)
model.save("ddpg_building1")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_building1")

obs = env.reset()
dones=False
c=[]
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    c.append(rewards)


# plt.plot(np.arange(8760),c)
# plt.show()
mean_100ep_reward = round(np.mean(c[-100:]), 1)
print("Mean reward 100 derniers j:", mean_100ep_reward, "Num episodes:", len(c))

print(env.cost())

# plt.plot(np.arange(8760),c[0:8760])
# plt.show()
