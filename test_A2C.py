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

##################################################################################################################
# Load environment


climate_zone = 5
data_path = Path("data/Climate_Zone_"+str(climate_zone))
sim_period = (0, 8760) #1 year
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
env.step(env.observation_space.step())



# ########A2C##########
# from stable_baselines3.common.evaluation import evaluate_policy

# model = A2C("MlpPolicy", env, verbose=1)
# model.save("a2c_1building")

# # mean_reward_before_train = evaluate_policy(model, env, n_eval_episodes=1)
# # print(mean_reward_before_train)
# cost=[]
# n_epochs=1

# for i in range (n_epochs):
#     model.learn(total_timesteps=8760*i) #train on i epochs
#     obs = env.reset()
#     dones=False
#     list_rewards=[]
    
#     for i in range(2):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         list_rewards.append(rewards)
#     # cost.append(env.cost())
    
#     # print(env.cost()['net_electricity_consumption'])
    
# print(list_rewards)
# plt.plot(n_epochs, [cost[i]['net_electricity_consumption'] for i in range(n_epochs)])
# plt.show()



# mean_100ep_reward = round(np.mean(list_rewards[-100:]), 1)
# print("Mean reward 100 derniers j:", mean_100ep_reward, "Num episodes:", len(list_rewards))



# #########################################################################
# ##plot
# interval = range(sim_period[0], sim_period[1])
# plt.figure(figsize=(16,5))
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RBC for storage(kW)'])
# plt.show()

########################################################################





# ######CallBack function##########

# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq:
#     :param log_dir: Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: Verbosity level.
#     """
#     def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.save_path = os.path.join(log_dir, 'best_model')
#         self.best_mean_reward = -np.inf

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:

#           # Retrieve training reward
#           x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#           if len(x) > 0:
#               # Mean training reward over the last 100 episodes
#               mean_reward = np.mean(y[-100:])
#               if self.verbose > 0:
#                 print(f"Num timesteps: {self.num_timesteps}")
#                 print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

#               # New best model, you could save the agent here
#               if mean_reward > self.best_mean_reward:
#                   self.best_mean_reward = mean_reward
#                   # Example for saving best model
#                   if self.verbose > 0:
#                     print(f"Saving new best model to {self.save_path}")
#                   self.model.save(self.save_path)

#         return True

# # Create log dir
# log_dir = "tmp/"
# os.makedirs(log_dir, exist_ok=True)

# # Create and wrap the environment
# env = CityLearn(**params)
# env = Monitor(env, log_dir)

# model = A2C("MlpPolicy", env, verbose=1)
# # Create the callback: check every 1000 steps
# callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
# # Train the agent
# timesteps = 1000
# model.learn(total_timesteps=int(timesteps), callback=callback)
# model.save("a2c_1building")
# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "a2c_1building")
# plt.show()

