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


##################################################################################################################
# Load environment

climate_zone = 5
data_path = Path("data/Climate_Zone_"+str(climate_zone))
# sim_period = (0, 8760*4-1)
sim_period = (0, 300)
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
# print(env.step([0.78]))
observations_spaces, actions_spaces = env.get_state_action_spaces() #bornes inf et sup des states et actions

# # Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

###########################################################################################################################


params_agent = {'building_ids':building_ids,
                 'buildings_states_actions':os.path.join(data_path,'buildings_state_action_space.json'), 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}



# Instantiating the control agent(s)
agents = SAC(**params_agent)
state = env.reset()
done = False

action, coordination_vars = agents.select_action(state)    #basée sur ma policy, je vais faire une nouvelle action


for i in range (10):
    next_state, reward, done, _ = env.step(action) # environnement donne un nouvel état et reward associé à l'action réalisée
    action_next, coordination_vars_next = agents.select_action(next_state) 
#     agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next) #j'enregistre états/action/reward
    coordination_vars = coordination_vars_next
    state = next_state
    print(state)
    action = action_next

# # env.cost()
# # print(env.cost())