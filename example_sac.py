
import os
import sys
sys.path.insert(0,'..')
from citylearn import  CityLearn
from pathlib import Path
import numpy as np                                   
import torch
import matplotlib.pyplot as plt
from agents.sac import SAC as Agent


# In[11]:

###############################################################################################################
# Load environment
climate_zone = 5
data_path = Path("data/Climate_Zone_"+str(climate_zone))
sim_period = (0, 1000)
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
        'central_agent': False,
        'save_memory': False }

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()
################################################################################################################


###########################################################################################################
#create agent

params_agent = {'building_ids':building_ids,
                 'buildings_states_actions':os.path.join(data_path,'buildings_state_action_space.json'), 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}

# Instantiating the control agent(s)
agents = Agent(**params_agent)


##########################################################################################################"
#training loop
for i in range (4):
    state = env.reset()
    done = False
    list_rewards=[]
    action, coordination_vars = agents.select_action(state)    
    while not done:
        next_state, reward, done, _ = env.step(action)
        list_rewards.append(reward)
        action_next, coordination_vars_next = agents.select_action(next_state)
    # print(next_state[0], reward, action_next)
        agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
        coordination_vars = coordination_vars_next
        state = next_state
        action = action_next
    print(env.cost())


#########################################################################
##plot
# interval = range(sim_period[0], sim_period[1])
# plt.figure(figsize=(16,5))
# # plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RBC for storage(kW)'])
# plt.show()

########################################################################


# # env.cost()
# print(env.cost())
# mean_100ep_reward = round(np.mean(list_rewards[-100:]), 1)
# print("Mean reward 100 derniers j:", mean_100ep_reward, "Num episodes:", len(list_rewards))
# plt.plot(np.arange(1000),list_rewards)
# plt.show()
# print(list_rewards[0:5])

# # # In[16]:


# sim_period = (0, 200)
# interval = range(sim_period[0], sim_period[1])
# plt.figure(figsize=(16,5))
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 
#             'Electricity demand with PV generation and without storage(kW)', 
#             'Electricity demand with PV generation and using SAC for storage control (kW)'])


# # In[17]:


# # Plotting summer operation in the last year
# interval = range(8760*3 + 24*30*6, 8760*3 + 24*30*6 + 24*10)
# plt.figure(figsize=(16,5))
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 
#             'Electricity demand with PV generation and without storage(kW)', 
# #             'Electricity demand with PV generation and using RBC for storage(kW)'])


# # In[18]:


# building_number = 'Building_1'
# interval = (range(24*30*6 + 8760*3,24*30*6 + 8760*3 + 24*4))
# plt.figure(figsize=(12,8))
# plt.plot(env.buildings[building_number].cooling_demand_building[interval])
# plt.plot(env.buildings[building_number].cooling_storage_to_building[interval] - env.buildings[building_number].hvac_device_to_cooling_storage[interval])
# plt.plot(env.buildings[building_number].hvac_device.cooling_supply[interval])
# plt.plot(env.electric_consumption_cooling[interval])
# plt.plot(env.buildings[building_number].hvac_device.cop_cooling[interval]*100,'--')
# plt.plot(env.buildings[building_number].cooling_storage.soc[interval],'--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Cooling Demand (kWh)',
#             'Energy Balance of Chilled Water Tank (kWh)', 
#             'Heat Pump Total Cooling Supply (kWh)', 
#             'Heat Pump Electricity Consumption (kWh)',
#             'Heat Pump COP x100',
#             'Cooling Storage State of Charge (kWh)'])


# # In[19]:


# building_number = 'Building_9'
# interval = range(8760*3 + 24*30*6, 8760*3 + 24*30*6 + 24*4)
# plt.figure(figsize=(12,8))
# plt.plot(env.buildings[building_number].cooling_storage_soc[interval])
# plt.plot(env.buildings[building_number].dhw_storage_soc[interval])
# plt.plot(env.buildings[building_number].electrical_storage_soc[interval])
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Cooling Storage Device SoC',
#             'Heating Storage Device SoC', 
#             'Electrical Storage Device SoC'])


# # In[ ]:




