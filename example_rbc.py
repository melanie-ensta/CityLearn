
import os
import sys
sys.path.insert(0,'..')
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from agents.rbc import RBC



#In[2]:


# Select the climate zone and load environment
climate_zone = 5
data_path = Path("data/Climate_Zone_"+str(climate_zone))
sim_period = (0, 8760)
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

env = CityLearn(**params)

observations_spaces, actions_spaces = env.get_state_action_spaces()


# RULE-BASED CONTROLLER (RBC) (Stores energy at night and releases it during the day)
# In this example, each building has its own RBC, which tries to flatten a generic building load 
# by storing energy at night and using it during the day, which isn't necessarily the best solution 
# in order to flatten the total load of the district.
# Select the climate zone and load environment

'''IMPORTANT: Make sure that the buildings_state_action_space.json file contains the hour of day as 3rd true state:
{"Building_1": {
    "states": {
        "month": true,
        "day": true,
        "hour": true
Alternative, modify the line: "hour_day = states[0][2]" of the RBC_Agent Class in agent.py
'''
import json
import time
# Instantiating the control agent(s)
agents = RBC(actions_spaces)


# Finding which state 
with open(os.path.join(data_path,'buildings_state_action_space.json')) as file:
    actions_ = json.load(file) #fichier json

indx_hour = -1
for obs_name, selected in list(actions_.values())[0]['states'].items():
    indx_hour += 1
    if obs_name=='hour':
        break
    assert indx_hour < len(list(actions_.values())[0]['states'].items()) - 1, "Please, select hour as a state for Building_1 to run the RBC"



state = env.reset()


done = False
rewards_list = []
start = time.time()
while not done:
    hour_state = np.array([[state[0][indx_hour]]]) #bc the rule is based on the hour
    action = agents.select_action(hour_state)
    next_state, rewards, done, _ = env.step(action)
    # print(next_state[0], rewards, action)
    state = next_state
    rewards_list.append(rewards)
    
    

# print(rewards_list)
#########################################################################
# plot
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

# plt.plot(np.arange(8760), rewards_list)
# plt.show()
mean_100ep_reward = round(np.mean(rewards_list[-100:]), 1)
print("Mean reward 100 derniers j:", mean_100ep_reward, "Num episodes:", len(rewards_list))
print(env.cost())


# # In[29]:


# 1.18602800e-01 + 0.034*2


# # In[52]:


# action[0][0]=0.034
# action[0][1]=0.034
# action[0][2]=0.034


# # In[61]:


# next_state, rewards, done, _ = env.step(action)


# # In[58]:


# action[0]


# # In[59]:


# state[0]


# # In[62]:


# next_state[0]


# # In[7]:


# cost_rbc


# # In[9]:


# # Plotting electricity consumption breakdown
# interval = range(sim_period[0], sim_period[1])
# plt.figure(figsize=(16,5))
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RBC for storage(kW)'])


# # In[10]:


# # Plotting 5 days of winter operation of year 1
# plt.figure(figsize=(16,5))
# interval = range(0,24*5)
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RBC for storage(kW)'])


# # In[10]:


# # Plotting summer operation of year 1
# plt.figure(figsize=(16,5))
# interval = range(24*30*7,24*30*7 + 24)
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RBC for storage(kW)'])


# # In[11]:


# # Plotting summer operation
# interval = range(5000,5000 + 24*10)
# plt.figure(figsize=(16,5))
# plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
# plt.plot(env.net_electric_consumption_no_storage[interval])
# plt.plot(env.net_electric_consumption[interval], '--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RBC for storage(kW)'])


# # In[11]:


# # Plot for one building of the total cooling supply, the state of charge, and the actions of the controller during winter
# building_number = 'Building_5'
# plt.figure(figsize=(12,8))
# plt.plot(env.buildings[building_number].cooling_demand_building[3500:3500+24*5])
# plt.plot(env.buildings[building_number].cooling_storage_soc[3500:3500+24*5])
# plt.plot(env.buildings[building_number].hvac_device_to_building[3500:3500+24*5] + env.buildings[building_number].hvac_device_to_cooling_storage[3500:3500+24*5])
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Building Cooling Demand (kWh)','Energy Storage State of Charge - SOC (kWh)', 'Heat Pump Total Cooling Supply (kW)'])


# # In[12]:


# building_number = 'Building_1'
# interval = range(0,24*4)
# plt.figure(figsize=(12,8))
# plt.plot(env.buildings[building_number].cooling_demand_building[interval])
# plt.plot(env.buildings[building_number].cooling_storage_to_building[interval] - env.buildings[building_number].hvac_device_to_cooling_storage[interval])
# plt.plot(env.buildings[building_number].hvac_device.cooling_supply[interval])
# plt.plot(env.electric_consumption_cooling[interval])
# plt.plot(env.buildings[building_number].hvac_device.cop_cooling[interval],'--')
# plt.xlabel('time (hours)')
# plt.ylabel('kW')
# plt.legend(['Cooling Demand (kWh)','Energy Balance of Chilled Water Tank (kWh)', 'Heat Pump Total Cooling Supply (kWh)', 'Heat Pump Electricity Consumption (kWh)','Heat Pump COP'])


# # # In[ ]:
