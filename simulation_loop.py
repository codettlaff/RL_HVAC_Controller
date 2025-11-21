import os
import pandas as pd
from environment import HVACTrainingEnv

def simulation_loop():

    data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
    price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
    outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
    non_hvac_load_df = pd.read_csv(os.path.join(data_folderpath, 'non_hvac_load.csv'))
    env = HVACTrainingEnv(price_profile_df, outdoor_temperature_df, non_hvac_load_df, render_mode="human", max_steps=287)

    obs, info = env.reset()
    done = False
    truncated = False

    total_reward = 0
    building_temperature = []

    while not (done or truncated):
        action = env.action_space.sample()
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        building_temperature.append(obs['indoor_temperature'])
        env.render()

    env.close()
    print("Total Reward: ", total_reward)

simulation_loop()