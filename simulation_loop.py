import os
import pandas as pd
from environment import HVACTrainingEnv

def simulation_loop():

    data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
    price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
    outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
    env = HVACTrainingEnv(price_profile_df, outdoor_temperature_df, render_mode="human", max_steps=287)

    obs, info = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()

simulation_loop()