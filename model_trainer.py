import os
import pandas as pd
from environment import HVACTrainingEnv
from stable_baselines3 import DQN

def train_MDP_controller(env):

    model = DQN(
        #"MlpPolicy",
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=500,
        verbose=1
    )
    model.learn(total_timesteps=287)
    model.save("hvac_dqn")

data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
env = HVACTrainingEnv(price_profile_df, outdoor_temperature_df, render_mode="human", max_steps=287)
train_MDP_controller(env)
model = DQN.load("hvac_dqn", env=env)
