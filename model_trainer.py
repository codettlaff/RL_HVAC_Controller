import os
import pandas as pd
from environment import HVACTrainingEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO

def train_MDP_controller(env):

    model = DQN(
        #"MlpPolicy",
        "MultiInputPolicy",
        env,
        learning_rate=1e-4, #1e-3
        gamma = 0.9592903523242123,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=500,
        verbose=1
    )
    model.learn(total_timesteps=50000)
    model.save("hvac_dqn")

# Very bad results
def train_MDP_controller_ppo(env):

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,          # stable for PPO
        n_steps=2048,                # large batch for slow dynamics
        batch_size=64,
        gamma=0.995,                 # longer horizon → better anticipation
        ent_coef=0.001,              # slight exploration encouragement
        gae_lambda=0.95,             # good default for control tasks
        clip_range=0.2,              # PPO clipping
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )

    # Train for more timesteps than DQN (PPO handles long runs well)
    model.learn(total_timesteps=300_000)

    model.save("hvac_ppo")

def train():

    data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
    price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
    outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
    non_hvac_load_df = pd.read_csv(os.path.join(data_folderpath, 'non_hvac_load.csv'))
    env = HVACTrainingEnv(price_profile_df, outdoor_temperature_df, non_hvac_load_df, render_mode="human",
                          max_steps=288)
    train_MDP_controller(env)
    #train_MDP_controller_ppo(env)

# train()



