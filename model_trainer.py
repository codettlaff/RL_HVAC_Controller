import os
import pandas as pd
from environment import HVACTrainingEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO

def train_MDP_controller(env):

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=0.001,          # optimized
        gamma=0.98,                   # optimized
        buffer_size=100000,           # optimized
        exploration_fraction=0.05,    # optimized
        exploration_final_eps=0.02,   # unchanged
        target_update_interval=500,   # unchanged
        verbose=1
    )

    model.learn(total_timesteps=500000)
    model.save("hvac_dqn_optimized")

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

#train()



