import os
import pandas as pd
from environment import HVACTrainingEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import SAC

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

def train_MDP_controller_ppo(env):

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=5e-5,          # optimized LR
        n_steps=256,                 # optimized rollout size
        batch_size=256,              # optimized batch size
        gamma=0.95,                  # optimized discount factor
        ent_coef=0.001,              # unchanged (helps exploration)
        gae_lambda=0.95,             # unchanged (good for control tasks)
        clip_range=0.2,              # unchanged PPO clipping
        vf_coef=0.5,                 # unchanged value function weight
        max_grad_norm=0.5,           # unchanged for stability
        verbose=1
    )

    # Train PPO (longer training is beneficial)
    model.learn(total_timesteps=300_000)

    # Save the trained model
    model.save("hvac_ppo_optimized")

def train_MDP_controller_sac(env):
    """
    Train a Soft Actor-Critic (SAC) model.
    NOTE: SAC requires a continuous action space (gymnasium.spaces.Box).
    Your environment must be updated before this will run.
    """

    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,     # good default for SAC
        gamma=0.99,             # long-horizon recommended
        batch_size=256,         # SAC performs best with large batches
        tau=0.005,              # target smoothing
        train_freq=1,           # update every step
        gradient_steps=1,       # 1 gradient update per step
        ent_coef="auto",        # automatic entropy tuning
        verbose=1
    )

    model.learn(total_timesteps=300_000)
    model.save("hvac_sac")

    print("SAC training complete and model saved as hvac_sac.zip")


def train():

    data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
    price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
    outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
    non_hvac_load_df = pd.read_csv(os.path.join(data_folderpath, 'non_hvac_load.csv'))
    env = HVACTrainingEnv(price_profile_df, outdoor_temperature_df, non_hvac_load_df, render_mode="human",
                          max_steps=288)
    # train_MDP_controller(env)
    # train_MDP_controller_ppo(env)
    train_MDP_controller_sac(env)

train()



