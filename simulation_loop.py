import os
import pandas as pd
from environment import HVACTrainingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3 import PPO

def simulation_loop():

    mode = 'model'

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
    action = env.action_space.sample()

    # --- Load the trained DQN model ---
    model = DQN.load("hvac_dqn.zip", env=env)
    #model = PPO.load("hvac_ppo.zip", env=env)

    while not (done or truncated):


        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        building_temperature.append(obs['indoor_temperature'][0])

        if mode=='baseline':
            if building_temperature[-1] > 21.67: action = 1
            else: action = 0

        if mode=='model':
            # --- Get action from trained RL model ---
            processed_obs = {}

            for k, v in obs.items():
                # Convert ints/floats/lists to np.array
                if not isinstance(v, np.ndarray):
                    v = np.array([v], dtype=np.float32)
                # Ensure shape is (1, n)
                v = v.reshape(1, -1)
                processed_obs[k] = v

            action, _ = model.predict(processed_obs, deterministic=True)
            action = action[0]

        env.render()

    env.close()
    env.final_render()
    print("Total Reward: ", total_reward)

simulation_loop()