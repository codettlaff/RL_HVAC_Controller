import os
import pandas as pd
from environment import HVACTrainingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC

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
    path = os.path.join(os.path.dirname(__file__), 'price_aware_results', 'hvac_dqn_optimized.zip')
    model = DQN.load(path, env=env)
    # model = PPO.load(path, env=env)
    # model = SAC.load(path, env=env)

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

def run_baseline_and_plot(save_path="baseline_results.png"):
    """
    Runs the baseline (bang–bang thermostat) controller:
        - HVAC ON when indoor_temp > 21.67°C
        - HVAC OFF otherwise
    Generates and saves a plot of Indoor vs Outdoor temperatures
    and HVAC load, similar to env.final_render().
    """

    # --- Load environment data ---
    data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
    price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
    outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
    non_hvac_load_df = pd.read_csv(os.path.join(data_folderpath, 'non_hvac_load.csv'))

    env = HVACTrainingEnv(price_profile_df, outdoor_temperature_df, non_hvac_load_df, max_steps=287)

    obs, info = env.reset()
    done = False
    truncated = False

    indoor_temps = []
    outdoor_temps = []
    hvac_profile = []

    # --- Baseline control loop ---
    while not (done or truncated):
        T = obs["indoor_temperature"][0]

        # Bang-bang rule
        if T > 21.67:
            action = 1
        else:
            action = 0

        obs, reward, done, truncated, info = env.step(action)

        indoor_temps.append(obs["indoor_temperature"][0])
        outdoor_temps.append(obs["outdoor_temperature"][0])
        hvac_profile.append(action * 2.5)  # consistent with env.hvac_load definition

    # ============================
    #        GENERATE PLOTS
    # ============================
    import matplotlib.pyplot as plt
    indoor = np.array(indoor_temps)
    outdoor = np.array(outdoor_temps)
    hvac = np.array(hvac_profile)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # --- Indoor + Outdoor Temperature ---
    axes[0].plot(indoor, label="Indoor Temperature (°C)", color="tab:blue")
    axes[0].plot(outdoor, label="Outdoor Temperature (°C)", color="tab:orange", alpha=0.8)

    axes[0].axhline(20.0, color='gray', linestyle='--', linewidth=1)
    axes[0].axhline(21.67, color='gray', linestyle='--', linewidth=1)

    axes[0].set_title("Baseline: Indoor vs Outdoor Temperature")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True)

    # --- HVAC Load ---
    axes[1].plot(hvac, color="tab:red", label="HVAC Load (kW)")
    axes[1].set_title("Baseline: HVAC Load Profile")
    axes[1].set_ylabel("kW")
    axes[1].set_xlabel("Timestep (5-minute index)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Baseline plot saved to: {save_path}")

    plt.show()

# run_baseline_and_plot()

simulation_loop()