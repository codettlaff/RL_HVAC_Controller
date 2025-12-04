import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import HVACTrainingEnv

# -----------------------------------------------------------
#  Environment factory (needed for vectorized environments)
# -----------------------------------------------------------
def make_env():
    data_folderpath = os.path.join(os.path.dirname(__file__), 'data')
    price_profile_df = pd.read_csv(os.path.join(data_folderpath, 'electricity_price.csv'))
    outdoor_temperature_df = pd.read_csv(os.path.join(data_folderpath, 'outdoor_temperature.csv'))
    non_hvac_load_df = pd.read_csv(os.path.join(data_folderpath, 'non_hvac_load.csv'))

    return HVACTrainingEnv(
        price_profile_df,
        outdoor_temperature_df,
        non_hvac_load_df,
        max_steps=287
    )


# -----------------------------------------------------------
#  Evaluation Function
# -----------------------------------------------------------
def evaluate_model(model):
    env = make_env()
    obs, info = env.reset()

    done = False
    truncated = False

    total_loss = 0.0
    comfort_lower = 20.0
    comfort_upper = 21.67
    T_mid = 0.5 * (comfort_lower + comfort_upper)

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        T = obs["indoor_temperature"][0]
        total_loss += abs(T - T_mid)

    return total_loss


# -----------------------------------------------------------
#  Hyperparameter Sweep
# -----------------------------------------------------------
def sweep_dqn():

    # Hyperparameter search grid
    learning_rates = [1e-4, 5e-4, 1e-3]
    gammas = [0.95, 0.98, 0.995]
    buffer_sizes = [50_000, 100_000]
    exploration_fracs = [0.05, 0.1, 0.2]

    best_loss = float("inf")
    best_params = None
    best_model = None

    trial = 0

    for lr in learning_rates:
        for gamma in gammas:
            for buf in buffer_sizes:
                for eps_frac in exploration_fracs:

                    trial += 1
                    print(f"\n===== Trial {trial} =====")
                    print(f"lr={lr}, gamma={gamma}, buffer={buf}, explore_frac={eps_frac}")

                    # Create training environment
                    env = DummyVecEnv([make_env])

                    # Build model
                    model = DQN(
                        "MultiInputPolicy",
                        env,
                        learning_rate=lr,
                        gamma=gamma,
                        buffer_size=buf,
                        exploration_fraction=eps_frac,
                        exploration_final_eps=0.02,
                        target_update_interval=500,
                        verbose=0
                    )

                    # Train model
                    model.learn(total_timesteps=50_000)

                    # Evaluate model on a full-day episode
                    loss = evaluate_model(model)

                    print(f"Total Loss = {loss}")

                    # Check if this is the best so far
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (lr, gamma, buf, eps_frac)
                        best_model = model

    # Save best model
    best_model.save("best_dqn_model")

    print("\n===== SWEEP COMPLETE =====")
    print("Best Loss:", best_loss)
    print("Best Parameters:")
    print("learning_rate =", best_params[0])
    print("gamma =", best_params[1])
    print("buffer_size =", best_params[2])
    print("exploration_fraction =", best_params[3])


if __name__ == "__main__":
    sweep_dqn()
