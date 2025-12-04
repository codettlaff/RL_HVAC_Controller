import os
import numpy as np
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import HVACTrainingEnv

# -----------------------------------------------------------
#  Environment Factory
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
#  Hyperparameter Sweep for SAC
# -----------------------------------------------------------
def sweep_sac():

    # Hyperparameter search grid for SAC
    learning_rates = [3e-4, 1e-4, 5e-5]
    gammas = [0.95, 0.98, 0.995]
    batch_sizes = [64, 128, 256]
    tau_values = [0.005, 0.01]          # Polyak smoothing
    ent_settings = ["auto", 0.001]      # Automatic entropy tuning vs fixed

    best_loss = float("inf")
    best_params = None
    best_model = None

    trial = 0

    for lr in learning_rates:
        for gamma in gammas:
            for batch in batch_sizes:
                for tau in tau_values:
                    for ent in ent_settings:

                        trial += 1
                        print(f"\n===== SAC Trial {trial} =====")
                        print(f"lr={lr}, gamma={gamma}, batch={batch}, tau={tau}, ent_coef={ent}")

                        # Create training environment
                        env = DummyVecEnv([make_env])

                        # Build SAC model
                        model = SAC(
                            "MultiInputPolicy",
                            env,
                            learning_rate=lr,
                            gamma=gamma,
                            batch_size=batch,
                            tau=tau,
                            train_freq=1,           # update every step
                            gradient_steps=1,       # 1 gradient step per env step
                            ent_coef=ent,           # "auto" or fixed
                            verbose=0
                        )

                        # Train SAC for 50k timesteps
                        model.learn(total_timesteps=500)

                        # Evaluate on full-day episode
                        loss = evaluate_model(model)
                        print(f"Total Loss = {loss}")

                        # Track best model
                        if loss < best_loss:
                            best_loss = loss
                            best_params = (lr, gamma, batch, tau, ent)
                            best_model = model

    # Save best model
    best_model.save("best_sac_model")

    print("\n===== SAC SWEEP COMPLETE =====")
    print("Best Loss:", best_loss)
    print("Best Parameters:")
    print("learning_rate =", best_params[0])
    print("gamma =", best_params[1])
    print("batch_size =", best_params[2])
    print("tau =", best_params[3])
    print("ent_coef =", best_params[4])


if __name__ == "__main__":
    sweep_sac()
