import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_input_profiles():
    """
    Loads the generated CSV profiles and produces:
      1. Outdoor temperature time series
      2. Electricity price profile
      3. Non-HVAC load profile
    Each plot is saved to the /data directory.
    """

    # --- Filepaths ---
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    temp_path = os.path.join(base_dir, "outdoor_temperature.csv")
    price_path = os.path.join(base_dir, "electricity_price.csv")
    load_path = os.path.join(base_dir, "non_hvac_load.csv")

    # --- Load CSVs ---
    temp_df = pd.read_csv(temp_path)
    price_df = pd.read_csv(price_path)
    load_df = pd.read_csv(load_path)

    # ============================
    # 1. Outdoor Temperature Plot
    # ============================
    plt.figure(figsize=(12, 4))
    plt.plot(temp_df["time_of_day"], temp_df["outdoor_temperature"], label="Outdoor Temperature (°C)")
    plt.title("Outdoor Temperature Profile (24 hours)")
    plt.xlabel("Time of Day (5-min Index)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "plot_outdoor_temperature.png"), dpi=300)
    plt.show()

    # ============================
    # 2. Electricity Price Plot
    # ============================
    plt.figure(figsize=(12, 4))
    # NOTE: your electricity_price.csv incorrectly names the column "outdoor_temperature"
    if "outdoor_temperature" in price_df.columns:
        y = price_df["outdoor_temperature"]
    else:
        y = price_df["electricity_price"]
    plt.plot(price_df["time_of_day"], y, label="Price ($/kWh)", color="tab:red")
    plt.title("Electricity Price Profile (24 hours)")
    plt.xlabel("Time of Day (5-min Index)")
    plt.ylabel("Price ($/kWh)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "plot_electricity_price.png"), dpi=300)
    plt.show()

    # ============================
    # 3. Non-HVAC Load Plot
    # ============================
    plt.figure(figsize=(12, 4))
    plt.plot(load_df["time_of_day"], load_df["non_hvac_kw"], label="Non-HVAC Load (kW)", color="tab:green")
    plt.title("Non-HVAC Load Profile (24 hours)")
    plt.xlabel("Time of Day (5-min Index)")
    plt.ylabel("Load (kW)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "plot_non_hvac_load.png"), dpi=300)
    plt.show()

    print("Plots saved to:", base_dir)

plot_input_profiles()