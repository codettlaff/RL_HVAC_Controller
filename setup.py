import numpy as np
import pandas as pd
import os

def generate_price_profile():
    # Base price curve (hourly) – realistic dynamic pricing pattern
    hourly_prices = np.array([
        0.12, 0.11, 0.10, 0.10,  # 12am–4am low overnight
        0.12, 0.14,              # 4am–6am ramp
        0.18, 0.22, 0.26,        # morning rise
        0.24, 0.20, 0.18,        # late morning dip
        0.16, 0.14, 0.13,        # midday solar oversupply
        0.15, 0.20,              # afternoon ramp
        0.30, 0.45, 0.60,        # evening peak 5–8 pm
        0.40, 0.28,              # evening cool-down
        0.20, 0.15               # late-night decline
    ])

    # Interpolate to 5-minute resolution (288 points)
    price_5min = np.interp(
        np.linspace(0, 23, 288),
        np.arange(24),
        hourly_prices
    )

    # Optionally add a slight random component (very realistic)
    noise = np.random.normal(scale=0.01, size=288)  # ±1¢ variation
    price_5min = np.clip(price_5min + noise, 0.05, 2.0)

    csv_filepath = os.path.join(os.path.dirname(__file__), 'data', 'electricity_price.csv')
    df = pd.DataFrame({
        "time_of_day": range(len(price_5min)),
        "outdoor_temperature": price_5min
    })
    df.to_csv(csv_filepath, index=False)

def generate_outdoor_temperature_profile():
    # Base outdoor temperature shape (hourly)
    # Represents a typical mild-climate day (in °C)

    hourly_temps = np.array([
        24, 23, 22, 22,  # 12am–4am: warm overnight low (72–75°F)
        23, 25,  # 4am–6am: early morning warming
        28, 30, 32,  # 6am–9am: warming quickly
        34, 36, 38,  # 9am–12pm: mid-day strong heat
        39, 40, 41,  # 12pm–3pm: peak heat (41°C ≈ 106°F)
        40, 38,  # 3pm–5pm: slow cooling
        36, 34, 32,  # 5pm–8pm: evening cooling
        30, 28,  # 8pm–10pm: late evening
        26, 25  # 10pm–12am: nighttime but still hot
    ])

    # Interpolate to 5-minute resolution (288 points)
    temp_5min = np.interp(
        np.linspace(0, 23, 288),
        np.arange(24),
        hourly_temps
    )

    # Add slight noise (±0.3°C)
    noise = np.random.normal(scale=0.3, size=288)
    temp_5min = temp_5min + noise

    # Clip to realistic bounds (−20 to 50°C)
    temp_5min = np.clip(temp_5min, -20, 50)

    # Save to CSV in your /data directory
    csv_filepath = os.path.join(os.path.dirname(__file__), 'data', 'outdoor_temperature.csv')

    df = pd.DataFrame({
        "time_of_day": range(len(temp_5min)),     # 0–287
        "outdoor_temperature": temp_5min
    })

    df.to_csv(csv_filepath, index=False)

    return temp_5min

def generate_non_hvac_load_profile():
    """
    Generates a realistic non-HVAC load profile (kW) for a single building cluster.
    Shape resembles plug loads, lighting, appliances, and general usage.
    Saved as: data/non_hvac_load.csv
    """

    # Typical hourly non-HVAC load shape (kW)
    # Represents common lighting + appliances + plug load behavior
    hourly_load = np.array([
        0.18, 0.17, 0.16, 0.16,      # 12am–4am: overnight baseline
        0.18, 0.22,                  # 4am–6am: morning warm-up
        0.30, 0.38, 0.42,            # 6am–9am: morning usage peak
        0.35, 0.32, 0.28,            # 9am–12pm: late morning dip
        0.26, 0.25, 0.24,            # 12pm–3pm: midday low use
        0.28, 0.35,                  # 3pm–5pm: ramp
        0.55, 0.70, 0.85,            # 5pm–8pm: strong evening peak
        0.60, 0.40,                  # 8pm–10pm: evening cool-down
        0.30, 0.22                   # 10pm–12am: night wind-down
    ])

    # Interpolate to 5-minute resolution (288 points)
    load_5min = np.interp(
        np.linspace(0, 23, 288),
        np.arange(24),
        hourly_load
    )

    # Add noise (±5% of load value)
    noise = np.random.normal(scale=0.05, size=288)  # 5% variation
    load_5min = load_5min * (1 + noise)

    # Clip to realistic ranges for plug loads (0.1 kW to ~2.0 kW)
    load_5min = np.clip(load_5min, 0.1, 2.0)

    # Save to CSV
    csv_filepath = os.path.join(os.path.dirname(__file__), 'data', 'non_hvac_load.csv')

    df = pd.DataFrame({
        "time_of_day": range(len(load_5min)),   # 0–287
        "non_hvac_kw": load_5min
    })

    df.to_csv(csv_filepath, index=False)

    return load_5min

# generate_outdoor_temperature_profile()
# generate_non_hvac_load_profile()

