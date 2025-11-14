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
        12, 11, 10, 10,          # 12am–4am cooling
        11, 13,                  # 4am–6am early morning warmup
        16, 18, 20,              # 6am–9am warming
        22, 24, 26,              # 9am–12pm mid-day warming
        27, 28, 29,              # 12pm–3pm afternoon peak
        28, 26,                  # 3pm–5pm cooling starts
        24, 22, 20,              # 5pm–8pm evening cooling
        18, 16,                  # 8pm–10pm late evening
        14, 13                   # 10pm–12am night cooling
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

generate_outdoor_temperature_profile()

