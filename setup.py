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

