
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from building_model import rc_building_model

# State Vector:
# 1. Indoor Temperature (Float 10.0 - 40.0)
# 2. Outdoor Temperature (Float 20.0 - 50.0)
# 3. Electricity Price (Float 0.0 - 2.0)
# 4. Time Index (Int 0-287)
# 5. HVAC ON/ OFF (Int 0-1)


class HVACTrainingEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, price_profile_df, outdoor_temperature_df, non_hvac_load_df, render_mode: Optional[str] = None, **env_config):

        super().__init__()

        # ----- Shapes -----
        self.price_profile = price_profile_df["electricity_price"].to_numpy()
        self.outdoor_temperature_profile = outdoor_temperature_df["outdoor_temperature"].to_numpy()
        self.non_hvac_load_profile = non_hvac_load_df["non_hvac_kw"].to_numpy()

        # ----- Environment Variables -----
        self.state = 0
        self.electricity_price = 0.0
        self.indoor_temperature = 20.56
        self.outdoor_temperature = 0.0
        self.time_of_day = 0
        self.non_hvac_load = 0.0
        self.hvac_load = 0.0

        # ----- Save config and render mode -----
        self.render_mode = render_mode
        self.env_config = env_config  # e.g. {"max_steps": 100, ...}

        # ----- Define action space -----
        # Size-1 discrete action space: actions = {0, 1}
        self.action_space = spaces.Discrete(2)

        # ----- Define observation space -----
        # Dictionary observation space for multiple inputs
        self.observation_space = spaces.Dict({
            "electricity_price": spaces.Box(
                low=0.0,
                high=2.0,  # $0.00–$2.00 per kWh (covers most TOU, RTP, and peaks)
                shape=(1,),  # usually a single scalar
                dtype=np.float32
            ),
            "indoor_temperature": spaces.Box(
                low=10.0,
                high=40.0,  # 10°C–40°C (50°F–104°F) realistic for buildings
                shape=(1,),
                dtype=np.float32
            ),
            "outdoor_temperature": spaces.Box(
                low=-20.0,
                high=50.0,  # -20°C–50°C (extreme U.S. range)
                shape=(1,),
                dtype=np.float32
            ),
            "time_of_day": spaces.Box(
                low=0.0,
                high=287.0,
                shape=(1,),
                dtype=np.float32
            )   # 5-minute index: {0, ..., 287}
        })

        # Internal state
        self.current_step: int = 0
        self.max_steps: int = env_config.get("max_steps", 100)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "electricity_price": np.array([self.electricity_price], dtype=np.float32),
            "indoor_temperature": np.array([self.indoor_temperature], dtype=np.float32),
            "outdoor_temperature": np.array([self.outdoor_temperature], dtype=np.float32),
            "time_of_day": np.array([self.time_of_day], dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Obs is for Agent, Info is for Human
        """
        return {
            "step": self.current_step,
        }

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self.current_step = 0

        # Reset building internal state variables
        self.time_of_day = 0
        self.indoor_temperature = 20.56
        self.outdoor_temperature = float(self.outdoor_temperature_profile[0])
        self.electricity_price = float(self.price_profile[0])
        self.non_hvac_load = float(self.non_hvac_load_profile[0])

        # Build obs explicitly
        obs = self._get_obs()

        info = {}

        return obs, info

    def _get_reward(self, last_timestep_hvac_load):
        """
        Computes reward based on:
          - Staying in comfort band (20°C to 21.67°C)
          - Avoiding HVAC on/off switching
          - Minimizing energy use during high price periods
        """

        # --- Comfort penalty ---
        # Comfort band: 20°C – 21.67°C
        T = self.indoor_temperature
        comfort_lower = 20.0
        comfort_upper = 21.67

        if T < comfort_lower:
            comfort_penalty = (comfort_lower - T) * 2.0
        elif T > comfort_upper:
            comfort_penalty = (T - comfort_upper) * 2.0
        else:
            comfort_penalty = 0.0

        # --- Switching penalty ---
        # Penalize toggling HVAC on/off
        switching_penalty = 1.0 if (last_timestep_hvac_load != self.hvac_load) else 0.0

        # --- Price penalty ---
        # Higher price → larger penalty for HVAC power
        # Price in [0.1, 0.2]
        price = self.electricity_price
        price_penalty = (price - 0.1) * 10

        # --- Scaling ---
        comfort_penalty = comfort_penalty * 10
        switching_penalty = switching_penalty * 1
        price_penalty = price_penalty * 1

        # --- Total reward ---
        # Negative because each is a penalty
        total_penalty = comfort_penalty + switching_penalty + price_penalty

        return -total_penalty

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        Returns: observation, reward, terminated, truncated, info
        (Gymnasium 0.28+ API)
        """
        # ----- Validate action -----
        assert self.action_space.contains(action), f"{action} is an invalid action"

        # ----- Apply action & update state -----
        # Example: simple state update (replace with your dynamics)
        self.current_step += 1
        last_timestep_hvac_load = self.hvac_load
        self.hvac_load = action * 2.5
        self.state = action

        # ----- Update Environment Variables -----
        self.time_of_day += 1
        # 1. Update Outdoor Temperature from Shape
        self.electricity_price = self.price_profile[self.time_of_day-1]
        # 2. Update Electricity Price from Shape
        self.outdoor_temperature = self.outdoor_temperature_profile[self.time_of_day-1]
        # 3. Update Non Hvac Load From Shape
        self.non_hvac_load = self.non_hvac_load_profile[self.time_of_day-1]
        # 4. Update Indoor Temperature from Building Model
        self.indoor_temperature, building_kw_demand, building_kvar_demand = rc_building_model(5, self.non_hvac_load, self.hvac_load, self.indoor_temperature, self.outdoor_temperature)
        self.indoor_temperature = self.indoor_temperature[0]

        # ----- Compute reward -----
        reward = self._get_reward(last_timestep_hvac_load)

        # ----- Check termination conditions -----
        # terminated: task completed or failed (absorbing)
        # terminated = False  # e.g. abs(self.state[0]) > 10
        terminated = self.current_step >= self.max_steps

        # truncated: time limit or external truncation
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        """
        Clean up any resources (e.g., windows, files) here.
        """
        pass