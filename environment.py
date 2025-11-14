
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# State Vector:
# 1. Indoor Temperature (Float 10.0 - 40.0)
# 2. Outdoor Temperature (Float 20.0 - 50.0)
# 3. Electricity Price (Float 0.0 - 2.0)
# 4. Time Index (Int 0-287)
# 5. HVAC ON/ OFF (Int 0-1)


class HVACTrainingEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, price_profile_df, outdoor_temperature_df, render_mode: Optional[str] = None, **env_config):

        super().__init__()

        # ----- Shapes -----
        self.price_profile = price_profile_df["electricity_price"].to_numpy()
        self.outdoor_temperature_profile = outdoor_temperature_df["outdoor_temperature"].to_numpy()

        # ----- Environment Variables -----
        self.electricity_price = 0.0
        self.indoor_temperature = 0.0
        self.outdoor_temperature = 0.0
        self.time_of_day = 0

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
            "time_of_day": spaces.Discrete(288)   # 5-minute index: {0, ..., 287}
        })

        # Internal state
        self.current_step: int = 0
        self.max_steps: int = env_config.get("max_steps", 100)

    def _get_obs(self) -> np.ndarray:
        return {
        "electricity_price": self.electricity_price,
        "indoor_temperature": self.indoor_temperature,
        "outdoor_temperature": self.outdoor_temperature,
        "time_of_day": self.time_of_day }

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
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state and return (observation, info).

        Gymnasium API: reset must accept keyword-only seed, options.
        """
        super().reset(seed=seed)
        # If you want reproducibility based on seed, you can use self.np_random here

        self.current_step = 0

        # Initialize state (example: zeros)
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Optionally randomize initial state
        # self.state = self.np_random.normal(size=self.observation_space.shape).astype(np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_reward(self):
        return 1 # Placeholder

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

        # ----- Compute reward -----
        hvac_mode = action

        # ----- Update Environment Variables -----
        # 1. Update Outdoor Temperature from Shape
        self.electricity_price = self.price_profile[self.time_of_day]
        # 2. Update Electricity Price from Shape
        self.outdoor_temperature = self.outdoor_temperature_profile[self.time_of_day]
        # 3. Update datetime, time_of_day
        # 4. Update Indoor Temperature from Building Model

        # ----- Compute reward -----
        reward = self._get_reward()

        # ----- Check termination conditions -----
        # terminated: task completed or failed (absorbing)
        terminated = False  # e.g. abs(self.state[0]) > 10

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