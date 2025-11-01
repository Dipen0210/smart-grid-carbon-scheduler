from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    emissions: float
    done: bool


class CarbonAwareEnv:
    """Simple carbon-aware scheduling environment."""

    def __init__(
        self,
        carbon_intensity: np.ndarray,
        energy_kwh: float,
        max_power_kw: float,
    ) -> None:
        self.carbon_intensity = carbon_intensity
        self.energy_kwh = energy_kwh
        self.max_power_kw = max_power_kw
        self.horizon = len(carbon_intensity)
        self.reset()

    def reset(self) -> np.ndarray:
        self.hour = 0
        self.remaining_energy = self.energy_kwh
        self.total_emissions = 0.0
        return self._state()

    def step(self, action_kw: float) -> StepResult:
        action_kw = float(np.clip(action_kw, 0.0, self.max_power_kw))
        energy_used = min(action_kw, self.remaining_energy)
        carbon = self.carbon_intensity[self.hour]
        emissions = energy_used * carbon / 1000.0  # convert to kg
        self.remaining_energy -= energy_used
        self.total_emissions += emissions
        reward = -emissions

        self.hour += 1
        done = self.hour >= self.horizon or self.remaining_energy <= 1e-6
        return StepResult(
            state=self._state(),
            reward=reward,
            emissions=emissions,
            done=done,
        )

    def _state(self) -> np.ndarray:
        current_intensity = self.carbon_intensity[min(self.hour, self.horizon - 1)]
        return np.array(
            [
                current_intensity,
                self.remaining_energy,
                self.hour / self.horizon,
            ],
            dtype=float,
        )


__all__ = ("StepResult", "CarbonAwareEnv")
