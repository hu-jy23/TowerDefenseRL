import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LegacyVectorObservationWrapper(gym.ObservationWrapper):
    """
    Legacy flat observation for MLP-based models.
    This mirrors the older vector encoding used before the grid-based CNN input.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._init_static_specs()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.total_features_count,),
            dtype=np.float32,
        )

    def _init_static_specs(self) -> None:
        game_info = self.env.unwrapped.game_info
        map_info = game_info["map"]

        self.map_width = map_info["width"]
        self.map_height = map_info["height"]
        self.cell_size = map_info["cell_size"]
        self.path_cells_coordinates_normalized = self._normalize_path_cells(game_info)

        self.max_towers = (
            (self.map_width // self.cell_size)
            * (self.map_height // self.cell_size)
            - (map_info["path_length"] // self.cell_size)
        )
        self.max_enemies = self._calculate_total_enemies(game_info)

        self.tower_types = game_info["towers"]
        self.enemy_types = game_info["waves"]["enemy_types"]
        self.tower_type_to_index = {
            tower["type"]: idx for idx, tower in enumerate(self.tower_types)
        }
        self.enemy_type_to_index = {
            enemy_type: idx for idx, enemy_type in enumerate(self.enemy_types)
        }

        self.max_tower_dps = max(tower["dps"] for tower in self.tower_types)
        self.max_attack_cooldown = game_info["slower_tower_sample"]["attackCooldown"]

        self.max_time = game_info["max_global_info"]["gameTime"]
        self.max_wave = game_info["max_global_info"]["waveNumber"]
        self.max_money = game_info["max_global_info"]["money"]
        self.max_lives = game_info["max_global_info"]["lives"]

        self.global_feature_count = 5 + len(self.path_cells_coordinates_normalized)
        self.features_per_tower = 5 + len(self.tower_types)
        self.tower_feature_count = self.max_towers * self.features_per_tower
        self.features_per_enemy = 5 + len(self.enemy_types)
        self.enemy_feature_count = self.max_enemies * self.features_per_enemy
        self.total_features_count = (
            self.global_feature_count + self.tower_feature_count + self.enemy_feature_count
        )

    def observation(self, obs):
        game_state = self.env.unwrapped.game_state
        observation = np.zeros((self.total_features_count,), dtype=np.float32)

        observation[0] = game_state["gameTime"] / self.max_time
        observation[1] = game_state["waveNumber"] / self.max_wave
        observation[2] = game_state["money"] / self.max_money
        observation[3] = game_state["lives"] / self.max_lives
        observation[4] = 1.0 if game_state["gameOver"] else 0.0
        observation[
            5 : 5 + len(self.path_cells_coordinates_normalized)
        ] = self.path_cells_coordinates_normalized

        for idx, tower in enumerate(game_state["towers"]):
            offset = self.global_feature_count + idx * self.features_per_tower
            observation[offset] = 1.0
            observation[offset + 1] = tower["position"]["x"] / self.map_width
            observation[offset + 2] = tower["position"]["y"] / self.map_height
            observation[offset + 3] = tower["attackCooldown"] / self.max_attack_cooldown
            observation[offset + 4] = (
                self.tower_types[self.tower_type_to_index[tower["type"]]]["dps"]
                / self.max_tower_dps
            )
            observation[offset + 5 + self.tower_type_to_index[tower["type"]]] = 1.0

        for idx, enemy in enumerate(game_state["enemies"]):
            offset = (
                self.global_feature_count
                + self.tower_feature_count
                + idx * self.features_per_enemy
            )
            observation[offset] = 1.0
            observation[offset + 1] = enemy["position"]["x"] / self.map_width
            observation[offset + 2] = enemy["position"]["y"] / self.map_height
            observation[offset + 3] = enemy["currentHealth"] / enemy["fullHealth"]
            observation[offset + 4] = enemy["pathProgress"]
            observation[offset + 5 + self.enemy_type_to_index[enemy["type"]]] = 1.0

        return observation

    def _normalize_path_cells(self, game_info) -> list[float]:
        normalized_coordinates = []
        for cell in game_info["map"]["path_cells"]:
            normalized_coordinates.append(cell["x"] / self.map_width)
            normalized_coordinates.append(cell["y"] / self.map_height)
        return normalized_coordinates

    def _calculate_total_enemies(self, game_info) -> int:
        wave_delay = game_info["waves"]["wave_delay"]
        wave_max_enemies = game_info["waves"]["max_enemies"]
        spawn_delay = game_info["waves"]["spawn_delay"]
        slower_enemy_time = (
            game_info["map"]["path_length"]
            / game_info["waves"]["slower_enemy_sample"]["currentSpeed"]
        )
        total_enemies = int(
            slower_enemy_time * wave_max_enemies
            / (wave_delay + spawn_delay * wave_max_enemies)
        )
        if slower_enemy_time < wave_delay:
            total_enemies = wave_max_enemies
        return total_enemies
