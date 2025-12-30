"""
可以直接用这个入口（在 TowerDefenseRL 下运行）：
python evaluate.py --model-path ./models/PPO/MaskablePPO_wxy/MaskablePPO-CNN-OriginReward/29_12_2025_Skip120cooldown/ppo_tower_defense.zip --obs-mode cnn --map-file ./custom-maps.json --map-name map1 --episodes 10 --port 3000
python evaluate.py --model-path ./models/PPO/MaskablePPO_wxy/MaskablePPO-MLP-DamageReward/29_12_2025Modif_reward/ppo_tower_defense.zip --obs-mode mlp --map-file ./mlp-testmap.json --map-name default --episodes 10 --port 3000

D:/Study/RL/Proj/NewWorkspace/TowerDefenseRL/models/PPO/MaskablePPO_wxy/MaskablePPO-CNN-OriginReward/29_12_2025_Skip120cooldown/ppo_tower_defense.zip
D:/Study/RL/Proj/NewWorkspace/TowerDefenseRL/models/PPO/MaskablePPO_wxy/MaskablePPO-MLP-DamageReward/29_12_2025Modif_reward/ppo_tower_defense.zip
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import requests
import gymnasium_env.envs  # registers the custom env
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymnasium_env.wrappers.legacy_vector_observation import (
    LegacyVectorObservationWrapper,
)
from gymnasium_env.wrappers.wrap import SkipFrame

ENV_ID = "gymnasium_env/TowerDefenseWorld-v0"


def load_map_entries(map_file: str) -> list[dict]:
    with open(map_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "waypoints" in data:
        name = data.get("name", "map0")
        return [{"name": name, "waypoints": data["waypoints"]}]

    if isinstance(data, list):
        if not data:
            raise ValueError(f"Map file is empty: {map_file}")

        if isinstance(data[0], dict) and "waypoints" in data[0]:
            return data

        if isinstance(data[0], dict) and "x" in data[0] and "y" in data[0]:
            return [{"name": "map0", "waypoints": data}]

    raise ValueError(f"Unsupported map format in {map_file}")


def get_map_by_name(entries: list[dict], map_name: str) -> dict:
    for entry in entries:
        if entry.get("name") == map_name:
            return entry
    raise ValueError(f"Map name not found: {map_name}")


def get_map_by_index(entries: list[dict], map_index: int) -> dict:
    if map_index < 0 or map_index >= len(entries):
        raise ValueError(f"Map index out of range: {map_index} (0..{len(entries)-1})")
    return entries[map_index]


def get_maps_by_names(entries: list[dict], names: list[str]) -> list[dict]:
    by_name = {entry.get("name"): entry for entry in entries}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Map names not found in map-file: {', '.join(missing)}")
    return [by_name[name] for name in names]


def set_server_map(waypoints: list[dict], port: int) -> None:
    url = f"http://localhost:{port}/set-map"
    response = requests.post(url, json=waypoints)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to set map: {response.text}")

def debug_server_map(port: int) -> None:
    info_url = f"http://localhost:{port}/info"
    response = requests.get(info_url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to read /info: {response.text}")
    info = response.json()
    path_cells = info.get("map", {}).get("path_cells", [])
    print(
        "Server map:",
        f"{info.get('map', {}).get('width')}x{info.get('map', {}).get('height')}",
        f"cell={info.get('map', {}).get('cell_size')}",
        f"path_cells={len(path_cells)}",
        f"path_length={info.get('map', {}).get('path_length')}",
    )


def validate_obs_space(env: gym.Env, obs_mode: str) -> None:
    obs_space = env.observation_space

    if obs_mode == "cnn":
        if not isinstance(obs_space, spaces.Dict):
            raise ValueError("cnn mode requires a Dict observation_space.")
        map_space = obs_space.spaces.get("map_input")
        global_space = obs_space.spaces.get("global_input")
        if map_space is None or global_space is None:
            raise ValueError("cnn mode requires map_input/global_input keys.")
        if map_space.shape[0] != 13:
            raise ValueError(
                f"map_input channel mismatch: expected 13, got {map_space.shape[0]}"
            )
    else:
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
            raise ValueError("mlp mode requires a 1D Box observation_space.")


def make_base_env(
    port: int,
    obs_mode: str,
    skip_frames: int,
    debug: bool,
    debug_every: int,
) -> gym.Env:
    env = gym.make(ENV_ID, port=port)
    if hasattr(env.unwrapped, "debug"):
        env.unwrapped.debug = debug
    if hasattr(env.unwrapped, "debug_every"):
        env.unwrapped.debug_every = max(1, debug_every)

    if obs_mode == "mlp":
        env = LegacyVectorObservationWrapper(env)

    if skip_frames > 1:
        env = SkipFrame(env, skip=skip_frames)

    return env


def evaluate(model_path: str,
             port: int,
             obs_mode: str,
             map_waypoints: list[dict] | None,
             map_label: str | None,
             episodes: int,
             skip_frames: int,
             vecnorm_path: str | None,
             no_vecnorm: bool,
             debug: bool,
             debug_every: int) -> list[int]:
    if map_waypoints:
        set_server_map(map_waypoints, port)
        if map_label:
            print(f"Evaluating map: {map_label}")
        if debug:
            debug_server_map(port)

    env = make_base_env(
        port=port,
        obs_mode=obs_mode,
        skip_frames=skip_frames,
        debug=debug,
        debug_every=debug_every,
    )
    validate_obs_space(env, obs_mode)
    env = DummyVecEnv([lambda: env])

    if not no_vecnorm:
        if vecnorm_path:
            env = VecNormalize.load(vecnorm_path, env)
        else:
            print("Warning: vecnormalize.pkl not found; using raw observations.")
            env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
        env.training = False
        env.norm_reward = False

    model = MaskablePPO.load(model_path, env=env)

    episode_waves = []
    obs = env.reset()

    try:
        while len(episode_waves) < episodes:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=False, action_masks=action_masks)
            obs, rewards, dones, infos = env.step(action)

            if dones[0]:
                info = infos[0]
                wave = info.get("wave_number", 0)
                episode_waves.append(int(wave))
                prefix = f"[{map_label}] " if map_label else ""
                print(f"{prefix}Episode {len(episode_waves)}/{episodes}: wave {wave}")
    finally:
        env.close()

    return episode_waves


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MaskablePPO model on a fixed map."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a .zip model checkpoint (e.g., models/<run>/checkpoints/ppo_*.zip).",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["cnn", "mlp"],
        default="cnn",
        help="Observation mode: cnn uses the dict map_input/global_input; mlp uses legacy flat vector.",
    )
    parser.add_argument(
        "--map-file",
        default=None,
        help="Optional JSON map file (e.g., custom-maps.json).",
    )
    parser.add_argument(
        "--map-name",
        default=None,
        help="Optional map name to select from the JSON list. Use 'all' to eval map1-map10 (cnn only).",
    )
    parser.add_argument(
        "--map-index",
        type=int,
        default=0,
        help="Map index to select when map-name is not provided.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Game server port (default: 3000).",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=120,
        help="Skip-frame setting used during training (default: 120).",
    )
    parser.add_argument(
        "--vecnorm",
        default=None,
        help="Optional VecNormalize stats file (vecnormalize.pkl).",
    )
    parser.add_argument(
        "--no-vecnorm",
        action="store_true",
        help="Disable VecNormalize wrapping during evaluation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug checkpoints during evaluation.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=1,
        help="Print debug info every N env steps (default: 1).",
    )
    return parser.parse_args()


def find_vecnorm_path(model_path: str) -> str | None:
    model_dir = os.path.dirname(model_path)
    candidate = os.path.join(model_dir, "vecnormalize.pkl")
    if os.path.isfile(candidate):
        return candidate
    return None


def main() -> None:
    args = parse_arguments()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    vecnorm_path = args.vecnorm
    if not args.no_vecnorm and vecnorm_path is None:
        vecnorm_path = find_vecnorm_path(args.model_path)

    map_waypoints = None
    map_label = None
    entries = None

    if args.map_file:
        entries = load_map_entries(args.map_file)

    if args.map_name == "all":
        if args.obs_mode != "cnn":
            raise ValueError("map-name=all is only supported for cnn mode.")
        if entries is None:
            raise ValueError("map-file is required when map-name=all.")
        names = [f"map{i}" for i in range(1, 11)]
        selected = get_maps_by_names(entries, names)

        print("=" * 40)
        print("Evaluating maps: " + ", ".join(names))
        print("=" * 40)
        summary = {}
        for entry in selected:
            waves = evaluate(
                model_path=args.model_path,
                port=args.port,
                obs_mode=args.obs_mode,
                map_waypoints=entry["waypoints"],
                map_label=entry.get("name"),
                episodes=args.episodes,
                skip_frames=args.skip_frames,
                vecnorm_path=vecnorm_path,
                no_vecnorm=args.no_vecnorm,
                debug=args.debug,
                debug_every=args.debug_every,
            )
            mean_wave = float(np.mean(waves)) if waves else 0.0
            summary[entry.get("name", "unknown")] = {
                "mean_wave": mean_wave,
                "waves": waves,
            }
            print("-" * 40)
            print(f"{entry.get('name', 'map')}: mean {mean_wave:.2f}")
            print(f"{entry.get('name', 'map')}: waves {waves}")

        print("=" * 40)
        print("Summary:")
        for name in names:
            item = summary.get(name)
            if item:
                print(f"{name}: mean {item['mean_wave']:.2f}")
        return

    if entries is not None:
        if args.map_name:
            entry = get_map_by_name(entries, args.map_name)
        else:
            entry = get_map_by_index(entries, args.map_index)
        map_waypoints = entry["waypoints"]
        map_label = entry.get("name")

    waves = evaluate(
        model_path=args.model_path,
        port=args.port,
        obs_mode=args.obs_mode,
        map_waypoints=map_waypoints,
        map_label=map_label,
        episodes=args.episodes,
        skip_frames=args.skip_frames,
        vecnorm_path=vecnorm_path,
        no_vecnorm=args.no_vecnorm,
        debug=args.debug,
        debug_every=args.debug_every,
    )

    mean_wave = float(np.mean(waves)) if waves else 0.0
    print("=" * 40)
    print(f"Mean wave: {mean_wave:.2f}")
    print(f"All waves: {waves}")


if __name__ == "__main__":
    main()
