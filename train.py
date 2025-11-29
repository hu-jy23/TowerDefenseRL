import json
import gymnasium_env.envs  # ensure the custom environment is registered
import gymnasium as gym
import logging
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from gymnasium_env.wrappers.random_map_wrapper import RandomMapWrapper
from gymnasium_env.wrappers.wrap import wrap_env
from gymnasium_env.wrappers.flatten_multidiscrete import FlattenMultiDiscreteAction
from custom_callbacks.tensor_board_info import TensorboardInfoCallback
from custom_callbacks.save_agent_actions import SaveAgentActionsCallback
import argparse

# -------- 全局配置（可以改成命令行参数也行） --------
hours_to_train = 1
video_number = 10  # number of videos to record during training

mean_time_fps = 330  # ~mean time/fps from tensor board, steps per second (obviously varies)
mean_episode_steps = 2500  # ~mean steps per episode from tensor board (also varies)

training_steps = round(mean_time_fps * hours_to_train * 3600)  # total number of training steps
episode_recording_gap = (training_steps / mean_episode_steps) // video_number  # one episode = one game

env_name = "gymnasium_env/TowerDefenseWorld-v0"
seed = 87


# ========= 抽出来的模块化函数 =========

def make_env(random_maps_path: str | None,
             seed_value: int = seed,
             episode_gap: int = int(episode_recording_gap),
             run_prefix: str | None = None):
    """
    创建并包装 Tower Defense 环境。
    - 负责 gym.make + wrap_env + RandomMapWrapper
    - 与算法无关
    """
    if run_prefix is None:
        run_prefix = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")

    env = gym.make(env_name)
    #env = wrap_env(env, episode_gap, run_prefix)

    if random_maps_path:
        with open(random_maps_path, "r") as f:
            data = json.load(f)
        # set seed for reproducibility (same seed -> same map sequence)
        env.reset(seed=seed_value)
        env = RandomMapWrapper(env, map_list=data)

    return env


def make_model(algo: str,
               env,
               load_model_path: str | None,
               tensorboard_log: str = "./logs/"):
    """
    根据算法名称创建/加载模型。
    目前实现：
      - ppo: MaskablePPO（与原版保持一致）
    以后可以很方便地加：
      - dqn / trpo / sac 等
    """
    algo = algo.lower()

    if algo == "ppo":
        if load_model_path:
            logging.info(f"[Algo=ppo] Loading model from: {load_model_path}")
            model = MaskablePPO.load(load_model_path, env, tensorboard_log=tensorboard_log)
        else:
            logging.info("[Algo=ppo] Creating new MaskablePPO model (MlpPolicy)")
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
            )
        return model

    elif algo in ("dqn", "dqn_sb3"):
        # Flatten MultiDiscrete([A,T,X,Y]) -> Discrete(A*T*X*Y)
        try:
            env = FlattenMultiDiscreteAction(env)
        except Exception as e:
            logging.error(f"Failed to wrap env with FlattenMultiDiscreteAction: {e}")
            raise

        if load_model_path:
            logging.info(f"[Algo=dqn_sb3] Loading model from: {load_model_path}")
            model = DQN.load(load_model_path, env, tensorboard_log=tensorboard_log)
        else:
            logging.info("[Algo=dqn_sb3] Creating new DQN model (MlpPolicy)")
            policy_kwargs = dict(net_arch=[1024, 1024])
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100_000,
                batch_size=256,
                train_freq=4,
                target_update_interval=2_500,
                learning_starts=10_000,
                gamma=0.99,
                exploration_fraction=0.2,
                exploration_final_eps=0.05,
                verbose=1,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
            )
        return model

    # ==== 预留扩展位：DQN / TRPO 等（示例代码，可按需启用） ====
    # elif algo == "dqn":
    #     from stable_baselines3 import DQN
    #     if load_model_path:
    #         logging.info(f"[Algo=dqn] Loading model from: {load_model_path}")
    #         model = DQN.load(load_model_path, env, tensorboard_log=tensorboard_log)
    #     else:
    #         logging.info("[Algo=dqn] Creating new DQN model (MlpPolicy)")
    #         model = DQN(
    #             "MlpPolicy",
    #             env,
    #             verbose=1,
    #             tensorboard_log=tensorboard_log,
    #         )
    #     return model

    # elif algo == "trpo":
    #     # 需要 stable-baselines（不是 SB3），这里只示意接口
    #     from stable_baselines import TRPO
    #     from stable_baselines.common.policies import MlpPolicy
    #     if load_model_path:
    #         logging.info(f"[Algo=trpo] Loading model from: {load_model_path}")
    #         model = TRPO.load(load_model_path, env)
    #         model.tensorboard_log = tensorboard_log
    #     else:
    #         logging.info("[Algo=trpo] Creating new TRPO model (MlpPolicy)")
    #         model = TRPO(
    #             MlpPolicy,
    #             env,
    #             verbose=1,
    #             tensorboard_log=tensorboard_log,
    #         )
    #     return model

    else:
        raise ValueError(f"Unknown algorithm: {algo}. Supported: ppo")


# ========= 主逻辑 =========

def main(load_model_path: str | None,
         random_maps_path: str | None,
         algo: str):
    run_prefix = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")

    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ---- 环境创建（算法无关） ----
    env = make_env(
        random_maps_path=random_maps_path,
        seed_value=seed,
        episode_gap=int(episode_recording_gap),
        run_prefix=run_prefix,
    )

    # save 3 checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=training_steps // 3,
        save_path=f"./models/{run_prefix}/checkpoints/",
        name_prefix=f"{algo}_tower_defense",
    )
    # custom tensorboard callback to log wave number and tower counts
    tensorboard_info_callback = TensorboardInfoCallback()
    # custom callback to save best agent performance
    save_actions_callback = SaveAgentActionsCallback()

    try:
        logging.info(f"--- Starting New Training Run ---")
        logging.info(f"Algorithm: {algo}")
        logging.info(f"Environment: {env_name} (reset seed = {seed})")
        logging.info(
            f"Total Timesteps: {training_steps} (~{training_steps * 0.1 / 3600:.2f} hours of playing)"
        )
        logging.info(f"Video Recording Period: {episode_recording_gap} games")

        # ---- 模型创建/加载（算法相关） ----
        model = make_model(
            algo=algo,
            env=env,
            load_model_path=load_model_path,
            tensorboard_log="./logs/",
        )

        logging.info("Starting model training...")
        start = datetime.datetime.now()

        # 如果是加载模型，保持 timestep 连续
        reset_num_timesteps = not bool(load_model_path)

        model.learn(
            total_timesteps=training_steps,
            callback=[checkpoint_callback, tensorboard_info_callback, save_actions_callback],
            reset_num_timesteps=reset_num_timesteps,
        )

        logging.info(
            f"Model training completed in {(datetime.datetime.now() - start).total_seconds() / 3600:.2f} hours "
            f"({hours_to_train} planned)."
        )

        # ---- 保存最终模型 ----
        model.save(f"./models/{run_prefix}/{algo}_tower_defense.zip")
        logging.info("Model saved.")

        # ---- 保存最佳 episode 的动作序列 ----
        best_performance_data = save_actions_callback.get_best_agent_performance()
        with open(f"./models/{run_prefix}/best_episode_actions.json", "w") as f:
            json.dump(best_performance_data, f)
        logging.info("Best episode actions saved.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train an RL agent on the Tower Defense environment (modular, algo-pluggable)."
    )
    parser.add_argument("--load-model", help="Optional path to the model zip file.")
    parser.add_argument(
        "--random-maps",
        help="Optional path to the custom maps JSON file for random map training.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help=(
            "RL algorithm to use. Supported: "
            "ppo | dqn_sb3 (alias: dqn). For a future handmade DQN, use a separate entry."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        load_model_path=args.load_model,
        random_maps_path=args.random_maps,
        algo=args.algo,
    )
