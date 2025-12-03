import json
import os
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


def load_config():
    """
    加载配置优先级: A > B > C
    A. config.json (本地配置)
    B. config.default.json (仓库默认配置)
    C. 硬编码默认值 (下方 config 字典)
    """
    config = {
        "hours_to_train": 1,
        "video_number": 10,
        "mean_time_fps": 330,
        "mean_episode_steps": 2500,
        "env_name": "gymnasium_env/TowerDefenseWorld-v0",
        "seed": 87
    }
    
    loaded_file = "Hardcoded Defaults"
    
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config.update(json.load(f))
        loaded_file = "config.json"
    elif os.path.exists("config.default.json"):
        with open("config.default.json", "r") as f:
            config.update(json.load(f))
        loaded_file = "config.default.json"
        
    print(f"Loaded configuration from: {loaded_file}") # 在终端确认用了哪个配置
    return config

# 加载配置
CONFIG = load_config()

# 从配置中读取参数
hours_to_train = CONFIG["hours_to_train"]
video_number = CONFIG["video_number"]
mean_time_fps = CONFIG["mean_time_fps"]
mean_episode_steps = CONFIG["mean_episode_steps"]
env_name = CONFIG["env_name"]
seed = CONFIG["seed"]
training_steps = round(mean_time_fps * hours_to_train * 3600)  # total number of training steps
episode_recording_gap = (training_steps / mean_episode_steps) // video_number  # one episode = one game
if episode_recording_gap < 1:
    episode_recording_gap = 1


def make_env(random_maps_path: str | None,
             seed_value: int = seed,
             episode_gap: int = int(episode_recording_gap),
             run_prefix: str | None = None):
    """
    创建并返回环境。
    """
    if run_prefix is None:
        run_prefix = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")
  
    # 创建环境实例 TowerDefenseWorldEnv
    env = gym.make(env_name)

    # 如果传了 random_maps_path，RandomMapWrapper 会让 reset() 时随机换一张新图，然后再重置游戏。
    if random_maps_path:
        with open(random_maps_path, "r") as f:
            data = json.load(f)
        env.reset(seed=seed_value)  # 只需要在最开始执行一次，之后不需要传入 seed，会用最开始 seed 产生的一系列序列
        env = RandomMapWrapper(env, map_list=data)
        
    # env = wrap_env(env, episode_gap, run_prefix)  # 录像和监控 Wrapper，用来保存视频和每一局的监控数据

    return env


def make_model(algo: str,
               env,
               load_model_path: str | None,
               tensorboard_log: str = "./logs/"):
    """
    根据算法名称创建/加载模型。
    数据日志记录到 tensorboard_log 指定的目录（"./logs/"）中。
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


def main(load_model_path: str | None,
         random_maps_path: str | None,
         algo: str):
    """
    主训练函数。
    """
    run_prefix = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")

    logging.basicConfig(
        filename="training.log",                                # 操作日志写入到 training.log 文件
        level=logging.INFO,                                     # 记录 INFO 及以上级别的日志
        format="%(asctime)s - %(levelname)s - %(message)s",     # 日志格式包含时间、级别和消息
        filemode="a",                                           # 追加模式
    )

    # 创建环境
    env = make_env(
        random_maps_path=random_maps_path,
        seed_value=seed,
        episode_gap=int(episode_recording_gap),
        run_prefix=run_prefix,
    )

    # 三个回调函数
    # 定期存档: 在训练过程中自动保存模型的快照（.zip 文件）。
    checkpoint_callback = CheckpointCallback(
        save_freq=training_steps // 3,                    # 每隔总训练步数的三分之一保存一次
        save_path=f"./models/{run_prefix}/checkpoints/",  # 存档位置
        name_prefix=f"{algo}_tower_defense",              # 存档文件名前缀
    )
    # 自定义 TensorBoard 回调，从环境的 info 字典里提取额外信息并记录到 TensorBoard。
    tensorboard_info_callback = TensorboardInfoCallback()
    # 保存最佳对局: 如果发现当前这一局的波数比历史最高记录还高，它就会把这一局的所有动作序列存下来。
    # 训练结束后，可以用 replay_actions.py 直接重播“最高分”对局。
    save_actions_callback = SaveAgentActionsCallback()

    try:
        # 记录训练开始信息到日志 trainning.log 里
        logging.info(f"--- Starting New Training Run ---")
        logging.info(f"Algorithm: {algo}")
        logging.info(f"Environment: {env_name} (reset seed = {seed})")
        logging.info(
            f"Total Timesteps: {training_steps} (~{training_steps * 0.1 / 3600:.2f} hours of playing)"
        )
        logging.info(f"Video Recording Period: {episode_recording_gap} games")

        # 创建或加载模型
        model = make_model(
            algo=algo,
            env=env,
            load_model_path=load_model_path,
            tensorboard_log="./logs/",
        )

        logging.info("Starting model training...")
        start = datetime.datetime.now()

        # 开始训练
        model.learn(
            total_timesteps=training_steps,                                                    # 训练总步数
            callback=[checkpoint_callback, tensorboard_info_callback, save_actions_callback],  # 回调函数列表，在训练过程中会被定期调用
            reset_num_timesteps=not bool(load_model_path),                                     # 如果是加载旧模型继续训练，是否重置训练步数
        )

        logging.info(
            f"Model training completed in {(datetime.datetime.now() - start).total_seconds() / 3600:.2f} hours "
            f"({hours_to_train} planned)."
        )

        # 保存最终模型
        model.save(f"./models/{run_prefix}/{algo}_tower_defense.zip")
        logging.info(f"Model saved in: ./models/{run_prefix}/{algo}_tower_defense.zip")

        # 保存最佳 episode 的动作序列
        best_performance_data = save_actions_callback.get_best_agent_performance()
        with open(f"./models/{run_prefix}/best_episode_actions.json", "w") as f:
            json.dump(best_performance_data, f)
        logging.info(f"Best episode actions saved in: ./models/{run_prefix}/best_episode_actions.json")
        
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
        default=None,   # 默认不使用随机地图
        help="Optional path to the custom maps JSON file for random map training.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",  # 默认训练 PPO
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
