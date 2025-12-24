# TowerDefenseRL
Reinforcement learning agent for my [Tower Defense Game](https://github.com/Jacky8703/TowerDefenseGame).

## Features

- Integration with the Tower Defense Game (custom Gymnasium environment)
- Training script using Stable Baselines3 (maskable PPO algorithm)
- Tensorboard for monitoring training progress (with custom metrics)
- Model saving and loading
- Video recording of agent gameplay
- Script for replaying the best agent game actions from training (max waves reached)
- Random map selection during training for better generalization

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Jacky8703/TowerDefenseRL.git
    cd TowerDefenseRL
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Clone and set up the [Tower Defense Game](https://github.com/Jacky8703/TowerDefenseGame) server as described in its README.

## Usage

The tower defense game server must be running before executing any scripts.

### Train an agent
1. Set ```hours_to_train``` in ```train.py```

2. Execute the game server:
    ```bash
    (cd ../TowerDefenseGame && npm run start:api)
    ```

3. Execute script (default map):
    ```bash 
    python train.py
    ```
    Or with random maps:
    ```bash
    python train.py --random-maps custom-maps.json
    ```
4. Monitor training progress via TensorBoard (or at the end of training) in `http://localhost:6006/`:
    ```bash
    tensorboard --logdir ./logs/
    ```
5. The trained model will be saved in the `models/` directory.

In addition to the final model, a json file with the best agent performance (max waves reached) and a csv file with basic training metrics (reward, episode length and training time) will be saved.

In the `models/checkpoints/` directory, you will find periodic checkpoints of the model during training.

In the `models/videos/` directory, you will find videos of the agent's gameplay recorded at intervals during training.

In the `logs/` directory, a log file containing training metrics (visible via TensorBoard) will be created.

### Load a pre-trained model
1. If you want to continue the old training logs, add the `tb_log_name` argument to the `model.learn()` function in `train.py` with the corresponding tensorboard log name, e.g.:
    ```python
    model.learn(total_timesteps=training_steps, ..., tb_log_name="PPO_1")
    ```
2. Execute script specifying the model path:
    ```bash
    python train.py --load-model ./models/data_time/ppo_tower_defense.zip
    ```

### Replay best agent game (works only for the default map for now)
1. Execute the game server:
    ```bash
    # MUST EXECUTE
    (cd ../TowerDefenseGame && npm run start:api)
    ```

2. Execute the replay script specifying the json file with the best agent actions:
    ```bash
    python replay_actions.py --actions-file ./models/date_time/best_episode_actions.json
    ```
    Optionally, you can save the frames to a `best_frames` directory next to the actions file in the same `./models/date_time/`directory by adding the `--save-frames` argument (for future loading purposes):
    ```bash
    # MUST EXECUTE
    python replay_actions.py --actions-file ./models/date_time/best_episode_actions.json --save-frames
    ```

3. If you have already saved the frames, you can load them directly by using the `--load-frames` argument with the path to the `best_frames` directory (much faster):
    ```bash
    python replay_actions.py --load-dir ./models/date_time/best_frames
    ```
    On Ubuntu systems, you might need to first install `moviepy`:
    ```bash
    # Tsinghua mirror for faster installation
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    
    pip install moviepy --break-system-packages
    ```
    Then check `moviepy` version to ensure it's installed correctly:
    ```bash
    python -c "import moviepy; print(moviepy.__version__)"
    ```

    To create a video from the saved frames, use `make_video.py`:
    ```bash
    # MUST EXECUTE
    python make_video.py --load-dir ./models/date_time/best_frames
    ```
    