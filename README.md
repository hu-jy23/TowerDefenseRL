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
### 注意
- 先检测CUDA版本
- 完美，我来直接给你 **最安全、最兼容、不会再有 NumPy / cv2 / SB3 / Gym 等冲突** 的安装方案。

你的 `requirements.txt` **无法直接用**，因为它里面包含大量 **不兼容 NumPy 2.x 或不兼容 Torch 2.8** 的库。
如果你按它安装，你的新环境马上就会像之前一样炸掉。

---

### ✅ **结论：Python 3.10 是最推荐的！**

✔ 完全支持 Stable-Baselines3
✔ 完全支持 Gymnasium
✔ 完全支持 PyTorch
✔ 各种科学计算库也兼容最佳

→ 所以你现在的 `(rl)` 环境选 Python 3.10 **非常正确**。

---

### ❗ 你必须重新整理 requirements（尤其是 numpy / fsspec / protobuf）

 requirements 包含：

```
numpy==2.3.3
fsspec==2025.9.0
protobuf==6.32.1
torch==2.8.0
```

这些版本会让绝大部分 RL 库直接崩溃。

---

### 🚫 **这些版本不能用（会崩）**

| 包名                  | 你的版本                            | 问题                       |
| ------------------- | ------------------------------- | ------------------------ |
| **numpy 2.3.3**     | ❌ 不兼容 opencv、numba、gym 的部分 C 扩展 | 你的之前报错就是因为这个             |
| **fsspec 2025.9.0** | ❌ 未来版本，datasets 不能用             | 没有理由装这么新的                |
| **protobuf 6.32.1** | ❌ wandb、tensorboard 不兼容         |                          |
| **torch 2.8.0**     | ❌ SB3 未验证，可能会导致 CUDA / API 不兼容  | SB3 官方验证只到 torch==2.3 左右 |

---

### 🟩 正确、安全的 RL 环境需求如下：

### **核心依赖（SB3 官方推荐）**

| 包                 | 推荐版本                    |
| ----------------- | ----------------------- |
| numpy             | **1.24.x** 或 1.26.x     |
| torch             | **2.3.x**（正式支持） 或 2.2.x |
| gymnasium         | 1.0+                    |
| stable-baselines3 | 2.3.2 或 2.7.0           |
| opencv-python     | 最新（兼容 numpy 1.x）        |

---

### 🧩 **所以你应该这样安装，不要用原本的 requirements.txt**

我给你一套最兼容的 RL 环境安装命令（适用于 CPU 或 GPU）。

---

### ✅ **第一步：激活环境**

```bash
conda activate rl
```

---

### ✅ **第二步：安装 PyTorch （推荐版本，不是 2.8）**

### ⭐ 如果你用 CPU：

```bash
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ⭐ 如果你用 NVIDIA GPU（CUDA 12.1）：

```bash
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Torch 2.8 对 SB3 来说太新，不稳定，也不兼容你项目里可能用到的其他库。

---

### ✅ **第三步：安装 stable-baselines3 与 extra 依赖**

✔ 这个命令 *自动匹配* 所有依赖最佳版本
✔ 避免 numpy 2.x
✔ 避免崩溃

```bash
pip install "stable-baselines3[extra]"
```

---

### ⭐ **此时会自动安装：**

* numpy **1.26.x**
* gymnasium 1.2
* pandas
* matplotlib
* opencv-python
* tensorboard 2.15.x
* cloudpickle、protobuf 等兼容版本

不会出现 numpy / cv2 崩溃问题。

---

### ✅ **第四步：安装你的项目需要的少量库（来自你的 requirements）**

从你发的 requirements.txt 中
**只有这些库应该保留并安装：**

```bash
pip install requests jinja2 networkx
```

原因：

* `requests` 必须（你塔防环境会访问 [http://localhost:3000）](http://localhost:3000）)
* `jinja2` 如果你项目用就装，不用就算了
* `networkx` Stable Baselines3 Extra 用的

---

# 🚫 **不要安装以下这些（会破坏环境）**

* numpy==2.3.3 ❌
* fsspec==2025.9.0 ❌
* protobuf==6.32.1 ❌
* torch==2.8.0 ❌
* pandas==2.3.2（SB3 会自动装更兼容的）
* matplotlib==3.10.x（SB3 会自动装更稳定的）

你的 requirements.txt 是**坏的原环境 dump**，
不能直接重装。

---

### 🌟 最终健康的 RL 环境（我帮你列出最终版本）

你成功安装后 `pip list` 会类似：

```
torch 2.3.1
numpy 1.26.x
gymnasium 1.2.0
stable_baselines3 2.7.0
opencv-python 4.10.x
pandas 2.1.x
protobuf 4.x
tensorboard 2.15.x
```

这一套非常稳定，不会崩。

---

### 📌 然后启动塔防API服务
---
---
# 官方安装步骤

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

2. Execute script (default map):
    ```bash 
    python train.py
    ```
    ### 可能需要注释掉env = xxx.warp()哪一行
    Or with random maps:
    ```bash
    python train.py --random-maps custom-maps.json
    ```
3. Monitor training progress via TensorBoard (or at the end of training):
    ```bash
    tensorboard --logdir ./logs/
    ```
4. The trained model will be saved in the `models/` directory.

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
    python train.py --load-model ./path/to/maskable_ppo_tower_defense.zip
    ```

### Replay best agent game (works only for the default map for now)
1. Execute the replay script specifying the json file with the best agent actions:
    ```bash
    python replay_actions.py --actions-file ./models/date_time/best_episode_actions.json
    ```
    Optionally, you can save the frames to a `best_frames` directory next to the actions file by adding the `--save-frames` argument (for future loading purposes).

2. If you have already saved the frames, you can load them directly by using the `--load-frames` argument with the path to the `best_frames` directory (much faster):
    ```bash
    python replay_actions.py --load-dir ./models/date_time/best_frames
    ```