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

# TODO list
1. - [ ] agent 只要返回就返回有意义的action，减少mask的使用 -- 目前的mask是“事后惩罚”，而不是“事前禁止”
2. - [ ] DQN 实现，baseline DQN v.s. PPO 官方实现
3. - [ ] DQN 优化（缩小 action space / double DQN / dueling DQN， offline study）
4. - [ ] 是否可以引入人类先验？如何表示人类策略？（模仿学习 / 离线 RL）
5. - [ ] 引入LLM 作为策略指导/打分

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


# Infra
## Observation
- Observation o（代理实际可见）
  - 类型与范围：spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=float32)，见 TowerDefenseRL/gymnasium_env/envs/
    tower_defense_world.py:42。
  - 组成（每个分量都归一化到 [0,1]）：
      - 全局特征（5 + 2×|path cells|），见 142 与 148：
          - [0] 游戏时间 / 上限、[1] 波数 / 上限、[2] 金钱 / 上限、[3] 生命 / 上限、[4] gameOver(0/1)
          - [5:] 路径上每个格子的归一化坐标 x,y 串接
      - 塔槽位特征（固定槽位，最大塔数 × 每塔 （5+|塔类型|）），见 151–159：
          - 每塔：active(1/0), x, y, 攻速冷却/最慢塔冷却, dps/全塔最大 dps, 以及塔类型 one-hot
      - 敌人槽位特征（固定槽位，最大敌人数 × 每敌 （5+|敌类型|）），见 161–169：
          - 每敌：active(1/0), x, y, 当前血量/满血, pathProgress(0..1), 敌类型 one-hot
  - 维度如何确定（默认地图，代码取自 GameConfig）：
      - 地图宽高 900×600，网格 50px → 18×12 共 216 格（TowerDefenseGame/src/core/GameConfig.ts:54）。
      - 路径像素长约 2500 → 路径格数 ≈ 2500/50 = 50。
      - 全局特征维数：5 + 2×50 = 105。
      - 最大塔数：216 - 50 = 166；塔类型数=3 → 每塔 8 维 → 166×8=1328。
      - 最多同时在场敌人数（保守上界）：依据波设置计算约 33 个（TowerDefenseRL/gymnasium_env/envs/
        tower_defense_world.py:212–221）。敌类型数=3 → 每敌 8 维 → 33×8=264。
      - 总观测维数 N ≈ 105 + 1328 + 264 = 1697。
  - 注意：这是一条“固定长度的拼接向量”，前若干维是全局与路径常量，中间一段是塔槽位，尾部是敌人槽位，未占用槽位全 0。

- 例子1：刚 reset 完（还没放塔，也没敌人）

  - 全局段（索引 0..4）
      - [0] gameTime/1300 = 0.0
      - [1] waveNumber/50 = 0.0
      - [2] money/999 ≈ 40/999 ≈ 0.040
      - [3] lives/3 = 1.0
      - [4] gameOver = 0.0
  - 路径坐标段（索引 5..104，共 100 个数）
      - 这是路径每个格中心的坐标按 (x/900, y/600) 依次串起来的常量。例如，假如前两个路径格中心大概是 (75,25)、(75,75)，
        则：
          - [5]=75/900≈0.083，[6]=25/600≈0.042
          - [7]=75/900≈0.083，[8]=75/600=0.125
      - 直到把约 50 个格子的 (x/900,y/600) 都放完，共 100 个值。
  - 塔槽位段（索引 105..1432，共 166 个槽 × 8 维）
      - 因为还没塔，全部为 0。
  - 敌槽位段（索引 1433..1696，共 33 个槽 × 8 维）
      - 因为还没敌，也全部为 0。
```bash
›     # encodes the self game state into a tensor of shape self.observation_space.shape
      def __get_observation(self) -> np.ndarray:
          shape = self.observation_space.shape
          if shape is None:
              raise ValueError("Observation space shape is not defined")
          observation = np.zeros(shape, dtype=np.float32)

          # global features normalized
          observation[2] = self.game_state["money"] / self.game_info["max_global_info"]["money"]
          observation[3] = self.game_state["lives"] / self.game_info["max_global_info"]["lives"]
          observation[4] = self.game_state["gameOver"]
          observation[5:5+len(self.path_cells_coordinates_normalized)] = self.path_cells_coordinates_normalized
          #observation[4:4+self.map_horizontal_cells*self.map_vertical_cells] = self.__calculate_grid_map()

          # tower features normalized
          for idx, tower in enumerate(self.game_state["towers"]):
  #注意self.global_feature_count = 5+len(self.path_cells_coordinates_normalized) # game time, wave number, money, lives,
  game over, path cells coordinates
              offset = self.global_feature_count + idx * self.features_per_tower
              observation[offset] = 1 # active
              observation[offset+1] = tower["position"]["x"] / self.game_info["map"]["width"] # normalized x
              observation[offset+2] = tower["position"]["y"] / self.game_info["map"]["height"] # normalized y
              observation[offset+3] = tower["attackCooldown"] / self.game_info["slower_tower_sample"]["attackCooldown"]
  # normalized attack cooldown
              observation[offset+4] = self.tower_types[self.tower_type_to_index[tower["type"]]]["dps"] /
  self.max_tower_dps # normalized dps
              observation[offset+5+self.tower_type_to_index[tower["type"]]] = 1 # one-hot encoding type

          # enemy features normalized
          for idx, enemy in enumerate(self.game_state["enemies"]):
              #注意self.tower_feature_count = self.max_towers * self.features_per_tower
              offset = self.global_feature_count + self.tower_feature_count + idx * self.features_per_enemy
              observation[offset] = 1 # active
              observation[offset+1] = enemy["position"]["x"] / self.game_info["map"]["width"] # normalized x
              observation[offset+2] = enemy["position"]["y"] / self.game_info["map"]["height"] # normalized y
              observation[offset+3] = enemy["currentHealth"] / enemy["fullHealth"] # normalized health
              observation[offset+4] = enemy["pathProgress"]
              observation[offset+5+self.enemy_type_to_index[enemy["type"]]] = 1 # one-hot encoding type

          return observation
```
### 注意这里用到的global_feature_count和tower_feature_count都是地图设定之初就设定好了的。
- 这些 offset 与维度（global_feature_count、tower_feature_count、enemy_feature_count）确实是“由地图几何决定”的；
- 但它们只在环境构造时（init 里第一次 GET /info）根据“当时的地图”计算一次并固定下来。之后如果用 RandomMapWrapper 在
    reset 前换图，环境并不会重新计算这些量。除非你重建一个新的 env 实
  例，否则它不会随每张随机地图重新定 shape 或重算这些偏移。
          
    observation是graph-specific的，在进入每张图初始化的时候就customize了一个observation

## Action
### Action a（代理输出）

  - 空间：spaces.MultiDiscrete([A, T, X, Y])，见 TowerDefenseRL/gymnasium_env/envs/tower_defense_world.py:29。
      - A=动作类型数；T=塔类型数；X/Y=横纵坐标格数。
      - 默认配置下：A=2（NONE、BUILD_TOWER，见 TowerDefenseGame/src/api.ts:208）、T=3（archer/cannon/sniper，
        GameConfig.ts:118）、X=18、Y=12。
  - 语义（step 时如何落地），见 TowerDefenseRL/gymnasium_env/envs/tower_defense_world.py:70–76：
      - 如果选的是 BUILD_TOWER，会把塔类型与格点转换为像素中心坐标并提交给服务端。
      - 如果选 NONE，坐标与塔型分量被忽略（仍占位于动作向量中）。
  - 动作掩码（非法动作屏蔽），见 120–133：
      - 钱不够时禁用 BUILD_TOWER（A 维的对应类目）；
      - 对于单个塔类型，若钱不够或未解锁则屏蔽该塔型（T 维的类目）；
      - X/Y 维不屏蔽（MultiDiscrete 无法表达“特定格子非法”的交叉约束），坐标非法由环境返回 -1 小惩罚处理。
      - − 持币过多（超过最贵塔价）：线性罚（鼓励把钱转换为战力）
      - − 掉命：每点生命 -20
      - − 游戏结束：-100
      - 额外：若步进请求因非法动作被服务器拒绝（如放在路径上/占用格），本步直接记 -1，并返回“未更新”的观测，见 81–86。
