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
“事前禁止”
1. - [ ] 跑通sb3-DQN ，baseline DQN v.s. PPO 官方实现
2. - [ ] handmade DQN
3. - [ ] DQN 优化（缩小 action space / double DQN / dueling DQN， offline study）
4. - [ ] agent 只要返回就返回有意义的action，减少mask的使用 -- 目前的mask是“事后惩罚”，而不是“事前禁止”
5. - [ ] 是否可以引入人类先验？如何表示人类策略？（模仿学习 / 离线 RL）
6. - [ ] 引入LLM 作为策略指导/打分

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

## 状态 s 与观测 o 的关系

  - s 是服务端游戏引擎的真实状态（时间、波数、金钱、生命、塔列表、敌人列表、路径进度等，见 TowerDefenseGame/src/
    api.ts:101 与 GameEngine 相关模块）。
  - o 是按上述规则把 s 归一化/铺平成固定长度向量后的结果，供策略网络直接输入；agent 不读原始图像。
  ## Episode 与 done 定义

  - 终止 terminated 条件（任一为真），见 69–87 与 87 之后逻辑：
      - gameOver=true，或达上限：波数≥50、金钱≥999（这些上限由 /info 提供，TowerDefenseGame/src/api.ts:200）
  - 截断 truncated 条件：游戏时间≥1300（同样来自 /info，TowerDefenseGame/src/api.ts:200）
  - reset 返回初始观测与 info，见 55–67。

## 训练时 agent 实际“读到的数据”

  - 每一步：o_t（约 1697 维 float32 向量）、可选动作掩码 m_t（长度 A+T+X+Y 的布尔串）、根据策略采样的 a_t（4 维
    MultiDiscrete）、环境反馈 r_t、下一个 o_{t+1}，以及 done 标志。
  - info 只用于日志/回调（波数、各塔数量、本回合动作序列等），不进入策略更新，见 173–184 与自定义回调 TowerDefenseRL/
    custom_callbacks/*。
  - 渲染帧仅用于可视化/录像，不参与训练输入，见 render 定义 69 之后与 TowerDefenseRL/gymnasium_env/envs/
    tower_defense_world.py:69–87。

## Reward
  - 奖励 r_t 的分解（对应环境里的奖励函数，见 TowerDefenseRL/gymnasium_env/envs/tower_defense_world.py:223）
      - +1 击杀奖励：这步敌人数量从 2 变 1，多了1个击杀 → +1
      - +6 过波奖励：波数从 2 升到 3，规则是“新波数×2” → 3×2=+6
      - 0 建塔奖励：这步没新建塔 → 0
      - −2 囤钱惩罚：钱=52，最贵塔=50，超过部分罚 (52-50)=−2
      - −20 掉命惩罚：掉了 1 条命 → −20
      - 0 结束惩罚：没结束 → 0
      - 合计：1 + 6 − 2 − 20 = −15
  - 观测 o_{t+1} 的关键变化（对应观测编码函数，见 TowerDefenseRL/gymnasium_env/envs/tower_defense_world.py:136）
      - [1] 波数归一化：从 2/50 变 3/50=0.06（波数上升）
      - [3] 生命归一化：从 2/3≈0.667 变 1/3≈0.333（少了一条命）
      - 敌人槽位：这步后“存活的敌人数量”从 2 变 1
          - 环境会按当前敌人列表，重头把“第 0 槽、1 槽、…”依次填上 active=1 的敌人特征，其余槽清零
          - 所以你看到的是“前面若干槽 active=1，后面的槽 active=0”；从 2→1 的变化就是有一个槽从一整段非零变成全 0

# DQN
› 能调包吗？自己实现困难吗？


• 结论

  - 直接“调包用 DQN”可以，但要先把动作空间从 MultiDiscrete([2,3,18,12]) 扁平成 Discrete(1296)。SB3 的 DQN不支持
    MultiDiscrete。
  - 自己实现 DQN 难度不高，但要做的工程比直接用 SB3 大很多；而且本项目的动作巨大且带掩码，纯 DQN 学习效率会明显弱
    于 MaskablePPO。

  能否“调包”

  - 可以：用 Stable-Baselines3 的 DQN（或 sb3_contrib 的 QRDQN），前提是把动作改造为 Discrete。
  - 不能直接对接 MultiDiscrete：SB3 的 DQN不支持 MultiDiscrete/动作掩码。

  推荐接入方案（最小改动）

  - 用一个 ActionWrapper 把 MultiDiscrete 扁平化：
      - n_actions = 2318*12 = 1296
      - 扁平编码：idx = a*(TXY) + t*(XY) + x(Y) + y
      - 反解码：依次整除取模还原 [a,t,x,y]
  - 在 train.py 的 make_model 里加一个 “dqn” 分支，用 SB3 DQN 包起来训练。

  样例代码（可直接放进 repo）

  - 扁平动作包装器（新建 gymnasium_env/wrappers/flatten_multidiscrete.py）：
      - 逻辑：新 action_space 改为 Discrete(n_actions)，step 时把离散 id 解码成原 MultiDiscrete 向量后再转发给底层 env。
  - 训练脚本改动思路（在 TowerDefenseRL/train.py 的 make_model 加一个 elif algo == "dqn":）
      - from stable_baselines3 import DQN
      - 包装 env：env = FlattenMultiDiscreteAction(env)
      - model = DQN("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000, batch_size=256, train_freq=4,
        target_update_interval=1_000, learning_starts=10_000, gamma=0.99, verbose=1, tensorboard_log="./logs/")
  注意事项与风险

  - 动作掩码会丢失：DQN 没有 Maskable 版本。非法动作（建塔钱不够/未解锁/占用/在路径上）会被环境以 -1 轻罚处理，探索时会
    经常撞到，学习效率比 MaskablePPO 差。
  - 大动作空间：离散 1296 维 Q 输出，收敛更慢。建议：
      - 调大学习网络：policy_kwargs=dict(net_arch=[1024, 1024])
      - 合理探索策略：exploration_fraction≈0.2、final_eps≈0.05

  接入结果

  - 新增包装器：gymnasium_env/wrappers/flatten_multidiscrete.py 将 MultiDiscrete([A,T,X,Y]) 扁平成 Discrete(A*T*X*Y)，供
    SB3 DQN 使用。
  - 扩展训练脚本：TowerDefenseRL/train.py:1 支持 --algo dqn_sb3（别名 dqn）。内部自动包上扁平包装器，再构建 SB3 DQN。
  - 超参默认（可调）：buffer_size=100k, batch_size=256, train_freq=4, target_update_interval=2500, learning_starts=10k,
    gamma=0.99, exploration_fraction=0.2, exploration_final_eps=0.05, net_arch=[1024,1024], lr=3e-4。

  如何运行

  - PPO（原样）：
      - python train.py --algo ppo
  - DQN（SB3 版）：
      - python train.py --algo dqn_sb3
      - 也可用别名：python train.py --algo dqn
  - 其余参数沿用现有脚本：可加 --random-maps custom-maps.json（注意下方“地图一致性”）、或 --load-model 续训。

  文件与改动

  - 新增: TowerDefenseRL/gymnasium_env/wrappers/flatten_multidiscrete.py
      - 将 [a,t,x,y] 编码为单一 action_id ∈ [0, 1296) 并在 step() 中解码回去。
  - 修改: TowerDefenseRL/train.py:1
      - 新增 from stable_baselines3 import DQN 与 from gymnasium_env.wrappers.flatten_multidiscrete import
        FlattenMultiDiscreteAction
      - 在 make_model() 中加入 elif algo in ("dqn", "dqn_sb3"): 分支，自动包裹扁平化动作并创建/加载 DQN。
      - CLI 帮助文案包含 ppo | dqn_sb3。

  注意事项（强烈建议先按此来）

  - 地图一致性：现环境的观测 shape 在构造时由“初始地图”确定。若用随机地图训练，请确保所有地图的 path_cells 数量一致，否
    则建议先不要启用随机地图，或等我们改为固定维的“网格图特征”。PPO/DQN 都受此约束。
  - 动作掩码：DQN 不支持动作掩码。非法坐标/未解锁/钱不够会被环境以 -1 轻罚来兜底，探索阶段会更频繁撞到无效动作，收敛慢于
    MaskablePPO 属正常现象。可通过加大网络/回放/训练步数来弥补。
  - 回调兼容：Checkpoint/TensorBoard/“最佳动作保存”在 DQN 路径下依然可用；日志节奏与 PPO 略有差异但不影响使用。

  后续：handy DQN 端口

  - 你可以新增 --algo dqn_handy 分支，逐步实现以下模块对齐 SB3 DQN：
      - Q 网络（MLP 输出 1296 维）、目标网络、重放缓冲、ε-greedy 探索、训练循环（train_freq/target_update 逻辑）
      - 对照我设的 SB3 超参逐一复现，保证 fair 对比
  - 我可以帮你脚手架 dqn_handy 的最小可运行版本，再逐个模块替换/优化（双重 DQN、Dueling、优先经验回放等）。

  需要我现在把 --algo dqn_handy 的空分支先加好并放一个最小骨架（含 ReplayBuffer/网络/训练循环）吗？你可以边跑 sb3 版作为
  对照，边完善自研版。