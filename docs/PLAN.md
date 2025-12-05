# TowerDefenseRL 项目实施路线图

## 1. 核心思路：应对塔防 RL 的两大挑战

在动手之前，我们需要时刻铭记塔防游戏（Tower Defense）区别于一般 Atari 游戏的两个核心痛点。所有的算法选型和 Trick 都是为了杀死这两个恶魔：

1.  **巨大的离散动作空间 (Combinatorial Action Space)**
    * **问题**：动作由 `[操作类型, 塔类型, X坐标, Y坐标]` 组成。在现有代码中，动作空间大小为 $2 \times 3 \times 18 \times 12 \approx 1300$ 维。
    * **后果**：DQN 难以遍历所有动作，探索效率极低，且容易陷入“不知道该选哪个格子”的局部最优。
2.  **稀疏且延迟的奖励 (Sparse & Delayed Rewards)**
    * **问题**：建造防御塔的动作发生在 $t$ 时刻，但收益（打死怪物）可能发生在 $t+100$ 时刻。且建造本身是扣钱的（负奖励）。
    * **后果**：Agent 容易学会“什么都不做”（因为攒钱不扣分），或者无法理解“花钱造塔”与“后来没死”之间的因果关系。

---

## 2. 通用基建 (General Tricks & Infrastructure)

> **关于分工的重要建议**：
> 这部分是所有赛道的基础，直接决定了模型上限。**强烈建议在项目启动的前 2-3 天，由全员共同攻克**。
> * **GridObservationWrapper**：建议由负责 PPO 的同学主笔（因为 CNN 对数据格式最敏感），其他同学协助测试。写好后作为一个公共模块（`common_utils.py`），供三个赛道共同调用。

### 2.1 状态空间重构：Grid-based Feature Map
* **现状**：目前 `tower_defense_world.py` 输出的是 1697 维的扁平向量，丢失了空间结构。
* **改进**：构建 $C \times H \times W$ 的 3D 张量，让卷积神经网络（CNN）能像人类一样“看”懂地图。
    * **Dimensions**: $12 \times 18$ (对应地图网格)。
    * **Channels (建议 5 层)**:
        1.  **Path Mask**: 0/1，标识路径位置（告诉 Agent 敌人从哪走）。
        2.  **Tower Type**: 0/1/2/3，标识已建塔的位置和类型。
        3.  **Enemy Density**: 该格子内敌人归一化血量之和（类似热力图）。
        4.  **Range Coverage**: 当前所有塔的火力覆盖叠加图。
        5.  **Buildable Mask**: 0/1，标识哪些格子能造塔（Action Mask 的空间版）。
* **收益**：极大地提升泛化性，Agent 能学会“堵路口”、“补漏”等空间策略，而不是死记硬背坐标。

### 2.2 奖励塑形：Reward Shaping
* **现状**：主要靠“击杀奖励”和“建塔瞬间的 heuristic 奖励”（如 `path_coverage`）。
* **改进**：引入**有效伤害奖励 (Damage Reward)**。
    * 修改 `step` 函数，计算 `old_total_enemy_hp - new_total_enemy_hp`，将造成的真实伤害按比例（如 0.1）作为 Reward。
* **收益**：缓解奖励延迟，直接告诉 Agent “掉血就是好事”，鼓励造出 DPS 最高的塔。

---

## 3. 三大赛道规划 (Three Tracks)

采用 **“一人一赛道，纵向深挖”** 的策略。每人负责一种算法范式，从 V1 (保底) 做到 V3 (进阶)。

### 赛道 A：价值派进阶 (The Optimizer)

* **核心算法**：**DQN (Deep Q-Network)**
* **负责人特点**：擅长理解 Q-Value，喜欢钻研网络结构。
* **核心挑战**：如何让 DQN 在大动作空间下不发散？

#### **实施路线图**

* **Version 1 (保底 - MlpPolicy + Flatten)**
    * **任务**：使用现有的 `FlattenMultiDiscreteAction` + `SB3 DQN` 跑通流程。
    * **目标**：获得一个 Baseline 分数，观察 Loss 曲线。
    * **工作量**：极低（现有代码已支持）。

* **Version 2 (核心 - Action Masking + CNN)**
    * **任务**：
        1.  接入通用的 `GridObservationWrapper`，启用 `CnnPolicy`。
        2.  **攻坚点**：实现 **Masked DQN**。SB3 的 DQN 原生不支持 Masking。需要修改 DQN 的 `predict` 方法或自定义 Policy，在 `argmax Q` 之前将非法动作（没钱/位置占用）的 Q 值置为 $-\infty$。
    * **参考**：*INF581_report* 中提到 Masking 能让分数从 811 提升到 1291。
    * **工作量**：中等。

* **Version 3 (探索 - 结构优化)**
    * **任务**：
        1.  开启 **Dueling DQN** (在 `policy_kwargs` 中设置 `dueling=True`)。
        2.  尝试 **Multi-step Learning** (N-step returns)，缓解奖励延迟。
    * **工作量**：中等。

---

### 赛道 B：策略派进阶 (The Architect)
* **核心算法**：**Maskable PPO (Proximal Policy Optimization)**
* **负责人特点**：擅长调参，对 CNN/Attention/Transformer 架构熟悉。
* **核心挑战**：如何提升状态表征的有效性，像 AlphaStar 一样“看”地图？

#### **实施路线图**

* **Version 1 (保底 - CNN Policy)**
    * **任务**：接入通用的 `GridObservationWrapper`，使用 `MaskablePPO` + `CnnPolicy`。
    * **目标**：证明 CNN 在随机地图上的收敛速度优于 MLP。
    * **工作量**：低。

* **Version 2 (核心 - 时空感知 FrameStack)**
    * **任务**：
        1.  使用 `gym.wrappers.FrameStack`，将最近 4 帧的 Grid Map 叠加。
        2.  输入变为 $(4 \times C) \times H \times W$。
    * **原理**：单帧图像看不出敌人的移动速度，叠加帧能让卷积层感知到“怪物流”的流向和速度。
    * **工作量**：低（主要在调参）。

* **Version 3 (探索 - Attention 机制)**
    * **任务**：在 CNN 提取特征后，加入一层 **Self-Attention** (Transformer Block) 或 **Spatial Attention**。
    * **原理**：模拟人类玩家的注意力，聚焦于怪海最密集的区域，忽略无关背景。
    * **工作量**：高（需要自定义 `features_extractor` 网络）。

---

### 赛道 C：混合架构 (The Strategist)
* **核心算法**：**HRL (Hierarchical RL) / Hybrid Approach**
* **负责人特点**：逻辑强，代码能力强，喜欢写规则/脚本。
* **核心挑战**：通过架构创新，彻底绕开动作空间爆炸的问题。
* **参考**：*2406.07980v1* 论文思路：RL 做高层决策，脚本做低层执行。

#### **实施路线图**

* **Version 1 (保底 - 纯规则 Agent)**
    * **任务**：不训练 RL。写一个 Python 函数 `heuristic_agent(obs)`。
    * **规则**：有钱就造塔，造塔只造在 `path_coverage` 最大的位置。
    * **目标**：提供一个超强的 Baseline，用来打脸“还没训练好的 RL”。
    * **工作量**：中。

* **Version 2 (核心 - RL 指挥官 + 规则工兵)**
    * **任务**：
        1.  **修改 Action Space**：动作只剩 4 个：`[0: 攒钱, 1: 造Archer, 2: 造Cannon, 3: 造Sniper]`。
        2.  **Worker 实现**：当 RL 选择“造 Cannon”时，Python 脚本自动遍历地图，找到当前覆盖率最高且空闲的格子，执行建造。
        3.  **训练**：用 PPO 训练这个高层 Agent。
    * **成效**：**收敛神速**。因为动作空间从 1296 降到了 4！
    * **工作量**：中高（需要重写 Env 的 `step` 逻辑）。

* **Version 3 (探索 - 泛化性与鲁棒性)**
    * **任务**：因为 V2 训练很快，你有大把时间测试。
    * **实验**：在 Map 1-5 训练，在 Map 6-10 测试（Zero-shot Transfer）。
    * **工作量**：低（主要是跑实验）。

---

## 4. 算力与时间分配

鉴于你们拥有 **3张 RTX 4090**：
* **GPU 不是瓶颈**：这种规模的 CNN 训练对 4090 来说非常轻松。
* **CPU 是瓶颈**：Node.js 游戏服务器的运行速度决定了采样速度。

**加速建议**：
在 `train.py` 中使用 `SubprocVecEnv` 开启多进程采样：
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 开启 8-16 个并行环境，利用 CPU 多核加速与 Node.js 的交互
# 这样能让 GPU 始终有数据吃，训练速度提升 5-8 倍
env = make_vec_env(env_id, n_envs=8, vec_env_cls=SubprocVecEnv)