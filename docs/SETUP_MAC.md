# Mac (Apple Silicon) 强化学习塔防项目环境配置指南

> **适用系统** ：macOS Sequoia (15.x) 及以上
> **硬件架构** ：Apple Silicon (M1/M2/M3/M4)
> **核心目标** ：在 Mac 上配置本地环境，并开启 MPS (Metal Performance Shaders) 显卡加速替代 CUDA。

---

## 第一阶段：修复与安装系统工具 (Homebrew & Tmux)

由于 macOS 15 系统更新导致 Homebrew 核心逻辑变更，常规安装可能会报 `:date, :because` 或 API 连接错误。

### 1. 修复 Homebrew 并安装 Tmux

如果你的 `brew install` 报错，请使用以下“混合 API 模式”命令强制安装：

**Bash**

```
# 1. 强制指定 API 镜像源（绕过本地核心代码版本不匹配问题）
HOMEBREW_API_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api" brew install tmux

# 2. 验证安装
tmux -V
# 输出应类似：tmux 3.6a
```

### 2. (备选方案) 如果上述失败，手动克隆核心库

如果 API 模式无效，需手动从国内镜像拉取核心仓库：

**Bash**

```
# 清理旧目录
rm -rf /opt/homebrew/Library/Taps/homebrew/homebrew-core
rm -rf /opt/homebrew/Library/Taps/homebrew/homebrew-cask

# 手动克隆（使用中科大源）
cd /opt/homebrew/Library/Taps/homebrew
git clone https://mirrors.ustc.edu.cn/homebrew-core.git --depth=1
git clone https://mirrors.ustc.edu.cn/homebrew-cask.git --depth=1

# 绕过浅克隆限制进行安装
HOMEBREW_NO_INSTALL_FROM_API=1 HOMEBREW_NO_AUTO_UPDATE=1 brew install tmux
```

---

## 第二阶段：配置 Python 强化学习环境

Mac M 系列芯片对 Python 版本敏感，强烈建议使用 **Python 3.10** 以获得最佳兼容性。

### 1. 创建 Conda 环境

**Bash**

```
# 创建环境（推荐 3.10，避开 3.13 的兼容性问题）
conda create -n rl python=3.10 -y

# 激活环境
conda activate rl
```

### 2. 升级 Pip（关键）

M 系列芯片需要新版 Pip 才能识别 `arm64` 架构的安装包。

**Bash**

```
python -m pip install --upgrade pip
```

### 3. 安装核心依赖 (Gymnasium & SB3)

 **注意** ：部分国内镜像源可能缺少 Mac ARM64 的包，若报错 `(from versions: none)`，请使用官方源。

**Bash**

```
# 安装强化学习核心库
pip install gymnasium stable-baselines3 sb3-contrib -i https://pypi.org/simple

# 安装辅助工具 (视频录制、Web通信、可视化)
pip install moviepy tensorboard requests -i https://pypi.org/simple

# 安装/更新 PyTorch (确保支持 MPS)
pip install --upgrade torch torchvision torchaudio
```

---

## 第三阶段：配置前端游戏环境 (Node.js)

### 1. 解决权限与依赖丢失问题

如果是复制过来的项目，`node_modules` 可能会损坏或没有执行权限。

**Bash**

```
cd ../TowerDefenseGame

# 1. 赋予执行权限（解决 Permission denied）
chmod -R +x node_modules/.bin

# 2. 如果运行报错，建议执行“核弹级”重装
rm -rf node_modules package-lock.json
npm install
```

### 2. 启动测试

**Bash**

```
# 启动 API 服务器（必须保持运行）
npm run start:api

# (可选) 启动前端画面
npm run dev
```

---

## 第四阶段：代码适配 Mac 显卡加速 (MPS)

原代码通常写死 `device="cuda"`，在 Mac 上会报错或回退到 CPU。需修改 `train.py`。

### 修改 `train.py` 中的 `make_model` 函数

将原有代码替换为以下自动检测逻辑：

**Python**

```
def make_model(algo, env, load_model_path=None, tensorboard_log="./logs/"):
    import torch
  
    # --- 自动设备检测 ---
    if torch.cuda.is_available():
        device = "cuda"       # NVIDIA 显卡
    elif torch.backends.mps.is_available():
        device = "mps"        # Mac M1/M2/M3...
    else:
        device = "cpu"        # 兜底方案
  
    print(f"🚀 Detected device: {device}")
    # ------------------

    # 在创建或加载模型时，将 device 参数传入
    if algo == "dqn_hierarchical":
        if load_model_path:
            model = DQN.load(load_model_path, env, tensorboard_log=tensorboard_log, device=device)
        else:
            model = DQN("MlpPolicy", env, verbose=1, device=device, ...) # 省略其他参数
          
    # ... 其他算法同理
    return model
```

---

## 第五阶段：正式运行工作流 (Best Practice)

使用 `tmux` 管理多个后台任务，防止意外关闭。

1. **新建 Tmux 会话**
   **Bash**

   ```
   tmux new -s rl_train
   ```
2. **启动游戏 API (窗口 1)**
   **Bash**

   ```
   cd TowerDefenseGame
   npm run start:api
   ```
3. **启动训练脚本 (窗口 2)**

   * 按下 `Ctrl + B`，松开，再按 `%` (切分屏幕)。
   * 在右侧新窗口操作：

   **Bash**

   ```
   conda activate rl
   cd ../TowerDefenseRL

   # 运行训练
   python train.py --algo dqn_hierarchical
   ```
4. **验证加速**

   * 训练开始后，打开 Mac 的 **Activity Monitor (活动监视器)** ->  **GPU** 。
   * 如果看到 `python3.10` 占用 GPU 资源，说明 MPS 加速开启成功！

---

### 常见问题排查 (Troubleshooting)

* **报错 `ModuleNotFoundError: No module named 'moviepy'`**
  * 解法：`pip install moviepy -i https://pypi.org/simple`
* **报错 `ImportError: ... tensorboard is not installed`**
  * 解法：`pip install tensorboard`
* **报错 `sh: vite: Permission denied`**
  * 解法：`chmod -R +x node_modules/.bin`
* **训练速度极慢且显示 `Using cpu device`**
  * 解法：检查 `train.py` 是否已修改 `device="mps"` 逻辑，且 `torch` 版本是否是最新的。
