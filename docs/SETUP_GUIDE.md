# TowerDefense 容器环境配置指南

**适用环境：** 并行云 (Parallel Cloud)、Ubuntu 24.04、RTX 4090、Pytorch 2.7.0 容器

**核心原则：** 代码存放在共享存储 (`~/shared-nvme`)，复用系统预装的 PyTorch/CUDA，不使用虚拟环境，使用 `--break-system-packages` 补齐依赖。

---

## 第一步：配置网络与学术加速 (基础)

1.  **写入配置 (一键复制运行)：**
    ```bash
    cat >> ~/.bashrc <<EOF

    # --- Acceleration (Added by User) ---
    export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
    export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
    export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn,localhost,127.0.0.1"

    # 立即生效
    source ~/.bashrc
    ```

2.  **✅ 检查点 1：验证网络**
    ```bash
    # 1. 测试外网代理 (应返回 200 Connection established)
    curl -I https://huggingface.co

    # 2. 检查环境变量 (确认 no_proxy 包含 localhost)
    env | grep no_proxy
    ```

---

## 第二步：安装系统工具 & Node.js (基建)

**目的：** 安装开发工具 (Tmux) 和 游戏运行环境 (Node.js v22 + 图形库)。

1.  **安装系统依赖 (图形库用于 Canvas 编译)：**
    ```bash
    apt-get update
    apt-get install -y tmux htop git vim wget build-essential libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev
    ```

2.  **安装 Node.js v22 (LTS)：**
    ```bash
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
    ```

3.  **✅ 检查点 2：验证工具**
    ```bash
    node -v   # 应输出 v22.x.x
    npm -v    # 应输出 10.x.x
    tmux -V   # 应显示版本号
    ```

---

## 第三步：部署游戏端 TowerDefenseGame

1.  **进入项目并安装依赖：**
    ```bash
    cd ~/shared-nvme/TowerDefenseGame
    
    # 安装依赖 (Canvas 编译可能需要 1-2 分钟)
    npm install
    ```

2.  **✅ 检查点 3：浏览器运行测试**
    * 在终端运行：
        ```bash
        npm run dev
        ```
    * **操作：** 观察 VSCode 右下角弹窗，点击“在浏览器中打开”，或访问本地 `http://localhost:5173`。
    * **成功标志：** 浏览器能看到绿色的塔防游戏地图界面。
    * *测试完后按 `Ctrl+C` 关闭。*

---

## 第四步：部署算法端 TowerDefenseRL (大脑)

**重点：** 不要直接运行 `requirements.txt`，防止覆盖系统 PyTorch。只安装 RL 核心库。

1.  **进入目录：**
    ```bash
    cd ~/shared-nvme/TowerDefenseRL
    ```

2.  **安装 Python 依赖 (混合安装法)：**
    * 使用清华源加速，且允许打破系统限制。
    * **不要安装 torch！** 让 SB3 自动适配容器自带的 NVIDIA 优化版 torch。
    ```bash
    # 升级 pip
    pip install --upgrade pip --break-system-packages

    # 安装 Gymnasium, Stable-Baselines3, Tensorboard 等核心库
    pip install --break-system-packages gymnasium==1.2.0 stable_baselines3==2.7.0 sb3_contrib==2.7.0 https://pypi.tuna.tsinghua.edu.cn/simple/
    ```

3.  **✅ 检查点 4：验证 Python 环境**
    运行以下命令，必须全是 ✅ 才算成功：
    ```bash
    python -c "import gymnasium; import stable_baselines3; import torch; print(f'✅ SB3与Gym导入成功'); print(f'✅ Torch版本: {torch.__version__} (应含nv/cuda)'); print(f'✅ CUDA可用: {torch.cuda.is_available()} (应为True)')"
    ```

---

## 第五步：联调测试 (大脑连接身体)

**核心逻辑：** 必须先启动游戏 API，再运行 Python 脚本。

1.  **准备测试脚本 (`test_env.py`)：**
    在 `~/shared-nvme/TowerDefenseRL/` 下创建
    ```python
    import gymnasium as gym
    import gymnasium_env.envs
    try:
        env = gym.make("gymnasium_env/TowerDefenseWorld-v0")
        env.reset()
        env.step(env.action_space.sample())
        print("\n🎉 恭喜！RL 环境与游戏服务器连接正常！")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
    ```

2.  **执行联调：**
    * **终端窗口 A (bash)：**
        ```bash
        cd ~/shared-nvme/TowerDefenseGame
        npm run start:api
        # 保持开启，不要关闭！
        ```
    * **终端窗口 B (bash)：**
        ```bash
        cd ~/shared-nvme/TowerDefenseRL
        # 检查服务端是否存活
        curl -I http://localhost:3000/  # 应返回 200 OK
        
        # 运行测试脚本
        python test_env.py
        ```

3.  **✅ 检查点 5：** 终端 B 输出 `🎉 恭喜！RL 环境与游戏服务器连接正常！`

---

## 第六步：冒烟测试 (Smoke Test)

**目的：** 确保 `train.py` 能跑通全流程（训练、保存模型、记录日志）。

1.  **修改配置：**
    打开 `TowerDefenseRL/train.py`，临时修改训练时间：
    ```python
    # -------- 全局配置 --------
    hours_to_train = 0.005  # 原来是 1，改成极短时间 (约18秒) 用于测试
    ```

2.  **运行训练：**
    ```bash
    python train.py
    ```

3.  **✅ 检查点 6：**
    * 终端出现进度条。
    * 不报错，正常结束。
    * `models/` 目录下生成了新的日期文件夹。
    * `logs/` 目录下有新的 Tensorboard 日志。

4.  **复原配置：**
    测试成功后，记得把 `train.py` 改回去：
    ```python
    hours_to_train = 1  # 或者你想要的任何时长
    ```

---

## 📅 日常开发工作流 (Cheat Sheet)

每次重新连接容器后，建议使用 `tmux` 进行双开操作：

1.  **SSH 连接容器**。
2.  **创建 Tmux 会话：** `tmux new -s train`
3.  **上半屏 (游戏服务)：**
    ```bash
    cd ~/shared-nvme/TowerDefenseGame
    npm run start:api
    ```
4.  **切分屏幕：** 按 `Ctrl+b` 松开，再按 `"` (双引号)。
5.  **下半屏 (训练脚本)：**
    ```bash
    cd ~/shared-nvme/TowerDefenseRL
    # 开启 Tensorboard (可选)
    tensorboard --logdir ./logs/ --port 6006 &
    # 开始训练
    python train.py
    ```
6.  **挂起后台：** 按 `Ctrl+b` 松开，再按 `d`。
7.  **以后回来查看：** `tmux attach -t train`。