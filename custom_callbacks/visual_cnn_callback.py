import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class VisualCNNCallback(BaseCallback):
    """
    自定义回调函数：在第 17 波时捕获 CNN 通道数据并进行可视化。
    """
    def __init__(self, save_dir_base, verbose=0):
        super(VisualCNNCallback, self).__init__(verbose)
        self.save_dir_base = save_dir_base
        self.visual_channels_dir = os.path.join(save_dir_base, "visual_channels")
        self.capture_count = 0
        self.max_captures = 3
        self.target_wave = 17
        self.last_wave = -1
        
        # 通道名称映射 (根据 tower_defense_world.py)
        self.channel_names = {
            0: "Path",
            1: "Path Distance",
            2: "Firepower Map",
            3: "AOE Influence",
            4: "Invested Capital",
            5: "Efficiency",
            6: "Enemy Density",
            7: "Total HP",
            8: "Speed Threat",
            9: "Enemy Type (Tank)",
            10: "Build Mask",
            11: "Tower Presence",
            12: "Tower Readiness"
        }

    def _on_step(self) -> bool:
        # 获取环境信息
        infos = self.locals.get("infos")
        if not infos:
            return True
        
        # 仅处理第一个环境 (VecEnv)
        info = infos[0]
        current_wave = info.get("wave_number", 0)
        
        # 触发条件：进入第 17 波，且捕获次数未达上限
        if current_wave == self.target_wave and self.last_wave != self.target_wave and self.capture_count < self.max_captures:
            self._capture_data(info)
            self.capture_count += 1
            
        self.last_wave = current_wave
        return True

    def _capture_data(self, info):
        # 获取观测值
        obs = self.locals.get("new_obs")
        if obs is None or 'map_input' not in obs:
            return
        
        # map_input 形状为 (Batch, Channels, Rows, Cols)
        map_input = obs['map_input']
        if hasattr(map_input, "cpu"):
            map_input = map_input.cpu().numpy()
        
        grid_data = map_input[0] # 取 Batch 中的第一个
        
        capture_dir = os.path.join(self.visual_channels_dir, f"visual_cnn_{self.capture_count}")
        data_dir = os.path.join(capture_dir, "data")
        figures_dir = os.path.join(capture_dir, "figures")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        metadata = {
            "money": info.get("money"),
            "lives": info.get("lives"),
            "game_time": info.get("game_time"),
            "wave_number": info.get("wave_number"),
            "step": self.num_timesteps
        }
        
        path_grid = grid_data[0] # Channel 0 是路径
        
        n_channels = grid_data.shape[0]
        for i in range(n_channels):
            channel_data = grid_data[i]
            
            # 1. 保存 .npy
            np.save(os.path.join(data_dir, f"channel_{i}.npy"), channel_data)
            
            # 2. 保存 .json
            channel_json = {
                "metadata": metadata,
                "channel_index": i,
                "channel_name": self.channel_names.get(i, "Unknown"),
                "grid": channel_data.tolist()
            }
            with open(os.path.join(data_dir, f"channel_{i}.json"), "w") as f:
                json.dump(channel_json, f, indent=4)
            
            # 3. 绘制图片
            self._plot_channel(i, channel_data, path_grid, figures_dir, metadata)

    def _plot_channel(self, idx, data, path_grid, save_path, metadata):
        rows, cols = data.shape
        # 动态调整图片大小，确保格子标注清晰
        fig, ax = plt.subplots(figsize=(cols * 0.6 + 2, rows * 0.6 + 1))
        
        # 绘制热力图，数值大的颜色深
        im = ax.imshow(data, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=max(1.0, np.max(data)))
        
        # 额外勾勒路径 (Channel 0)
        # 使用 contour 绘制路径轮廓
        ax.contour(path_grid, levels=[0.5], colors='grey', linewidths=1, alpha=0.3)
        
        # 在每个格子里标注数值
        for r in range(rows):
            for c in range(cols):
                val = data[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", 
                        color="black", fontsize=6, alpha=0.8)
        
        channel_name = self.channel_names.get(idx, f"Channel {idx}")
        ax.set_title(f"{channel_name} (Wave {metadata['wave_number']}, Step {metadata['step']})")
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"channel_{idx}.png"), dpi=150)
        plt.close()
