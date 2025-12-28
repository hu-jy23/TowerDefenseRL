import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class GeneralizationCallback(BaseCallback):
    """
    专门用于测试泛化性的 Callback。
    它会在评估环境中运行测试，并记录 'wave_number' 到 TensorBoard。
    """
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5, deterministic: bool = True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        # 按照设定的频率触发评估
        if self.n_calls % self.eval_freq == 0:
            # 定义一个内部函数，用于在 evaluate_policy 运行时抓取 info
            wave_numbers = []
            
            def grab_info_callback(locals_, globals_):
                # 当一个 episode 结束时 (dones=True)
                if locals_['dones'][0]:
                    info = locals_['infos'][0]
                    if 'wave_number' in info:
                        wave_numbers.append(info['wave_number'])

            # 运行评估
            # 注意：这里我们只关心 info 抓取，所以忽略 evaluate_policy 返回的 reward
            evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                callback=grab_info_callback, # 注入抓取逻辑
                warn=False
            )

            # 记录数据到 TensorBoard
            if len(wave_numbers) > 0:
                mean_wave = np.mean(wave_numbers)
                self.logger.record("eval/mean_wave_number", mean_wave)
                
                if self.verbose > 0:
                    print(f"Eval Result at step {self.num_timesteps}: Mean Wave = {mean_wave:.2f}")

        return True