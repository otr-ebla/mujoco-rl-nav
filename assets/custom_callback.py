from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import time, subprocess, shutil


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_results = []
        self.episode_time_lengths = []
        self.current_episode_rewards = []
        self.current_episode_lengths = []
        
        # Counters for statistics
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.total_episodes = 0

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs
        self.current_episode_lengths = [0] * n_envs

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        rewards = self.locals['rewards']
        infos = self.locals['infos']

        for i, done in enumerate(dones):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_lengths[i] += 1

            if done:
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                
                # Extract episode result from info
                if 'episode_result' in infos[i]:
                    result = infos[i]['episode_result']
                    self.episode_results.append(result)
                    
                    # Update counters based on episode result
                    if result == "success":
                        self.success_count += 1
                    elif result == "collision":
                        self.collision_count += 1
                    elif result == "timeout":
                        self.timeout_count += 1

                    if 'episode_time_length' in infos[i]:
                        self.episode_time_lengths.append(infos[i]['episode_time_length'])
                        
                    self.total_episodes += 1
                
                self.current_episode_rewards[i] = 0.0
                self.current_episode_lengths[i] = 0

        # Log statistics to TensorBoard every 1000 steps
        if self.n_calls % 1000 == 0 and self.total_episodes > 0:
            # Calculate mean reward and episode length
            if len(self.episode_rewards) > 0:
                mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                mean_episode_length = sum(self.episode_lengths) / len(self.episode_lengths)
                
                # Initialize mean_episode_time_length with a default value
                mean_episode_time_length = 0.0
                if len(self.episode_time_lengths) > 0:
                    mean_episode_time_length = sum(self.episode_time_lengths) / len(self.episode_time_lengths)

                self.logger.record('metrics/mean_episode_reward', mean_reward)
                self.logger.record('metrics/mean_episode_length', mean_episode_length)
                self.logger.record('metrics/mean_episode_time_length', mean_episode_time_length)

            
            # Calculate and log success, collision, and timeout rates
            success_rate = self.success_count / self.total_episodes if self.total_episodes > 0 else 0
            collision_rate = self.collision_count / self.total_episodes if self.total_episodes > 0 else 0
            timeout_rate = self.timeout_count / self.total_episodes if self.total_episodes > 0 else 0
            
            self.logger.record('metrics/success_rate', success_rate)
            self.logger.record('metrics/collision_rate', collision_rate)
            self.logger.record('metrics/timeout_rate', timeout_rate)
            
            # Reset counters for the next logging interval
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_results = []
            self.episode_time_lengths = []

        return True
    
class GPUStatsCallback(BaseCallback):
    """
    Stampa e logga (TensorBoard) util% e memoria GPU a intervalli regolari.
    Funziona con on-policy (PPO) e off-policy (SAC/TD3/TQC).
    """
    def __init__(self, gpu_index: int = 0, min_interval_s: float = 10.0, print_to_stdout: bool = True):
        super().__init__(verbose=0)
        self.gpu_index = gpu_index
        self.min_interval_s = float(min_interval_s)
        self.print_to_stdout = print_to_stdout
        self._t_last = 0.0
        self._nvml = None

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml = (pynvml, handle)
        except Exception:
            self._nvml = None

    def _get_stats(self):
        # Prova NVML
        if self._nvml is None:
            self._init_nvml()
        if self._nvml is not None:
            pynvml, handle = self._nvml
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # %
                mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used, mem_total = int(mem.used // (1024**2)), int(mem.total // (1024**2))  # MiB
                try:
                    pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                except Exception:
                    pwr = None
                return util, mem_used, mem_total, pwr
            except Exception:
                self._nvml = None  # fallback a nvidia-smi

        # Fallback: nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i", str(self.gpu_index),
                ], stderr=subprocess.DEVNULL).decode().strip()
                util_s, mu_s, mt_s, pwr_s = [x.strip() for x in out.split(",")]
                util, mem_used, mem_total = int(util_s), int(mu_s), int(mt_s)
                pwr = float(pwr_s) if pwr_s not in ("", "N/A") else None
                return util, mem_used, mem_total, pwr
            except Exception:
                pass
        return None, None, None, None

    def _maybe_report(self):
        now = time.time()
        if now - self._t_last < self.min_interval_s:
            return
        self._t_last = now

        util, mem_used, mem_total, pwr = self._get_stats()
        # Statistiche PyTorch (se GPU attiva)
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                torch_alloc = torch.cuda.memory_allocated(dev) // (1024**2)
                torch_reserved = torch.cuda.memory_reserved(dev) // (1024**2)
            else:
                torch_alloc = torch_reserved = 0
        except Exception:
            torch_alloc = torch_reserved = 0

        # Log su TensorBoard
        if util is not None:
            self.model.logger.record("gpu/util_percent", float(util))
        if mem_used is not None and mem_total is not None:
            self.model.logger.record("gpu/memory_used_mib", float(mem_used))
            self.model.logger.record("gpu/memory_total_mib", float(mem_total))
        self.model.logger.record("gpu/torch_mem_alloc_mib", float(torch_alloc))
        self.model.logger.record("gpu/torch_mem_reserved_mib", float(torch_reserved))
        if pwr is not None:
            self.model.logger.record("gpu/power_w", float(pwr))

        # Print opzionale
        if self.print_to_stdout and util is not None and mem_used is not None:
            msg = (f"[GPU{self.gpu_index}] util={util:>3}%  mem={mem_used}/{mem_total} MiB  "
                   f"torch_alloc={torch_alloc} MiB  torch_reserved={torch_reserved} MiB")
            if pwr is not None:
                msg += f"  power={pwr:.0f} W"
            print(msg)

    # On-policy: stampa a fine rollout; Off-policy: throttling su _on_step
    def _on_rollout_end(self) -> bool:
        self._maybe_report()
        return True

    def _on_step(self) -> bool:
        self._maybe_report()
        return True
    









    
class GPUStatsCallback(BaseCallback):
    """
    Stampa e logga (TensorBoard) util% e memoria GPU a intervalli regolari.
    Funziona con on-policy (PPO) e off-policy (SAC/TD3/TQC).
    """
    def __init__(self, gpu_index: int = 0, min_interval_s: float = 10.0, print_to_stdout: bool = True):
        super().__init__(verbose=0)
        self.gpu_index = gpu_index
        self.min_interval_s = float(min_interval_s)
        self.print_to_stdout = print_to_stdout
        self._t_last = 0.0
        self._nvml = None

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml = (pynvml, handle)
        except Exception:
            self._nvml = None

    def _get_stats(self):
        # Prova NVML
        if self._nvml is None:
            self._init_nvml()
        if self._nvml is not None:
            pynvml, handle = self._nvml
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # %
                mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used, mem_total = int(mem.used // (1024**2)), int(mem.total // (1024**2))  # MiB
                try:
                    pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                except Exception:
                    pwr = None
                return util, mem_used, mem_total, pwr
            except Exception:
                self._nvml = None  # fallback a nvidia-smi

        # Fallback: nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i", str(self.gpu_index),
                ], stderr=subprocess.DEVNULL).decode().strip()
                util_s, mu_s, mt_s, pwr_s = [x.strip() for x in out.split(",")]
                util, mem_used, mem_total = int(util_s), int(mu_s), int(mt_s)
                pwr = float(pwr_s) if pwr_s not in ("", "N/A") else None
                return util, mem_used, mem_total, pwr
            except Exception:
                pass
        return None, None, None, None

    def _maybe_report(self):
        now = time.time()
        if now - self._t_last < self.min_interval_s:
            return
        self._t_last = now

        util, mem_used, mem_total, pwr = self._get_stats()
        # Statistiche PyTorch (se GPU attiva)
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                torch_alloc = torch.cuda.memory_allocated(dev) // (1024**2)
                torch_reserved = torch.cuda.memory_reserved(dev) // (1024**2)
            else:
                torch_alloc = torch_reserved = 0
        except Exception:
            torch_alloc = torch_reserved = 0

        # Log su TensorBoard
        if util is not None:
            self.model.logger.record("gpu/util_percent", float(util))
        if mem_used is not None and mem_total is not None:
            self.model.logger.record("gpu/memory_used_mib", float(mem_used))
            self.model.logger.record("gpu/memory_total_mib", float(mem_total))
        self.model.logger.record("gpu/torch_mem_alloc_mib", float(torch_alloc))
        self.model.logger.record("gpu/torch_mem_reserved_mib", float(torch_reserved))
        if pwr is not None:
            self.model.logger.record("gpu/power_w", float(pwr))

        # Print opzionale
        if self.print_to_stdout and util is not None and mem_used is not None:
            msg = (f"[GPU{self.gpu_index}] util={util:>3}%  mem={mem_used}/{mem_total} MiB  "
                   f"torch_alloc={torch_alloc} MiB  torch_reserved={torch_reserved} MiB")
            if pwr is not None:
                msg += f"  power={pwr:.0f} W"
            print(msg)

    # On-policy: stampa a fine rollout; Off-policy: throttling su _on_step
    def _on_rollout_end(self) -> bool:
        self._maybe_report()
        return True

    def _on_step(self) -> bool:
        self._maybe_report()
        return True