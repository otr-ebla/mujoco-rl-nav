from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os




class ScenarioSuccessCallback(BaseCallback):
    """
    Custom callback to track success rates for each scenario during training.
    """
    def __init__(self, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.scenario_stats = {}  # Will store stats for each scenario
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if any environment has finished an episode
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        if len(dones) > 0 and len(infos) > 0:
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and info:
                    self._log_episode_result(info)
        
        # Log summary statistics periodically
        if self.n_calls % self.log_freq == 0:
            self._log_scenario_summary()
            
        return True
    
    def _log_episode_result(self, info):
        """Log the result of a completed episode"""
        scenario_id = info.get('scenario_id', 'unknown')
        episode_result = info.get('episode_result', 'unknown')
        
        # Initialize scenario stats if not seen before
        if scenario_id not in self.scenario_stats:
            self.scenario_stats[scenario_id] = {
                'total_episodes': 0,
                'successes': 0,
                'human_collisions': 0,
                'obstacle_collisions': 0,
                'timeouts': 0
            }
        
        # Update stats
        stats = self.scenario_stats[scenario_id]
        stats['total_episodes'] += 1
        
        if episode_result == 'success':
            stats['successes'] += 1
        elif episode_result == 'human_collision':
            stats['human_collisions'] += 1
        elif episode_result == 'collision':
            stats['obstacle_collisions'] += 1
        elif episode_result == 'timeout':
            stats['timeouts'] += 1
            
        self.episode_count += 1
    
    def _log_scenario_summary(self):
        """Log summary statistics for all scenarios"""
        if not self.scenario_stats:
            return
            
        print(f"\n=== Scenario Success Rates (Step {self.num_timesteps}) ===")
        
        total_episodes_all = 0
        total_successes_all = 0
        
        for scenario_id, stats in sorted(self.scenario_stats.items()):
            if stats['total_episodes'] > 0:
                success_rate = stats['successes'] / stats['total_episodes']
                humans_collision_rate = stats['human_collisions'] / stats['total_episodes']
                collision_rate = stats['obstacle_collisions'] / stats['total_episodes']
                timeout_rate = stats['timeouts'] / stats['total_episodes']
                
                print(f"Scenario {scenario_id}: "
                      f"Episodes={stats['total_episodes']}, "
                      f"Success={success_rate:.3f}, "
                      f"Human Collisions={humans_collision_rate:.3f}, "
                      f"Obstacle Collision={collision_rate:.3f}, "
                      f"Timeout={timeout_rate:.3f}")
                
                # Log to tensorboard if available
                if hasattr(self, 'logger') and self.logger:
                    self.logger.record(f"scenario_{scenario_id}/success_rate", success_rate)
                    self.logger.record(f"scenario_{scenario_id}/collision_rate", collision_rate)
                    self.logger.record(f"scenario_{scenario_id}/timeout_rate", timeout_rate)
                    self.logger.record(f"scenario_{scenario_id}/total_episodes", stats['total_episodes'])
                
                total_episodes_all += stats['total_episodes']
                total_successes_all += stats['successes']
        
        # Overall success rate
        if total_episodes_all > 0:
            overall_success_rate = total_successes_all / total_episodes_all
            print(f"Overall Success Rate: {overall_success_rate:.3f} ({total_successes_all}/{total_episodes_all})")
            
            if getattr(self, 'logger', None):
                self.logger.record("training/overall_success_rate", overall_success_rate)

        # ALWAYS flush any recorded metrics
        if getattr(self, 'logger', None):
            self.logger.dump(self.num_timesteps)
        
        print("=" * 50)

class PolicySaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"policy_step_{self.num_timesteps}.zip")
            self.model.save(save_file)
            if self.verbose:
                print(f"✅ Saved model at step {self.num_timesteps} to {save_file}")
        return True
    
class RenderCallback(BaseCallback):
    def _on_step(self):
        try:
            env = self.model.get_env()
            base = env
            # un-wrap VecNormalize/VecEnvWrapper
            while hasattr(base, "venv"):
                base = base.venv
            # ora base è DummyVecEnv
            base.envs[0].render()
        except Exception as e:
            print("Render error:", e)
        return True
