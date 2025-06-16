import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch

class CurriculumStage:
    """
    Represents a stage in the curriculum learning process.
    """
    def __init__(
        self,
        difficulty_level: int,
        n_humans: int,
        max_episode_time: float,
        success_threshold: float,
        min_success_rate: float = 0.7,
        min_episodes: int = 50
    ):
        self.difficulty_level = difficulty_level
        self.n_humans = n_humans
        self.max_episode_time = max_episode_time
        self.success_threshold = success_threshold
        self.min_success_rate = min_success_rate
        self.min_episodes = min_episodes
        self.episodes_completed = 0
        self.successes = 0
        
    def update(self, episode_result: Optional[str]) -> bool:
        """
        Update curriculum stage statistics and check if ready to advance.
        Returns True if ready to advance to next stage.
        """
        if episode_result is not None:
            self.episodes_completed += 1
            if episode_result == "success":
                self.successes += 1
                
        return (
            self.episodes_completed >= self.min_episodes and
            (self.successes / self.episodes_completed) >= self.min_success_rate
        )

class CurriculumManager:
    """
    Manages the curriculum learning process, including stage progression.
    """
    def __init__(self, env: gym.Env):
        self.env = env
        self.current_stage_idx = 0
        self.stages = self._create_default_stages()
        
    def _create_default_stages(self) -> List[CurriculumStage]:
        """Create default curriculum stages."""
        return [
            CurriculumStage(
                difficulty_level=1,
                n_humans=1,
                max_episode_time=50.0,
                success_threshold=0.5,
                min_success_rate=0.8,
                min_episodes=20
            ),
            CurriculumStage(
                difficulty_level=2,
                n_humans=2,
                max_episode_time=40.0,
                success_threshold=0.5,
                min_success_rate=0.75,
                min_episodes=30
            ),
            CurriculumStage(
                difficulty_level=3,
                n_humans=3,
                max_episode_time=35.0,
                success_threshold=0.5,
                min_success_rate=0.7,
                min_episodes=40
            ),
            CurriculumStage(
                difficulty_level=4,
                n_humans=4,
                max_episode_time=30.0,
                success_threshold=0.5,
                min_success_rate=0.7,
                min_episodes=50
            ),
            CurriculumStage(
                difficulty_level=5,
                n_humans=5,
                max_episode_time=25.0,
                success_threshold=0.5,
                min_success_rate=0.65,
                min_episodes=60
            )
        ]
        
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def update(self, episode_result: Optional[str]) -> bool:
        """
        Update curriculum progress and check if should advance to next stage.
        Returns True if curriculum advanced to next stage.
        """
        current_stage = self.get_current_stage()
        ready_to_advance = current_stage.update(episode_result)
        
        if ready_to_advance and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self._apply_stage_parameters()
            return True
        return False
    
    def _apply_stage_parameters(self):
        """Apply the current stage's parameters to the environment."""
        stage = self.get_current_stage()
        if hasattr(self.env, 'envs'):  # Handle vectorized environments
            for env in self.env.envs:
                if hasattr(env, 'n_humans'):
                    env.n_humans = stage.n_humans
                if hasattr(env, 'max_episode_time'):
                    env.max_episode_time = stage.max_episode_time
                if hasattr(env, 'distance_success_threshold'):
                    env.distance_success_threshold = stage.success_threshold
        else:  # Handle single environment
            if hasattr(self.env, 'n_humans'):
                self.env.n_humans = stage.n_humans
            if hasattr(self.env, 'max_episode_time'):
                self.env.max_episode_time = stage.max_episode_time
            if hasattr(self.env, 'distance_success_threshold'):
                self.env.distance_success_threshold = stage.success_threshold

class CurriculumCallback(BaseCallback):
    """
    Callback for handling curriculum learning updates.
    """
    def __init__(self, curriculum_manager: CurriculumManager, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.last_logged_stage = -1
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        """Check if we should advance to the next curriculum stage."""
        # Get episode results from the environment
        episode_results = []
        if hasattr(self.model.env, 'get_attr'):
            episode_results = self.model.env.get_attr('last_episode_result')
        elif hasattr(self.model.env, 'last_episode_result'):
            episode_results = [self.model.env.last_episode_result]
            
        # Update curriculum for each episode result
        for result in episode_results:
            advanced = self.curriculum_manager.update(result)
            if advanced:
                current_stage = self.curriculum_manager.get_current_stage()
                self.logger.record('curriculum/stage', current_stage.difficulty_level)
                self.logger.record('curriculum/n_humans', current_stage.n_humans)
                self.logger.record('curriculum/max_time', current_stage.max_episode_time)
                
                if self.verbose >= 1:
                    print(f"\nAdvanced to curriculum stage {current_stage.difficulty_level}")
                    print(f"Humans: {current_stage.n_humans}")
                    print(f"Max episode time: {current_stage.max_episode_time}\n")
        
        # Log current stage if it changed
        current_stage = self.curriculum_manager.get_current_stage()
        if current_stage.difficulty_level != self.last_logged_stage:
            self.last_logged_stage = current_stage.difficulty_level
            self.logger.record('curriculum/stage', current_stage.difficulty_level)
            self.logger.record('curriculum/n_humans', current_stage.n_humans)
            self.logger.record('curriculum/max_time', current_stage.max_episode_time)