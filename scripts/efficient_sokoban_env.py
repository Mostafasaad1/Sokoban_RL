"""
EFFICIENT Sokoban Environment - No Padding Waste!
=================================================

This version eliminates the inefficient padding that was adding 51 zeros
to each observation. Now we use the true 7x7 grid (49 elements) directly,
making the neural network much more efficient.

CHANGES FROM OPTIMIZED VERSION:
- Observation space: 49 elements (7x7 grid) instead of 100 (padded)
- Network input layer: 49 instead of 100 neurons
- Much more efficient training and inference
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import sokoban_engine as soko

class EfficientSokobanEnv(gym.Env):
    """
    EFFICIENT Sokoban environment with no padding waste.
    Uses true 7x7 grid (49 elements) directly.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Grid cell constants
    WALL = 0
    EMPTY = 1
    BOX = 2
    TARGET = 3
    PLAYER = 4
    BOX_ON_TARGET = 5
    PLAYER_ON_TARGET = 6
    
    def __init__(self, curriculum_mode: bool = True, difficulty_level: int = 1, anti_hacking_strength: float = 1.0):
        super().__init__()
        
        # Initialize C++ engine
        self.game = soko.Sokoban()
        
        # Environment parameters
        self.curriculum_mode = curriculum_mode
        self.difficulty_level = difficulty_level
        self.anti_hacking_strength = anti_hacking_strength
        
        # TRUE 7x7 grid - no padding!
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(49,), dtype=np.int32  # 7*7 = 49
        )
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        
        # Curriculum levels (same as optimized version)
        self.curriculum_levels = {
            1: ["simple_levels_easy"],    # Basic pushing
            2: ["simple_levels_medium"],  # Multi-box coordination
            3: ["simple_levels_hard"]     # Complex puzzles
        }
        
        # Tracking variables
        self.episode_count = 0
        self.reset_tracking_vars()
        
        print("🚀 EFFICIENT SOKOBAN ENV INITIALIZED")
        print(f"📐 Observation space: {self.observation_space.shape} (TRUE 7x7 grid)")
        print(f"🎮 Action space: {self.action_space.n}")
        print(f"⚡ NO PADDING WASTE - Direct 49-element observations!")
        
    def reset_tracking_vars(self):
        """Reset episode-specific tracking variables"""
        self.step_count = 0
        self.total_reward = 0.0
        self.target_stability_bonus = 0
        self.push_count = 0
        self.useless_push_count = 0
        self.last_box_positions = set()
        self.box_oscillation_penalty = 0
        self.solved_step = None
        
    def get_current_level(self) -> str:
        """Get level based on curriculum progression"""
        if self.curriculum_mode:
            level_set = self.curriculum_levels[self.difficulty_level]
            # Cycle through levels of current difficulty
            return level_set[self.episode_count % len(level_set)]
        else:
            # Use default medium difficulty
            return self.curriculum_levels[2][0]
    
    def boxes_on_targets(self) -> int:
        """Count boxes currently on target positions"""
        grid = self.game.get_grid()
        count = 0
        for row in grid:
            for cell in row:
                if cell == self.BOX_ON_TARGET:
                    count += 1
        return count
    
    def num_targets(self) -> int:
        """Count total number of target positions"""
        grid = self.game.get_grid()
        count = 0
        for row in grid:
            for cell in row:
                if cell == self.TARGET or cell == self.BOX_ON_TARGET or cell == self.PLAYER_ON_TARGET:
                    count += 1
        return count
    
    def get_box_positions(self) -> list:
        """Get positions of all boxes"""
        grid = self.game.get_grid()
        positions = []
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == self.BOX or cell == self.BOX_ON_TARGET:
                    positions.append((x, y))
        return positions
    
    def get_target_positions(self) -> list:
        """Get positions of all targets"""
        grid = self.game.get_grid()
        positions = []
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == self.TARGET or cell == self.BOX_ON_TARGET or cell == self.PLAYER_ON_TARGET:
                    positions.append((x, y))
        return positions
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset tracking
        self.reset_tracking_vars()
        
        # Increment episode counter for curriculum
        self.episode_count += 1
        
        # Reset to built-in level
        self.game.reset()
        
        # Get initial observation - NO PADDING!
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Convert action
        action_map = {
            0: soko.UP,
            1: soko.DOWN, 
            2: soko.LEFT,
            3: soko.RIGHT
        }
        
        # Store state before action
        prev_boxes_on_targets = self.boxes_on_targets()
        prev_box_positions = set(self.get_box_positions())
        
        # Execute action
        obs_data, base_reward, done, pushed_box = self.game.step(action_map[action])
        
        # Update tracking
        self.step_count += 1
        current_boxes_on_targets = self.boxes_on_targets()
        current_box_positions = set(self.get_box_positions())
        
        # Track box movement patterns for anti-hacking
        if pushed_box:
            self.push_count += 1
            
            # Detect useless pushes (box returns to previous position)
            if current_box_positions == self.last_box_positions:
                self.useless_push_count += 1
                self.box_oscillation_penalty += 0.1 * self.anti_hacking_strength
        
        self.last_box_positions = prev_box_positions
        
        # Advanced reward calculation
        reward_components = {}
        
        # Base reward from game engine
        reward = base_reward
        reward_components['base'] = base_reward
        
        # Target progression bonus (primary signal)
        target_progress = current_boxes_on_targets - prev_boxes_on_targets
        if target_progress > 0:
            target_bonus = target_progress * 5.0
            reward += target_bonus
            reward_components['target_progress'] = target_bonus
        elif target_progress < 0:
            target_penalty = target_progress * 2.0  # Penalty for losing progress
            reward += target_penalty
            reward_components['target_regression'] = target_penalty
        
        # Stability bonus for maintaining boxes on targets
        stability_bonus = current_boxes_on_targets * 0.1
        self.target_stability_bonus += stability_bonus
        reward += stability_bonus
        reward_components['stability'] = stability_bonus
        
        # Efficiency incentives
        reward += 0.01  # Small step bonus to encourage exploration
        reward_components['step_bonus'] = 0.01
        
        # Anti-hacking measures
        if self.box_oscillation_penalty > 0:
            reward -= self.box_oscillation_penalty
            reward_components['anti_oscillation'] = -self.box_oscillation_penalty
            self.box_oscillation_penalty = 0  # Reset after applying
        
        # Completion bonus
        if done and self.game.is_solved():
            completion_bonus = 100.0 + max(0, 200 - self.step_count) * 0.5  # Efficiency bonus
            reward += completion_bonus
            reward_components['completion'] = completion_bonus
            self.solved_step = self.step_count
            print(f"🎉 Level completed in {self.step_count} steps! Bonus: +{completion_bonus:.1f}")
        
        # Step limit penalty
        if self.step_count >= 200:
            done = True
            if not self.game.is_solved():
                timeout_penalty = -10.0
                reward += timeout_penalty
                reward_components['timeout'] = timeout_penalty
        
        self.total_reward += reward
        
        # Get new observation - NO PADDING!
        obs = self._get_observation()
        info = self._get_info(reward_components)
        
        return obs, reward, done, False, info  # False for truncated
    
    def _get_observation(self) -> np.ndarray:
        """
        EFFICIENT OBSERVATION - NO PADDING!
        Returns the true 7x7 grid (49 elements) directly.
        """
        obs = self.game.get_observation()
        # Return TRUE 49-element observation - no padding waste!
        obs_array = np.array(obs, dtype=np.int32)
        
        # Verify it's exactly 49 elements (7x7 grid)
        if len(obs_array) != 49:
            print(f"⚠️ WARNING: Expected 49 elements but got {len(obs_array)}")
            # Fallback to ensure correct size
            if len(obs_array) < 49:
                padded = np.zeros(49, dtype=np.int32)
                padded[:len(obs_array)] = obs_array
                return padded
            else:
                return obs_array[:49]
        
        return obs_array
    
    def _get_info(self, reward_components: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return {
            'is_solved': self.game.is_solved(),
            'boxes_on_targets': self.boxes_on_targets(),
            'total_targets': self.num_targets(),
            'step_count': self.step_count,
            'push_count': self.push_count,
            'useless_push_ratio': self.useless_push_count / max(1, self.push_count),
            'reward_components': reward_components or {},
            'difficulty_level': self.difficulty_level,
            'solved_step': self.solved_step
        }
    
    def set_difficulty(self, level: int):
        """Set curriculum difficulty level"""
        temp = self.difficulty_level
        self.difficulty_level = min(3, max(1, level))
        changed_Difficulty = (temp != self.difficulty_level)
        if changed_Difficulty:
            print(f"🎯 Difficulty set to level {self.difficulty_level}")
    
    def get_success_rate(self) -> float:
        """Get current success rate (for curriculum progression)"""
        # This would be tracked externally in the training loop
        return 0.0
    
    def render_grid(self) -> str:
        """
        Render the TRUE 7x7 grid as ASCII art.
        This shows the actual game state without padding.
        """
        grid = self.game.get_grid()
        symbols = {
            self.WALL: '██',
            self.EMPTY: '  ',
            self.BOX: '📦',
            self.TARGET: '🎯',
            self.PLAYER: '🚀',
            self.BOX_ON_TARGET: '✅',
            self.PLAYER_ON_TARGET: '🎯'
        }
        
        result = "\n🎮 EFFICIENT SOKOBAN (TRUE 7x7 GRID)\n"
        result += "┌" + "──" * 7 + "┐\n"
        
        for row in grid:
            result += "│"
            for cell in row:
                result += symbols.get(cell, '??')
            result += "│\n"
        
        result += "└" + "──" * 7 + "┘\n"
        result += f"📦 Boxes on targets: {self.boxes_on_targets()}/{self.num_targets()}\n"
        result += f"👟 Steps: {self.step_count}\n"
        result += f"📋 Pushes: {self.push_count}\n"
        
        return result

# Test the efficient environment
if __name__ == "__main__":
    print("🧪 TESTING EFFICIENT SOKOBAN ENVIRONMENT")
    print("="*50)
    
    env = EfficientSokobanEnv()
    obs, info = env.reset()
    
    print(f"📐 Observation shape: {obs.shape}")
    print(f"📊 Observation: {obs}")
    print(f"📝 Info: {info}")
    
    print("\n🎮 Initial Grid:")
    print(env.render_grid())
    
    # Test a few random actions
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"\n🎯 Action {i+1}: {action}, Reward: {reward:.3f}")
        if done:
            print("🏁 Episode finished!")
            break
    
    print("\n✅ EFFICIENT ENVIRONMENT TEST COMPLETE!")