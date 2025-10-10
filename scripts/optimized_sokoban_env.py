import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional
import sokoban_engine as soko

class OptimizedSokobanEnv(gym.Env):
    """
    Highly optimized Sokoban environment with advanced reward shaping,
    curriculum learning, and anti-hacking mechanisms.
    
    Compatible with original environment structure - uses default built-in level.
    """
    
    def __init__(self, 
                 max_episode_steps: int = 400,  # Longer episodes for complex puzzles
                 curriculum_mode: bool = False,  # Disabled for now - use default level
                 difficulty_level: int = 1,  # 1=easy, 2=medium, 3=hard
                 anti_hacking_strength: float = 2.0):
        super().__init__()
        
        # Constants from sokoban engine
        self.WALL = soko.WALL  # 0
        self.EMPTY = soko.EMPTY  # 1
        self.PLAYER = soko.PLAYER  # 2
        self.BOX = soko.BOX  # 3
        self.TARGET = soko.TARGET  # 4
        self.BOX_ON_TARGET = soko.BOX_ON_TARGET  # 5
        self.PLAYER_ON_TARGET = soko.PLAYER_ON_TARGET  # 6
        
        self.max_episode_steps = max_episode_steps
        self.curriculum_mode = curriculum_mode
        self.difficulty_level = difficulty_level
        self.anti_hacking_strength = anti_hacking_strength
        
        # Game engine
        self.game = soko.Sokoban()
        
        # Episode counter for future curriculum support
        self.episode_count = 0
        
        # State tracking for advanced rewards
        self.reset_tracking_vars()
        
        # Observation and action spaces (match original environment)
        self.observation_space = gym.spaces.Box(
            low=0, high=6, shape=(100,), dtype=np.int32  # Flat 100-element like original
        )
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        
    def reset_tracking_vars(self):
        """Reset all tracking variables for reward computation"""
        self.step_count = 0
        self.boxes_on_targets_history = []
        self.last_boxes_on_targets = 0
        self.target_stability_bonus = 0
        self.push_count = 0
        self.useless_push_count = 0
        self.last_box_positions = set()
        self.box_oscillation_penalty = 0
        self.solved_step = None
    
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
    
    def total_box_target_distance(self) -> float:
        """Calculate total Manhattan distance from boxes to nearest targets"""
        box_positions = self.get_box_positions()
        target_positions = self.get_target_positions()
        
        if not box_positions or not target_positions:
            return 0.0
        
        total_distance = 0.0
        for box_pos in box_positions:
            min_distance = min(
                abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                for target_pos in target_positions
            )
            total_distance += min_distance
        
        return total_distance
    
    def level_width(self) -> int:
        """Get level width"""
        return self.game.get_width()
    
    def level_height(self) -> int:
        """Get level height"""
        return self.game.get_height()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset tracking
        self.reset_tracking_vars()
        
        # Increment episode counter for future curriculum support
        self.episode_count += 1
        
        # Use the default built-in level like the original environment
        self.game.reset()  # Uses built-in default level
        
        # Get initial observation
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
            
            self.last_box_positions = prev_box_positions.copy()
        
        # Track target stability
        self.boxes_on_targets_history.append(current_boxes_on_targets)
        if len(self.boxes_on_targets_history) > 10:
            self.boxes_on_targets_history.pop(0)
            
        # Calculate stability bonus (boxes staying on targets)
        if len(self.boxes_on_targets_history) >= 5:
            recent_avg = np.mean(self.boxes_on_targets_history[-5:])
            if recent_avg > 0.8 * self.num_targets():  # Most targets occupied consistently
                self.target_stability_bonus += 0.02
        
        # Compute advanced reward
        reward_components = self._compute_advanced_reward(
            base_reward, prev_boxes_on_targets, current_boxes_on_targets, 
            pushed_box, done
        )
        total_reward = sum(reward_components.values())
        
        # Check termination conditions
        truncated = self.step_count >= self.max_episode_steps
        if done and self.solved_step is None:
            self.solved_step = self.step_count
        
        # Get new observation and info
        obs = self._get_observation()
        info = self._get_info(reward_components)
        
        return obs, total_reward, done, truncated, info
    
    def _compute_advanced_reward(self, base_reward: float, prev_targets: int, 
                                curr_targets: int, pushed_box: bool, solved: bool) -> Dict[str, float]:
        """Compute sophisticated reward with anti-hacking mechanisms"""
        components = {
            'base': base_reward,
            'progress': 0.0,
            'stability': 0.0,
            'efficiency': 0.0,
            'exploration': 0.0,
            'anti_hack': 0.0,
            'step': -0.002  # Slightly higher step penalty for efficiency
        }
        
        # 1. PROGRESS REWARDS (primary signal)
        target_change = curr_targets - prev_targets
        if target_change > 0:
            # Box placed on target - BIG reward
            components['progress'] = 0.5 * target_change
            print(f"📦✅ Box(es) placed on target! Progress reward: {components['progress']:.3f}")
        elif target_change < 0:
            # Box removed from target - MASSIVE penalty
            components['progress'] = -1.0 * abs(target_change) * self.anti_hacking_strength
            print(f"📦❌ Box(es) removed from target! Penalty: {components['progress']:.3f}")
        
        # 2. STABILITY BONUS (keeping boxes on targets)
        if curr_targets > 0:
            components['stability'] = self.target_stability_bonus
        
        # 3. EFFICIENCY REWARDS
        if solved:
            # Huge completion bonus, scaled by efficiency
            efficiency_bonus = max(1.0, 2.0 - (self.step_count / 100))
            components['efficiency'] = 10.0 * efficiency_bonus
            print(f"🎉 PUZZLE SOLVED in {self.step_count} steps! Efficiency bonus: {components['efficiency']:.2f}")
        
        # 4. SMART EXPLORATION (distance-based, but limited)
        if pushed_box and curr_targets == prev_targets:  # Only for non-progress moves
            total_distance = self.total_box_target_distance()
            max_distance = self.level_width() + self.level_height()
            normalized_distance = total_distance / max_distance if max_distance > 0 else 0
            components['exploration'] = -0.005 * normalized_distance  # Very small exploration signal
        
        # 5. ANTI-HACKING PENALTIES
        # Penalize box oscillation (moving boxes back and forth)
        if self.box_oscillation_penalty > 0:
            components['anti_hack'] = -self.box_oscillation_penalty
            
        # Penalize excessive pushing without progress
        if self.push_count > 20:  # After some initial pushes
            useless_ratio = self.useless_push_count / self.push_count
            if useless_ratio > 0.3:  # More than 30% useless pushes
                components['anti_hack'] -= 0.1 * useless_ratio * self.anti_hacking_strength
        
        # Reduce push reward to almost nothing to discourage farming
        if pushed_box and curr_targets == prev_targets:
            components['exploration'] = min(components['exploration'], 0.005)  # Tiny push reward cap
        
        return components
    
    def _get_observation(self) -> np.ndarray:
        obs = self.game.get_observation()
        # Return flat observation like original environment, padded to 100 elements
        obs_array = np.array(obs, dtype=np.int32)
        
        # Pad to 100 elements if shorter
        if len(obs_array) < 100:
            padded = np.zeros(100, dtype=np.int32)
            padded[:len(obs_array)] = obs_array
            return padded
        else:
            return obs_array[:100]  # Truncate if longer
    
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
        """Set curriculum difficulty level (for future use)"""
        self.difficulty_level = min(3, max(1, level))
        print(f"🎯 Difficulty set to level {self.difficulty_level}")
    
    def get_success_rate(self) -> float:
        """Get current success rate (for curriculum progression)"""
        # This would be tracked externally in the training loop
        return 0.0