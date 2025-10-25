import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional
import sokoban_engine as soko
from sokoban_levels import SokobanLevelCollection
import random

global action_tracker
action_tracker = []

class OptimizedSokobanEnv(gym.Env):
    """
    Multi-level Sokoban environment with random level selection.
    
    Features:
    - 40+ levels of varying difficulty
    - Random level selection every episode
    - Curriculum-based level filtering
    - Balanced reward shaping
    """
    
    def __init__(self, 
                 max_episode_steps: int = 200,
                 curriculum_mode: bool = True,
                 difficulty_level: int = 1,
                 anti_hacking_strength: float = 0.5,
                 level_selection_mode: str = 'random',  # 'random', 'sequential', 'curriculum'
                 custom_levels: Optional[list] = None):
        
        super().__init__()
        global action_tracker
        
        # Constants from sokoban engine
        self.WALL = soko.WALL
        self.EMPTY = soko.EMPTY
        self.PLAYER = soko.PLAYER
        self.BOX = soko.BOX
        self.TARGET = soko.TARGET
        self.BOX_ON_TARGET = soko.BOX_ON_TARGET
        self.PLAYER_ON_TARGET = soko.PLAYER_ON_TARGET
        
        self.max_episode_steps = max_episode_steps
        self.curriculum_mode = curriculum_mode
        self.difficulty_level = difficulty_level
        self.anti_hacking_strength = anti_hacking_strength
        self.level_selection_mode = level_selection_mode
        
        # Level management
        self.level_collection = SokobanLevelCollection()
        self.custom_levels = custom_levels
        self.current_level_index = 0
        self.current_level_string = None
        
        # Statistics for each level
        self.level_stats = {}  # {level_index: {'attempts': 0, 'solves': 0}}
        
        # Game engine
        self.game = soko.Sokoban()
        
        # Episode counter
        self.episode_count = 0
        
        # State tracking for advanced rewards
        self.reset_tracking_vars()
        
        # Observation and action spaces (use maximum possible size)
        self.observation_space = gym.spaces.Box(
            low=0, high=6, shape=(200,), dtype=np.int32  # Increased for larger levels
        )
        self.action_space = gym.spaces.Discrete(4)
        
    def reset_tracking_vars(self):
        """Reset all tracking variables for reward computation"""
        global action_tracker
        self.step_count = 0
        self.boxes_on_targets_history = []
        self.last_boxes_on_targets = 0
        self.target_stability_bonus = 0
        self.push_count = 0
        self.useless_push_count = 0
        self.last_box_positions = set()
        self.box_oscillation_penalty = 0
        self.solved_step = None
        self.last_distance = None
    
    def get_available_levels(self) -> list:
        """Get list of levels based on current mode and difficulty"""
        if self.custom_levels:
            return self.custom_levels
        
        if self.level_selection_mode == 'curriculum' and self.curriculum_mode:
            return self.level_collection.get_curriculum_levels(self.difficulty_level)
        else:
            return self.level_collection.get_all_levels()
    
    def select_next_level(self):
        """Select the next level based on selection mode"""
        available_levels = self.get_available_levels()
        
        if self.level_selection_mode == 'random':
            # Random selection
            self.current_level_index = random.randint(0, len(available_levels) - 1)
        elif self.level_selection_mode == 'sequential':
            # Sequential with wraparound
            self.current_level_index = (self.current_level_index + 1) % len(available_levels)
        elif self.level_selection_mode == 'curriculum':
            # Weighted random based on performance
            self.current_level_index = self._select_curriculum_level(available_levels)
        else:
            self.current_level_index = 0
        
        self.current_level_string = available_levels[self.current_level_index]
        return self.current_level_string
    
    def _select_curriculum_level(self, available_levels):
        """Select level with curriculum strategy (focus on unsolved)"""
        # Build weights based on solve rate
        weights = []
        for i in range(len(available_levels)):
            stats = self.level_stats.get(i, {'attempts': 0, 'solves': 0})
            attempts = stats['attempts']
            solves = stats['solves']
            
            if attempts == 0:
                # Unseen level - high priority
                weight = 2.0
            else:
                solve_rate = solves / attempts
                # Lower solve rate = higher weight (focus on difficult levels)
                weight = max(0.1, 1.0 - solve_rate)
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted random choice
        return np.random.choice(len(available_levels), p=weights)
    
    def set_specific_level(self, level_index_or_string):
        """Set a specific level by index or string"""
        if isinstance(level_index_or_string, str):
            self.current_level_string = level_index_or_string
            self.current_level_index = -1  # Custom level
        else:
            available_levels = self.get_available_levels()
            self.current_level_index = level_index_or_string % len(available_levels)
            self.current_level_string = available_levels[self.current_level_index]
    
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
        global action_tracker
        super().reset(seed=seed)
        
        # Reset tracking
        self.reset_tracking_vars()
        
        # Increment episode counter
        self.episode_count += 1
        
        # Select and load new level
        level_string = self.select_next_level()
        self.game.load_level(level_string)
        
        # Initialize distance tracking
        self.last_distance = self.total_box_target_distance()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        info['level_index'] = self.current_level_index
        info['level_string'] = self.current_level_string
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        global action_tracker
        # Convert action
        action_map = {
            0: soko.UP,
            1: soko.DOWN, 
            2: soko.LEFT,
            3: soko.RIGHT
        }
        
        action_tracker.append(action)
        
        # Store state before action
        prev_boxes_on_targets = self.boxes_on_targets()
        prev_box_positions = set(self.get_box_positions())
        prev_distance = self.last_distance
        
        # Execute action
        obs_data, base_reward, done, pushed_box = self.game.step(action_map[action])
        
        # Update tracking
        self.step_count += 1
        current_boxes_on_targets = self.boxes_on_targets()
        current_box_positions = set(self.get_box_positions())
        current_distance = self.total_box_target_distance()
        self.last_distance = current_distance
        
        # Track box movement patterns
        if pushed_box:
            self.push_count += 1
            
            if current_box_positions == self.last_box_positions:
                self.useless_push_count += 1
                self.box_oscillation_penalty += 0.1 * self.anti_hacking_strength
            
            self.last_box_positions = prev_box_positions.copy()
        
        # Track target stability
        self.boxes_on_targets_history.append(current_boxes_on_targets)
        if len(self.boxes_on_targets_history) > 10:
            self.boxes_on_targets_history.pop(0)
            
        if len(self.boxes_on_targets_history) >= 5:
            recent_avg = np.mean(self.boxes_on_targets_history[-5:])
            if recent_avg > 0.8 * self.num_targets():
                self.target_stability_bonus += 0.02
        
        # Compute reward
        reward_components = self._compute_advanced_reward(
            base_reward, prev_boxes_on_targets, current_boxes_on_targets, 
            pushed_box, done, prev_distance, current_distance
        )
        total_reward = sum(reward_components.values())
        
        # Check termination
        truncated = self.step_count >= self.max_episode_steps
        if done and self.solved_step is None:
            self.solved_step = self.step_count
        
        # Update level statistics
        if done or truncated:
            if self.current_level_index >= 0:
                if self.current_level_index not in self.level_stats:
                    self.level_stats[self.current_level_index] = {'attempts': 0, 'solves': 0}
                
                self.level_stats[self.current_level_index]['attempts'] += 1
                if done:
                    self.level_stats[self.current_level_index]['solves'] += 1
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info(reward_components)
        info['level_index'] = self.current_level_index
        
        return obs, total_reward, done, truncated, info, action_tracker
    
    def _compute_advanced_reward(self, base_reward: float, prev_targets: int, 
                                curr_targets: int, pushed_box: bool, solved: bool,
                                prev_distance: float, curr_distance: float) -> Dict[str, float]:
        """Balanced reward structure"""
        components = {
            'base': base_reward,
            'progress': 0.0,
            'distance': 0.0,
            'stability': 0.0,
            'efficiency': 0.0,
            'anti_hack': 0.0,
            'step': -0.01
        }
        
        # Progress rewards
        target_change = curr_targets - prev_targets
        if target_change > 0:
            components['progress'] = 1.0 * target_change
        elif target_change < 0:
            components['progress'] = -1.0 * abs(target_change)
        
        # Stability bonus
        if curr_targets > 0:
            components['stability'] = self.target_stability_bonus
        
        # Distance-based shaping
        if pushed_box and target_change == 0 and prev_distance is not None:
            distance_improvement = prev_distance - curr_distance
            max_distance = (self.level_width() + self.level_height()) * self.num_targets()
            
            if max_distance > 0:
                normalized_improvement = distance_improvement / max_distance
                components['distance'] = 0.1 * normalized_improvement
        
        # Efficiency rewards
        if solved:
            num_boxes = self.num_targets()
            optimal_steps = num_boxes * 10  # Rough estimate
            efficiency_bonus = max(0.5, optimal_steps / max(self.step_count, 1))
            components['efficiency'] = 10.0 * efficiency_bonus
        
        # Anti-hacking penalties
        if self.box_oscillation_penalty > 0:
            components['anti_hack'] = -self.box_oscillation_penalty
            
        if self.push_count > 30:
            useless_ratio = self.useless_push_count / self.push_count
            if useless_ratio > 0.5:
                components['anti_hack'] -= 0.1 * useless_ratio * self.anti_hacking_strength
        
        return components
    
    def _get_observation(self) -> np.ndarray:
        obs = self.game.get_observation()
        obs_array = np.array(obs, dtype=np.int32)
        
        # Pad to 200 elements (supports larger levels)
        if len(obs_array) < 200:
            padded = np.zeros(200, dtype=np.int32)
            padded[:len(obs_array)] = obs_array
            return padded
        else:
            return obs_array[:200]
    
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
            'solved_step': self.solved_step,
            'level_width': self.level_width(),
            'level_height': self.level_height()
        }
    
    def set_difficulty(self, level: int):
        """Set curriculum difficulty level"""
        self.difficulty_level = min(4, max(1, level))
        print(f"🎯 Difficulty set to level {self.difficulty_level}")
    
    def get_level_statistics(self) -> Dict:
        """Get statistics for all attempted levels"""
        return self.level_stats.copy()
    
    def print_level_statistics(self):
        """Print formatted level statistics"""
        if not self.level_stats:
            print("No level statistics yet")
            return
        
        print("\n📊 Level Statistics:")
        print("="*60)
        for idx in sorted(self.level_stats.keys()):
            stats = self.level_stats[idx]
            attempts = stats['attempts']
            solves = stats['solves']
            rate = (solves / attempts * 100) if attempts > 0 else 0
            print(f"  Level {idx:3d}: {solves:3d}/{attempts:3d} solved ({rate:5.1f}%)")
        print("="*60)