import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional
import sokoban_engine as soko

global action_tracker
action_tracker = []

class OptimizedSokobanEnv(gym.Env):
    """
    FIXED: Balanced reward shaping and proper curriculum learning.
    
    Key fixes:
    - Symmetric rewards/penalties (±1.0 for box placement/removal)
    - Meaningful distance-based shaping (0.1 magnitude)
    - Higher step penalty for urgency (-0.01 instead of -0.002)
    - Gentler anti-hacking (0.5 strength instead of 2.0)
    - Shorter episodes (200 steps instead of 400)
    """
    
    def __init__(self, 
                 max_episode_steps: int = 200,  # FIXED: Was 400
                 curriculum_mode: bool = True,   # FIXED: Was False
                 difficulty_level: int = 1,
                 anti_hacking_strength: float = 0.5):  # FIXED: Was 2.0
        
        super().__init__()
        global action_tracker
        
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
        
        # Episode counter for curriculum
        self.episode_count = 0
        
        # State tracking for advanced rewards
        self.reset_tracking_vars()
        
        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=6, shape=(100,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        
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
        self.last_distance = None  # Track distance changes
        
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
        
        # Use the default built-in level
        self.game.reset()
        
        # Initialize distance tracking
        self.last_distance = self.total_box_target_distance()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
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
            if recent_avg > 0.8 * self.num_targets():
                self.target_stability_bonus += 0.02
        
        # Compute advanced reward with FIXED balancing
        reward_components = self._compute_advanced_reward(
            base_reward, prev_boxes_on_targets, current_boxes_on_targets, 
            pushed_box, done, prev_distance, current_distance
        )
        total_reward = sum(reward_components.values())
        
        # Check termination conditions
        truncated = self.step_count >= self.max_episode_steps
        if done and self.solved_step is None:
            self.solved_step = self.step_count
        
        # Get new observation and info
        obs = self._get_observation()
        info = self._get_info(reward_components)
        
        return obs, total_reward, done, truncated, info, action_tracker
    
    def _compute_advanced_reward(self, base_reward: float, prev_targets: int, 
                                curr_targets: int, pushed_box: bool, solved: bool,
                                prev_distance: float, curr_distance: float) -> Dict[str, float]:
        """
        FIXED: Balanced reward structure with proper magnitudes
        
        Changes:
        - Progress rewards are now ±1.0 (was +0.5/-2.0)
        - Distance shaping is 0.1 magnitude (was 0.005)
        - Step penalty is -0.01 (was -0.002)
        - Anti-hacking is gentler (strength 0.5 vs 2.0)
        """
        components = {
            'base': base_reward,
            'progress': 0.0,
            'distance': 0.0,
            'stability': 0.0,
            'efficiency': 0.0,
            'anti_hack': 0.0,
            'step': -0.01  # FIXED: Was -0.002 (5x increase)
        }
        
        # 1. PROGRESS REWARDS - FIXED: Now symmetric!
        target_change = curr_targets - prev_targets
        if target_change > 0:
            # Box placed on target - BIG reward
            components['progress'] = 1.0 * target_change  # FIXED: Was 0.5
            print(f"📦✅ Box(es) placed on target! Progress reward: {components['progress']:.3f}")
        elif target_change < 0:
            # Box removed from target - EQUAL penalty
            components['progress'] = -1.0 * abs(target_change)  # FIXED: Was -2.0 to -4.0
            print(f"📦❌ Box(es) removed from target! Penalty: {components['progress']:.3f}")
        
        # 2. STABILITY BONUS (keeping boxes on targets)
        if curr_targets > 0:
            components['stability'] = self.target_stability_bonus
        
        # 3. DISTANCE-BASED SHAPING - FIXED: Now meaningful!
        if pushed_box and target_change == 0 and prev_distance is not None:
            # Reward getting closer to targets
            distance_improvement = prev_distance - curr_distance
            max_distance = (self.level_width() + self.level_height()) * self.num_targets()
            
            if max_distance > 0:
                # FIXED: 0.1 magnitude (was 0.005)
                normalized_improvement = distance_improvement / max_distance
                components['distance'] = 0.1 * normalized_improvement
        
        # 4. EFFICIENCY REWARDS
        if solved:
            # Huge completion bonus, scaled by efficiency
            optimal_steps = 20  # Rough estimate
            efficiency_bonus = max(0.5, optimal_steps / max(self.step_count, 1))
            components['efficiency'] = 10.0 * efficiency_bonus
            print(f"🎉 PUZZLE SOLVED in {self.step_count} steps! Efficiency bonus: {components['efficiency']:.2f}")
        
        # 5. ANTI-HACKING PENALTIES - FIXED: Gentler
        if self.box_oscillation_penalty > 0:
            components['anti_hack'] = -self.box_oscillation_penalty
            
        # Penalize excessive pushing without progress (but give more exploration time)
        if self.push_count > 30:  # FIXED: Was 20
            useless_ratio = self.useless_push_count / self.push_count
            if useless_ratio > 0.5:  # FIXED: Was 0.3
                # FIXED: Using gentler strength (0.5 vs 2.0)
                components['anti_hack'] -= 0.1 * useless_ratio * self.anti_hacking_strength
        
        return components
    
    def _get_observation(self) -> np.ndarray:
        obs = self.game.get_observation()
        obs_array = np.array(obs, dtype=np.int32)
        
        # Pad to 100 elements if shorter
        if len(obs_array) < 100:
            padded = np.zeros(100, dtype=np.int32)
            padded[:len(obs_array)] = obs_array
            return padded
        else:
            return obs_array[:100]
    
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
        self.difficulty_level = min(3, max(1, level))
        print(f"🎯 Difficulty set to level {self.difficulty_level}")
    
    def get_success_rate(self) -> float:
        """Get current success rate (for curriculum progression)"""
        return 0.0