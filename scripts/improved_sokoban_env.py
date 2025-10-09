# improved_sokoban_env.py
import gymnasium as gym
import numpy as np
import sokoban_engine as soko
import pygame

WALL = soko.WALL
EMPTY = soko.EMPTY
PLAYER = soko.PLAYER
BOX = soko.BOX
TARGET = soko.TARGET
BOX_ON_TARGET = soko.BOX_ON_TARGET
PLAYER_ON_TARGET = soko.PLAYER_ON_TARGET

class ImprovedSokobanEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}

    def __init__(self, render_mode=None):
        self.game = soko.Sokoban()
        self.observation_space = gym.spaces.Box(low=0, high=6, shape=(100,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._current_obs = None
        self.grid_shape = (10, 10)  
        self.window_size = 600
        self.tile_size = self.window_size // self.grid_shape[1]
        
        # Episode tracking for smarter rewards
        self.step_count = 0
        self.max_steps = 200  # Prevent infinite episodes
        self.boxes_on_targets_history = []
        self.last_boxes_on_targets = 0
        
        if self.render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render_mode: {self.render_mode}")

    def count_boxes_on_targets(self, grid):
        """Count how many boxes are currently on target positions"""
        count = 0
        for cell in grid:
            if cell == BOX_ON_TARGET:
                count += 1
        return count

    def calculate_distance_reward(self, grid):
        """Calculate reward based on boxes getting closer to targets"""
        # Get player and box positions
        width = self.game.get_width()
        height = self.game.get_height()
        grid_2d = np.array(grid).reshape((height, width))
        
        boxes = []
        targets = []
        
        for y in range(height):
            for x in range(width):
                cell = grid_2d[y, x]
                if cell == BOX:
                    boxes.append((x, y))
                elif cell == TARGET:
                    targets.append((x, y))
                elif cell == BOX_ON_TARGET:
                    # Box already on target - this is good!
                    pass
        
        if not boxes or not targets:
            return 0.0
        
        # Calculate minimum distance from each box to nearest target
        total_distance = 0
        for box_x, box_y in boxes:
            min_dist = min(abs(box_x - target_x) + abs(box_y - target_y) 
                          for target_x, target_y in targets)
            total_distance += min_dist
        
        # Reward inverse of distance (closer = better)
        # Normalize by max possible distance to keep rewards small
        max_possible_distance = (width + height) * len(boxes)
        distance_reward = -total_distance / max_possible_distance
        return distance_reward * 0.01  # Keep small to not overshadow main rewards

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        grid = self.game.get_observation()
        self._current_obs = np.array(grid, dtype=np.int32)
        
        # Reset episode tracking
        self.step_count = 0
        self.last_boxes_on_targets = self.count_boxes_on_targets(grid)
        self.boxes_on_targets_history = [self.last_boxes_on_targets]
        
        if self.render_mode == "human":
            self.render()
        return self._current_obs, {}

    def step(self, action):
        grid, base_reward, terminated, box_pushed = self.game.step(action)
        self._current_obs = np.array(grid, dtype=np.int32)
        self.step_count += 1
        
        # Calculate improved reward
        total_reward = base_reward
        
        # 1. Main completion reward (unchanged)
        if base_reward > 0:
            print(f"🎉 PUZZLE SOLVED! Base reward: {base_reward}")
        
        # 2. Progress reward - boxes getting on targets
        current_boxes_on_targets = self.count_boxes_on_targets(grid)
        progress_reward = 0.0
        
        if current_boxes_on_targets > self.last_boxes_on_targets:
            # Box placed on target
            progress_reward = 0.3
            print(f"📦✅ Box placed on target! Progress reward: {progress_reward}")
        elif current_boxes_on_targets < self.last_boxes_on_targets:
            # Box removed from target - small penalty
            progress_reward = -0.1
            print(f"📦❌ Box removed from target! Penalty: {progress_reward}")
        
        # 3. Small distance-based reward for getting boxes closer to targets
        distance_reward = self.calculate_distance_reward(grid)
        
        # 4. Box push reward - only if making progress
        push_reward = 0.0
        if box_pushed:
            # Only reward box pushes if they lead to progress or at least don't hurt
            if progress_reward >= 0:
                push_reward = 0.02  # Much smaller than before
            else:
                push_reward = -0.01  # Small penalty for counterproductive pushes
        
        # 5. Small step penalty to encourage efficiency
        step_penalty = -0.001
        
        # 6. Truncation for episodes that go too long
        truncated = self.step_count >= self.max_steps
        if truncated:
            print(f"⏰ Episode truncated at {self.step_count} steps")
            total_reward += -0.05  # Penalty for not solving quickly
        
        # Combine all rewards
        total_reward += progress_reward + distance_reward + push_reward + step_penalty
        
        # Update tracking
        self.last_boxes_on_targets = current_boxes_on_targets
        self.boxes_on_targets_history.append(current_boxes_on_targets)
        
        # Debug output for significant rewards
        if abs(total_reward) > 0.01:
            components = {
                'base': base_reward,
                'progress': progress_reward, 
                'distance': distance_reward,
                'push': push_reward,
                'step': step_penalty
            }
            print(f"💰 Total reward: {total_reward:.3f} = {components}")
        
        if self.render_mode == "human":
            self.render()
        
        return self._current_obs, total_reward, terminated, truncated, {
            'boxes_on_targets': current_boxes_on_targets,
            'step_count': self.step_count,
            'boxes_on_targets_history': self.boxes_on_targets_history.copy()
        }

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()

    def _render_human(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Improved Sokoban")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Update grid_shape based on actual game dimensions
        width = self.game.get_width()
        height = self.game.get_height()
        self.grid_shape = (height, width)
        self.tile_size = min(self.window_size // width, self.window_size // height)
        
        grid = self._current_obs.reshape(self.grid_shape)
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                tile = grid[row, col]
                x = col * self.tile_size
                y = row * self.tile_size
                rect = pygame.Rect(x, y, self.tile_size, self.tile_size)

                # Determine base color
                is_target = tile in [TARGET, BOX_ON_TARGET, PLAYER_ON_TARGET]
                base_color = (144, 238, 144) if is_target else (255, 255, 255)
                pygame.draw.rect(self.window, base_color, rect)

                # Draw objects
                if tile == WALL:
                    pygame.draw.rect(self.window, (100, 100, 100), rect)
                elif tile == BOX:
                    pygame.draw.rect(self.window, (165, 42, 42), rect)
                elif tile == BOX_ON_TARGET:
                    pygame.draw.rect(self.window, (0, 128, 0), rect)
                elif tile in [PLAYER, PLAYER_ON_TARGET]:
                    center = (x + self.tile_size // 2, y + self.tile_size // 2)
                    radius = self.tile_size // 2 - 5
                    pygame.draw.circle(self.window, (0, 0, 255), center, radius)

                # Draw tile borders for clarity
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

        # Add step counter
        if hasattr(pygame, 'font'):
            font = pygame.font.Font(None, 36)
            text = font.render(f"Steps: {self.step_count}", True, (0, 0, 0))
            self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_ansi(self):
        if self._current_obs is None:
            return ""
        
        width = self.game.get_width()
        height = self.game.get_height()
        grid = self._current_obs.reshape((height, width))
        
        chars = {
            WALL: '#',
            EMPTY: ' ',
            PLAYER: '@',
            BOX: '$',
            TARGET: '.',
            BOX_ON_TARGET: '*',
            PLAYER_ON_TARGET: '+',
        }
        
        result = []
        for row in grid:
            result.append(''.join(chars.get(cell, '?') for cell in row))
        result.append(f"Steps: {self.step_count}")
        return '\n'.join(result)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None