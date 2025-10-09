# sokoban_env.py
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

class SokobanEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}

    def __init__(self, render_mode=None):
        self.game = soko.Sokoban()
        self.observation_space = gym.spaces.Box(low=0, high=6, shape=(100,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._current_obs = None
        self.grid_shape = (10, 10)  # Default 10x10 grid
        self.window_size = 600
        self.tile_size = self.window_size // self.grid_shape[1]
        
        if self.render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render_mode: {self.render_mode}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()  # This returns void, so don't assign to grid
        grid = self.game.get_observation()  # Get the observation after reset
        self._current_obs = np.array(grid, dtype=np.int32)
        if self.render_mode == "human":
            self.render()
        return self._current_obs, {}

    def step(self, action):
        # The C++ step returns (observation, reward, terminated)
        grid, reward, terminated = self.game.step(action)
        self._current_obs = np.array(grid, dtype=np.int32)
        truncated = False
        info = {}
        if self.render_mode == "human":
            self.render()
        return self._current_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()

    def _render_human(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Sokoban")
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
        return '\n'.join(result)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None