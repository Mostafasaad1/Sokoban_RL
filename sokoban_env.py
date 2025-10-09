import gymnasium as gym
import numpy as np
import pygame

# Constants will be imported from the compiled module
class SokobanEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}

    def __init__(self, render_mode=None):
        # Import here to avoid issues during compilation
        import sokoban_engine
        self.sokoban_engine = sokoban_engine

        self.game = sokoban_engine.Sokoban()

        # Get actual observation dimensions
        obs = self.game.get_observation()
        self.obs_dim = len(obs)

        self.observation_space = gym.spaces.Box(low=0, high=6, shape=(self.obs_dim,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._current_obs = None
        self.window_size = 600

        if self.render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render_mode: {self.render_mode}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        grid = self.game.get_observation()
        self._current_obs = np.array(grid, dtype=np.int32)

        # Update grid dimensions
        self.grid_width = self.game.get_width()
        self.grid_height = self.game.get_height()
        self.tile_size = min(self.window_size // self.grid_width, self.window_size // self.grid_height)

        if self.render_mode == "human":
            self.render()
        return self._current_obs, {}

    def step(self, action):
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

        grid = self._current_obs.reshape((self.grid_height, self.grid_width))
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                tile = grid[row, col]
                x = col * self.tile_size
                y = row * self.tile_size
                rect = pygame.Rect(x, y, self.tile_size, self.tile_size)

                # Determine base color
                is_target = tile in [self.sokoban_engine.TARGET, self.sokoban_engine.BOX_ON_TARGET, self.sokoban_engine.PLAYER_ON_TARGET]
                base_color = (144, 238, 144) if is_target else (255, 255, 255)
                pygame.draw.rect(self.window, base_color, rect)

                # Draw objects
                if tile == self.sokoban_engine.WALL:
                    pygame.draw.rect(self.window, (100, 100, 100), rect)
                elif tile == self.sokoban_engine.BOX:
                    pygame.draw.rect(self.window, (165, 42, 42), rect)
                elif tile == self.sokoban_engine.BOX_ON_TARGET:
                    pygame.draw.rect(self.window, (0, 128, 0), rect)
                elif tile in [self.sokoban_engine.PLAYER, self.sokoban_engine.PLAYER_ON_TARGET]:
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

        grid = self._current_obs.reshape((self.grid_height, self.grid_width))

        chars = {
            self.sokoban_engine.WALL: '#',
            self.sokoban_engine.EMPTY: ' ',
            self.sokoban_engine.PLAYER: '@',
            self.sokoban_engine.BOX: '$',
            self.sokoban_engine.TARGET: '.',
            self.sokoban_engine.BOX_ON_TARGET: '*',
            self.sokoban_engine.PLAYER_ON_TARGET: '+',
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
