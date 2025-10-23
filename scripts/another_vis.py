import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import pygame
from collections import deque

from optimized_sokoban_env import OptimizedSokobanEnv

class FixedPPONetwork(nn.Module):
    """Network architecture matching the training code"""
    
    def __init__(self, obs_shape, num_actions, hidden_size=256):
        super().__init__()
        
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        x = x.float() / 6.0
        x = self.input_norm(x)
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value.squeeze(-1)

class SokobanVisualizer:
    def __init__(self, model_path, env, device='cpu'):
        self.env = env
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Level tracking
        self.current_level_index = -1
        self.current_level_info = {}
        
        # Get initial dimensions
        self.level_width = 10
        self.level_height = 10
        
        # Pygame setup
        pygame.init()
        self.cell_size = 50
        self.margin = 2
        self.info_panel_height = 180
        self.stats_panel_height = 120
        
        # Colors
        self.colors = {
            'background': (40, 44, 52),
            'wall': (86, 98, 112),
            'floor': (56, 62, 74),
            'player': (97, 175, 239),
            'box': (229, 192, 123),
            'target': (152, 195, 121),
            'box_on_target': (198, 120, 221),
            'player_on_target': (86, 182, 194),
            'text': (220, 220, 220),
            'highlight': (255, 255, 255),
            'success': (152, 195, 121),
            'failure': (224, 108, 117)
        }
        
        # Initialize display
        self.screen_width = 700
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Sokoban Multi-Level Visualizer")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []
        self.action_history = deque(maxlen=10)
        self.level_specific_stats = {}
        
        # Controls
        self.paused = False
        self.auto_play = True
        self.step_delay = 0.1
        self.turbo_mode = False
        
        # Performance
        self.fps_history = deque(maxlen=100)
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        # Reset and update dimensions
        self.reset_episode()
        
        print("🎮 Multi-Level Visualizer Ready!")
        print("Controls: SPACE=Pause, R=Reset, N=Next Level, T=Turbo, +/-=Speed, ESC=Quit")

    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        obs_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n
        
        model = FixedPPONetwork(obs_shape, num_actions, hidden_size=256).to(self.device)
        state_dict = checkpoint.get('network_state_dict', checkpoint)
        
        try:
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully!")
        except RuntimeError as e:
            print(f"⚠️  Loading with strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        return model

    def update_dimensions(self):
        """Update screen dimensions based on current level"""
        self.level_width = self.env.level_width()
        self.level_height = self.env.level_height()
        
        calc_width = max(700, self.level_width * (self.cell_size + self.margin) + self.margin)
        calc_height = max(600, self.level_height * (self.cell_size + self.margin) + self.margin + 
                         self.info_panel_height + self.stats_panel_height)
        
        if calc_width != self.screen_width or calc_height != self.screen_height:
            self.screen_width = calc_width
            self.screen_height = calc_height
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def reset_episode(self):
        """Reset for new episode"""
        obs, info = self.env.reset()
        self.current_obs = obs
        self.episode_reward = 0
        self.episode_step = 0
        self.done = False
        self.solved = False
        self.action_history.clear()
        
        # Update level info
        self.current_level_index = info.get('level_index', -1)
        self.current_level_info = {
            'index': self.current_level_index,
            'width': info.get('level_width', 10),
            'height': info.get('level_height', 10),
            'boxes': info.get('total_targets', 2)
        }
        
        self.update_dimensions()
        return obs

    def get_action(self, obs):
        """Get action from model"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, value = self.model(obs_tensor)
            action = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
        return action.item(), probs[0].cpu().numpy(), value.item()

    def step_environment(self, action):
        """Take step in environment"""
        obs, reward, done, truncated, info, _ = self.env.step(action)
        self.current_obs = obs
        self.episode_reward += reward
        self.episode_step += 1
        self.done = done or truncated
        self.solved = info.get('is_solved', False)
        
        # Store action
        action_names = ['↑ Up', '↓ Down', '← Left', '→ Right']
        self.action_history.append({
            'step': self.episode_step,
            'action': action,
            'action_name': action_names[action] if action < 4 else str(action),
            'reward': reward
        })
        
        return obs, reward, done, info

    def get_grid(self):
        """Get grid from environment"""
        try:
            if hasattr(self.env, 'game') and hasattr(self.env.game, 'get_grid'):
                grid_list = self.env.game.get_grid()
                return np.array(grid_list, dtype=np.int32)
        except Exception:
            pass
        return np.zeros((self.level_height, self.level_width), dtype=np.int32)

    def draw_grid(self):
        """Draw the game grid"""
        grid = self.get_grid()
        if not hasattr(grid, 'shape'):
            grid = np.array(grid)
            
        grid_height, grid_width = grid.shape[0], grid.shape[1]
        
        self.screen.fill(self.colors['background'])
        
        for y in range(min(grid_height, self.level_height)):
            for x in range(min(grid_width, self.level_width)):
                rect = pygame.Rect(
                    x * (self.cell_size + self.margin) + self.margin,
                    y * (self.cell_size + self.margin) + self.margin,
                    self.cell_size, self.cell_size
                )
                
                cell = grid[y, x] if y < grid.shape[0] and x < grid.shape[1] else 0
                
                # Map to colors
                color_map = {
                    self.env.WALL: self.colors['wall'],
                    self.env.EMPTY: self.colors['floor'],
                    self.env.PLAYER: self.colors['player'],
                    self.env.BOX: self.colors['box'],
                    self.env.TARGET: self.colors['target'],
                    self.env.BOX_ON_TARGET: self.colors['box_on_target'],
                    self.env.PLAYER_ON_TARGET: self.colors['player_on_target']
                }
                
                color = color_map.get(cell, self.colors['floor'])
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (70, 74, 82), rect, 1)
                
                # Draw player indicator
                if cell in [self.env.PLAYER, self.env.PLAYER_ON_TARGET]:
                    center = (x * (self.cell_size + self.margin) + self.margin + self.cell_size // 2,
                             y * (self.cell_size + self.margin) + self.margin + self.cell_size // 2)
                    pygame.draw.circle(self.screen, (255, 255, 255), center, self.cell_size // 4)

    def draw_info_panel(self):
        """Draw info panel"""
        panel_y = self.screen_height - self.info_panel_height - self.stats_panel_height
        panel_rect = pygame.Rect(0, panel_y, self.screen_width, self.info_panel_height)
        pygame.draw.rect(self.screen, (50, 54, 62), panel_rect)
        
        # Status
        if self.solved:
            status_text, status_color = "✅ SOLVED!", self.colors['success']
        elif self.done:
            status_text, status_color = "❌ FAILED", self.colors['failure']
        else:
            status_text, status_color = "Playing...", self.colors['text']
        
        mode_text = "⏸ PAUSED" if self.paused else "▶ AUTO"
        if self.turbo_mode:
            mode_text += " 🚀"
        
        # Info texts
        level_idx = self.current_level_index
        texts = [
            f"Episode: {len(self.episode_rewards) + 1}",
            f"Step: {self.episode_step}/200",
            f"Reward: {self.episode_reward:.2f}",
            f"Level: {level_idx if level_idx >= 0 else 'Custom'}",
            f"Size: {self.level_width}x{self.level_height}"
        ]
        
        for i, text in enumerate(texts):
            surf = self.font_medium.render(text, True, self.colors['text'])
            self.screen.blit(surf, (15, panel_y + 15 + i * 28))
        
        # Status (large)
        status_surf = self.font_large.render(status_text, True, status_color)
        self.screen.blit(status_surf, (15, panel_y + 150))
        
        # Mode (right)
        mode_surf = self.font_medium.render(mode_text, True, self.colors['highlight'])
        self.screen.blit(mode_surf, (self.screen_width - 250, panel_y + 15))

    def draw_stats_panel(self):
        """Draw stats panel"""
        panel_y = self.screen_height - self.stats_panel_height
        panel_rect = pygame.Rect(0, panel_y, self.screen_width, self.stats_panel_height)
        pygame.draw.rect(self.screen, (45, 49, 57), panel_rect)
        
        n = min(50, len(self.success_history))
        success_rate = np.mean(self.success_history[-n:]) if n > 0 else 0
        avg_reward = np.mean(self.episode_rewards[-n:]) if n > 0 else 0
        avg_steps = np.mean(self.episode_steps[-n:]) if n > 0 else 0
        
        # FPS
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_fps_update >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.fps_history.append(fps)
            self.frame_count = 0
            self.last_fps_update = current_time
        
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Level-specific stats
        level_idx = self.current_level_index
        level_stats = self.level_specific_stats.get(level_idx, {'attempts': 0, 'solves': 0})
        level_rate = (level_stats['solves'] / level_stats['attempts']) if level_stats['attempts'] > 0 else 0
        
        texts = [
            f"📊 Episodes: {len(self.episode_rewards)}",
            f"🏆 Success: {success_rate:.1%} (last {n})",
            f"💰 Avg Reward: {avg_reward:.2f}",
            f"📏 Avg Steps: {avg_steps:.1f}",
            f"⚡ FPS: {avg_fps:.0f}",
            f"🗺️  Level {level_idx}: {level_stats['solves']}/{level_stats['attempts']} ({level_rate:.1%})"
        ]
        
        for i, text in enumerate(texts):
            surf = self.font_small.render(text, True, self.colors['text'])
            self.screen.blit(surf, (15, panel_y + 10 + i * 18))

    def draw_action_history(self):
        """Draw recent actions"""
        if not self.action_history:
            return
            
        history_x = self.screen_width - 290
        history_y = 15
        history_height = min(250, 30 + len(self.action_history) * 22)
        
        rect = pygame.Rect(history_x, history_y, 280, history_height)
        pygame.draw.rect(self.screen, (55, 59, 67), rect)
        pygame.draw.rect(self.screen, self.colors['text'], rect, 2)
        
        title = self.font_medium.render("Recent Actions", True, self.colors['highlight'])
        self.screen.blit(title, (history_x + 10, history_y + 8))
        
        for i, action_info in enumerate(list(self.action_history)[-8:]):
            color = self.colors['success'] if action_info['reward'] > 0 else self.colors['text']
            text = f"{action_info['action_name']} (R:{action_info['reward']:.2f})"
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (history_x + 10, history_y + 35 + i * 22))

    def render(self):
        """Render everything"""
        self.draw_grid()
        self.draw_info_panel()
        self.draw_stats_panel()
        self.draw_action_history()
        pygame.display.flip()

    def run(self):
        """Main loop"""
        last_step_time = time.time()
        running = True
        
        print("\n🎮 Visualizer running...")
        
        while running:
            current_time = time.time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                    elif event.key == pygame.K_n:
                        # Next level
                        self.reset_episode()
                        print(f"📍 Switched to level {self.current_level_index}")
                    elif event.key == pygame.K_t:
                        self.turbo_mode = not self.turbo_mode
                        print(f"🚀 Turbo: {'ON' if self.turbo_mode else 'OFF'}")
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        self.step_delay = max(0.01, self.step_delay * 0.5)
                    elif event.key == pygame.K_MINUS:
                        self.step_delay = min(1.0, self.step_delay * 2.0)
            
            # Take action
            if not self.paused and not self.done:
                step_ready = self.turbo_mode or (current_time - last_step_time >= self.step_delay)
                
                if step_ready:
                    action, _, _ = self.get_action(self.current_obs)
                    self.step_environment(action)
                    last_step_time = current_time
            
            # Handle episode end
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_steps.append(self.episode_step)
                self.success_history.append(self.solved)
                
                # Update level stats
                if self.current_level_index >= 0:
                    if self.current_level_index not in self.level_specific_stats:
                        self.level_specific_stats[self.current_level_index] = {'attempts': 0, 'solves': 0}
                    self.level_specific_stats[self.current_level_index]['attempts'] += 1
                    if self.solved:
                        self.level_specific_stats[self.current_level_index]['solves'] += 1
                
                status = "✅ SOLVED" if self.solved else "❌ FAILED"
                print(f"{status} - Level {self.current_level_index}, Steps: {self.episode_step}, Reward: {self.episode_reward:.2f}")
                
                if self.auto_play:
                    if not self.turbo_mode:
                        pygame.time.delay(500)
                    self.reset_episode()
                else:
                    self.paused = True
            
            self.render()
            
            if not self.turbo_mode:
                pygame.time.delay(10)
        
        # Final stats
        if self.episode_rewards:
            print(f"\n📊 Final: {len(self.episode_rewards)} episodes, {np.mean(self.success_history):.1%} success rate")
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Multi-Level Sokoban Visualizer')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--level-mode', type=str, default='random', 
                       choices=['random', 'sequential', 'curriculum'])
    parser.add_argument('--difficulty', type=int, default=1)
    parser.add_argument('--delay', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--turbo', action='store_true')
    
    args = parser.parse_args()
    
    # Create environment
    env = OptimizedSokobanEnv(
        max_episode_steps=200,
        curriculum_mode=False,
        difficulty_level=args.difficulty,
        anti_hacking_strength=0.5,
        level_selection_mode=args.level_mode
    )
    
    visualizer = SokobanVisualizer(args.model_path, env, args.device)
    visualizer.step_delay = args.delay
    visualizer.turbo_mode = args.turbo
    
    try:
        visualizer.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()

if __name__ == '__main__':
    main()