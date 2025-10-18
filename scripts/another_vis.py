import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import pygame
import sys
from collections import deque

from optimized_sokoban_env import OptimizedSokobanEnv

class FixedPPONetwork(nn.Module):
    """
    FIXED network architecture matching the training code.
    Must match exactly for loading weights!
    """
    
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
        
        # Load model with correct architecture
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Get environment dimensions
        self.level_width = env.level_width()
        self.level_height = env.level_height()
        
        print(f"📐 Level dimensions: {self.level_width}x{self.level_height}")
        
        # Pygame setup
        pygame.init()
        self.cell_size = 50
        self.margin = 2
        self.info_panel_height = 140
        self.stats_panel_height = 100
        
        # Colors - modern design
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
        self.screen_width = self.level_width * (self.cell_size + self.margin) + self.margin
        self.screen_height = (self.level_height * (self.cell_size + self.margin) + self.margin + 
                            self.info_panel_height + self.stats_panel_height)
        
        self.screen_width = max(self.screen_width, 700)
        self.screen_height = max(self.screen_height, 600)
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Sokoban RL Visualizer - FIXED MODEL")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []
        self.action_history = deque(maxlen=10)
        
        # Controls
        self.paused = False
        self.step_mode = False
        self.auto_play = True
        self.step_delay = 0.1
        self.turbo_mode = False
        
        # Performance tracking
        self.fps_history = deque(maxlen=100)
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        # Reset environment
        self.reset_episode()
        
        print("🎮 FIXED Sokoban Visualizer Initialized!")
        print(f"📊 Model: {model_path}")
        print("Controls:")
        print("  SPACE: Play/Pause")
        print("  R: Reset episode")
        print("  T: Turbo mode")
        print("  +/-: Speed control")
        print("  ESC: Quit")

    def load_model(self, model_path):
        """Load the trained model with correct architecture"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        obs_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n
        
        # Create model with FIXED architecture (256 hidden units)
        model = FixedPPONetwork(obs_shape, num_actions, hidden_size=256).to(self.device)
        
        state_dict = checkpoint.get('network_state_dict', checkpoint)
        
        try:
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully!")
        except RuntimeError as e:
            print(f"⚠️  Model architecture mismatch: {e}")
            print("Attempting compatibility load...")
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model loaded with compatibility mode")
        
        return model

    def reset_episode(self):
        """Reset the environment for a new episode"""
        obs, _ = self.env.reset()
        self.current_obs = obs
        self.episode_reward = 0
        self.episode_step = 0
        self.done = False
        self.solved = False
        self.action_history.clear()
        return obs

    def get_action(self, obs):
        """Get action from the model"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, value = self.model(obs_tensor)
            
            # Use argmax for deterministic behavior
            action = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
        return action.item(), probs[0].cpu().numpy(), value.item()

    def step_environment(self, action):
        """Take a step in the environment"""
        obs, reward, done, truncated, info, action_tracker = self.env.step(action)
        self.current_obs = obs
        self.episode_reward += reward
        self.episode_step += 1
        self.done = done or truncated
        self.solved = info.get('is_solved', False)
        
        # Store action for history
        action_names = ['↑ Up', '↓ Down', '← Left', '→ Right']
        self.action_history.append({
            'step': self.episode_step,
            'action': action,
            'action_name': action_names[action] if action < len(action_names) else str(action),
            'reward': reward,
            'total_reward': self.episode_reward
        })
        
        return obs, reward, done, info

    def get_grid_representation(self):
        """Extract grid representation from the environment"""
        try:
            if hasattr(self.env, 'game') and hasattr(self.env.game, 'get_grid'):
                grid_list = self.env.game.get_grid()
                grid = np.array(grid_list, dtype=np.int32)
                return grid
        except Exception:
            pass
        
        return np.zeros((self.level_height, self.level_width), dtype=np.int32)

    def draw_grid(self):
        """Draw the Sokoban grid"""
        grid = self.get_grid_representation()
        
        if not hasattr(grid, 'shape'):
            grid = np.array(grid)
            
        grid_height, grid_width = grid.shape[0], grid.shape[1]
        draw_width = min(grid_width, self.level_width)
        draw_height = min(grid_height, self.level_height)
        
        self.screen.fill(self.colors['background'])
        
        for y in range(draw_height):
            for x in range(draw_width):
                rect = pygame.Rect(
                    x * (self.cell_size + self.margin) + self.margin,
                    y * (self.cell_size + self.margin) + self.margin,
                    self.cell_size, self.cell_size
                )
                
                cell_value = grid[y, x] if y < grid.shape[0] and x < grid.shape[1] else 0
                
                # Map cell values to colors
                if cell_value == self.env.WALL:
                    color = self.colors['wall']
                elif cell_value == self.env.EMPTY:
                    color = self.colors['floor']
                elif cell_value == self.env.PLAYER:
                    color = self.colors['player']
                elif cell_value == self.env.BOX:
                    color = self.colors['box']
                elif cell_value == self.env.TARGET:
                    color = self.colors['target']
                elif cell_value == self.env.BOX_ON_TARGET:
                    color = self.colors['box_on_target']
                elif cell_value == self.env.PLAYER_ON_TARGET:
                    color = self.colors['player_on_target']
                else:
                    color = self.colors['floor']
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (70, 74, 82), rect, 1)
                
                # Draw player indicator
                if cell_value in [self.env.PLAYER, self.env.PLAYER_ON_TARGET]:
                    center_x = x * (self.cell_size + self.margin) + self.margin + self.cell_size // 2
                    center_y = y * (self.cell_size + self.margin) + self.margin + self.cell_size // 2
                    pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), self.cell_size // 4)

    def draw_info_panel(self):
        """Draw the information panel"""
        panel_y = self.screen_height - self.info_panel_height - self.stats_panel_height
        panel_rect = pygame.Rect(0, panel_y, self.screen_width, self.info_panel_height)
        pygame.draw.rect(self.screen, (50, 54, 62), panel_rect)
        
        # Episode info
        if self.solved:
            status_text = "✅ SOLVED!"
            status_color = self.colors['success']
        elif self.done:
            status_text = "❌ FAILED"
            status_color = self.colors['failure']
        else:
            status_text = "Playing..."
            status_color = self.colors['text']
        
        mode_text = "⏸ PAUSED" if self.paused else "▶ AUTO" if self.auto_play else "⏯ STEP"
        if self.turbo_mode:
            mode_text += " 🚀"
        
        # Left column
        left_texts = [
            f"Episode: {len(self.episode_rewards) + 1}",
            f"Step: {self.episode_step}/200",
            f"Reward: {self.episode_reward:.2f}",
        ]
        
        for i, text in enumerate(left_texts):
            text_surface = self.font_medium.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (15, panel_y + 15 + i * 30))
        
        # Right column
        status_surface = self.font_large.render(status_text, True, status_color)
        self.screen.blit(status_surface, (15, panel_y + 100))
        
        mode_surface = self.font_medium.render(mode_text, True, self.colors['highlight'])
        self.screen.blit(mode_surface, (self.screen_width - 200, panel_y + 15))
        
        delay_surface = self.font_small.render(f"Delay: {self.step_delay:.3f}s", True, self.colors['text'])
        self.screen.blit(delay_surface, (self.screen_width - 200, panel_y + 45))

    def draw_stats_panel(self):
        """Draw the statistics panel"""
        panel_y = self.screen_height - self.stats_panel_height
        panel_rect = pygame.Rect(0, panel_y, self.screen_width, self.stats_panel_height)
        pygame.draw.rect(self.screen, (45, 49, 57), panel_rect)
        
        # Calculate statistics
        n = min(50, len(self.success_history))
        success_rate = np.mean(self.success_history[-n:]) if n > 0 else 0
        avg_reward = np.mean(self.episode_rewards[-n:]) if n > 0 else 0
        avg_steps = np.mean(self.episode_steps[-n:]) if n > 0 else 0
        
        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_fps_update >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.fps_history.append(fps)
            self.frame_count = 0
            self.last_fps_update = current_time
        
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        stats_texts = [
            f"📊 Episodes: {len(self.episode_rewards)}",
            f"🏆 Success Rate: {success_rate:.1%} (last {n})",
            f"💰 Avg Reward: {avg_reward:.2f}",
            f"📏 Avg Steps: {avg_steps:.1f}",
            f"⚡ FPS: {avg_fps:.0f}"
        ]
        
        for i, text in enumerate(stats_texts):
            text_surface = self.font_small.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (15, panel_y + 10 + i * 18))

    def draw_action_history(self):
        """Draw recent action history"""
        if not self.action_history:
            return
            
        history_width = 280
        history_x = self.screen_width - history_width - 15
        history_y = 15
        history_height = min(250, 30 + len(self.action_history) * 22)
        
        history_rect = pygame.Rect(history_x, history_y, history_width, history_height)
        pygame.draw.rect(self.screen, (55, 59, 67), history_rect)
        pygame.draw.rect(self.screen, self.colors['text'], history_rect, 2)
        
        title = self.font_medium.render("Recent Actions", True, self.colors['highlight'])
        self.screen.blit(title, (history_x + 10, history_y + 8))
        
        for i, action_info in enumerate(list(self.action_history)[-8:]):
            reward_color = self.colors['success'] if action_info['reward'] > 0 else self.colors['text']
            action_text = f"{action_info['action_name']} (R:{action_info['reward']:.2f})"
            text_surface = self.font_small.render(action_text, True, reward_color)
            self.screen.blit(text_surface, (history_x + 10, history_y + 35 + i * 22))

    def render(self):
        """Render the complete visualization"""
        self.draw_grid()
        self.draw_info_panel()
        self.draw_stats_panel()
        self.draw_action_history()
        
        pygame.display.flip()

    def run(self):
        """Main visualization loop"""
        last_step_time = time.time()
        running = True
        
        print("\n🎮 Starting visualization...")
        print("📊 Watch the agent play with the FIXED model!\n")
        
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
                    elif event.key == pygame.K_t:
                        self.turbo_mode = not self.turbo_mode
                        print(f"🚀 Turbo mode: {'ON' if self.turbo_mode else 'OFF'}")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.step_delay = max(0.01, self.step_delay * 0.5)
                        print(f"⚡ Speed increased: delay = {self.step_delay:.3f}s")
                    elif event.key == pygame.K_MINUS:
                        self.step_delay = min(1.0, self.step_delay * 2.0)
                        print(f"🐢 Speed decreased: delay = {self.step_delay:.3f}s")
            
            # Take action if not paused and not done
            if not self.paused and not self.done:
                step_ready = False
                
                if self.turbo_mode:
                    step_ready = True
                elif self.auto_play:
                    step_ready = (current_time - last_step_time >= self.step_delay)
                
                if step_ready:
                    action, probs, value = self.get_action(self.current_obs)
                    self.step_environment(action)
                    last_step_time = current_time
            
            # Reset if episode is done
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_steps.append(self.episode_step)
                self.success_history.append(self.solved)
                
                if self.solved:
                    print(f"✅ Episode {len(self.episode_rewards)} SOLVED in {self.episode_step} steps! Reward: {self.episode_reward:.2f}")
                else:
                    print(f"❌ Episode {len(self.episode_rewards)} failed. Steps: {self.episode_step}, Reward: {self.episode_reward:.2f}")
                
                if self.auto_play and not self.turbo_mode:
                    pygame.time.delay(500)
                    self.reset_episode()
                elif self.auto_play and self.turbo_mode:
                    self.reset_episode()
                else:
                    self.paused = True
            
            # Render
            self.render()
            
            if not self.turbo_mode:
                pygame.time.delay(10)
        
        # Final statistics
        if len(self.episode_rewards) > 0:
            print("\n" + "="*60)
            print("📊 FINAL STATISTICS")
            print("="*60)
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Success Rate: {np.mean(self.success_history):.1%}")
            print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Average Steps: {np.mean(self.episode_steps):.1f}")
            print("="*60)
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Visualize FIXED Sokoban Model')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to the trained model (fixed_sokoban_model_*.pt)')
    parser.add_argument('--difficulty', type=int, default=1, 
                       help='Difficulty level (1-3)')
    parser.add_argument('--delay', type=float, default=0.1, 
                       help='Delay between steps in seconds')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to use')
    parser.add_argument('--turbo', action='store_true', 
                       help='Start in turbo mode')
    
    args = parser.parse_args()
    
    # Create environment with FIXED settings
    env = OptimizedSokobanEnv(
        max_episode_steps=200,
        curriculum_mode=False,
        difficulty_level=args.difficulty,
        anti_hacking_strength=0.5
    )
    
    # Create and run visualizer
    visualizer = SokobanVisualizer(
        model_path=args.model_path,
        env=env,
        device=args.device
    )
    visualizer.step_delay = args.delay
    visualizer.turbo_mode = args.turbo
    
    try:
        visualizer.run()
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        raise

if __name__ == '__main__':
    main()