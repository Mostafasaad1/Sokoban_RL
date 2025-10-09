#!/usr/bin/env python3
"""
Reinforcement Learning Training Visualization Example

Demonstrates how to integrate the Sokoban visualization with RL training loops.
Shows training progress, episode statistics, and game state visualization.
"""

import sys
sys.path.append('.')

from visualize_sokoban import SokobanVisualizer, setup_matplotlib_for_plotting
import sokoban_engine
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

class SokobanRLEnv:
    """OpenAI Gym-style environment wrapper with visualization support."""
    
    def __init__(self, width: int = 10, height: int = 10, level: str = None):
        self.game = sokoban_engine.Sokoban(width, height)
        if level:
            self.game.load_level(level)
        
        self.action_space_n = 4
        self.observation_space_shape = (width * height,)
        self.visualizer = SokobanVisualizer(self.game)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_length = 0
        self.current_episode_reward = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        self.game.reset()
        self.current_episode_length = 0
        self.current_episode_reward = 0
        return np.array(self.game.get_observation(), dtype=np.int32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (observation, reward, done, info)."""
        obs, reward, done = self.game.step(action)
        
        self.current_episode_length += 1
        self.current_episode_reward += reward
        
        obs = np.array(obs, dtype=np.int32)
        info = {
            'player_pos': self.game.get_player_position(),
            'is_solved': self.game.is_solved(),
            'episode_length': self.current_episode_length,
            'episode_reward': self.current_episode_reward
        }
        
        return obs, reward, done, info
    
    def render(self, mode: str = 'ascii') -> None:
        """Render the current game state."""
        if mode == 'ascii':
            self.visualizer.print_ascii()
        elif mode == 'rgb_array':
            self.visualizer.render_static()
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        if not self.episode_rewards:
            return {'episodes': 0, 'avg_reward': 0, 'avg_length': 0}
        
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'total_reward': np.sum(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'success_rate': np.mean([r > 0 for r in self.episode_rewards])
        }

def random_agent_demo(episodes: int = 10, max_steps: int = 100, visualize_every: int = 5):
    """Demonstrate random agent training with visualization."""
    
    print(f"🤖 Random Agent Demo - {episodes} episodes")
    print("="*50)
    
    # Create environment
    env = SokobanRLEnv()
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n🎮 Episode {episode + 1}/{episodes}")
        
        # Show initial state for some episodes
        if episode % visualize_every == 0:
            print("Initial state:")
            env.render('ascii')
        
        for step in range(max_steps):
            # Random action
            action = np.random.randint(0, 4)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if done:
                print(f"✅ Episode completed in {steps} steps! Reward: {reward}")
                if episode % visualize_every == 0:
                    print("Final state:")
                    env.render('ascii')
                    env.visualizer.render_static(f'rl_episode_{episode+1}_final.png')
                break
        else:
            print(f"⏰ Episode timeout after {max_steps} steps")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Print progress
        if episode % 5 == 0 or episode == episodes - 1:
            stats = env.get_training_stats()
            print(f"📊 Progress: Episodes {stats['episodes']}, "
                  f"Avg Reward: {stats['avg_reward']:.3f}, "
                  f"Success Rate: {stats['success_rate']:.1%}")
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_lengths)
    
    # Final statistics
    print(f"\n📈 Final Training Statistics:")
    stats = env.get_training_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")

def plot_training_progress(rewards: List[float], lengths: List[int]):
    """Plot training progress charts."""
    setup_matplotlib_for_plotting()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    episodes = range(1, len(rewards) + 1)
    
    # Plot rewards
    ax1.plot(episodes, rewards, 'b-o', markersize=4)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(episodes, lengths, 'r-o', markersize=4)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode') 
    ax2.set_ylabel('Steps')
    ax2.grid(True, alpha=0.3)
    
    # Plot cumulative success rate
    successes = [1 if r > 0 else 0 for r in rewards]
    cumulative_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
    ax3.plot(episodes, cumulative_success, 'g-o', markersize=4)
    ax3.set_title('Cumulative Success Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training_progress.png', dpi=150, bbox_inches='tight')
    print("📊 Training progress saved as rl_training_progress.png")

def simple_heuristic_agent_demo(episodes: int = 5):
    """Demonstrate a simple heuristic agent."""
    
    print(f"🧠 Heuristic Agent Demo - {episodes} episodes")
    print("Strategy: Move towards nearest box, then towards nearest target")
    print("="*60)
    
    env = SokobanRLEnv()
    
    def heuristic_action(obs, player_pos):
        """Simple heuristic: try to move towards boxes/targets."""
        x, y = player_pos
        
        # Simple strategy: try different directions
        actions = [
            env.game.UP,
            env.game.DOWN, 
            env.game.LEFT,
            env.game.RIGHT
        ]
        
        # For demo purposes, just cycle through actions
        return actions[np.random.randint(0, 4)]
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n🎯 Episode {episode + 1}/{episodes}")
        env.render('ascii')
        
        for step in range(50):  # Max 50 steps
            player_pos = env.game.get_player_position()
            action = heuristic_action(obs, player_pos)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                print(f"🎉 Solved in {steps} steps!")
                env.render('ascii')
                env.visualizer.render_static(f'heuristic_episode_{episode+1}_solved.png')
                break
        else:
            print(f"🔄 Episode ended after {steps} steps (no solution)")
        
        print(f"Episode reward: {episode_reward}")

def create_training_level_examples():
    """Create various level examples for RL training."""
    
    print("🏗️  Creating training level examples...")
    
    levels = {
        'easy': """
#####
#@$.#
#####""",
        
        'medium': """
#######
# . . #
# $@$ #
# . . #
#######""",
        
        'hard': """
#########
#   .   #
# $ $ $ #
#   @   #
# $ $ $ #
#   .   #
#########""",
        
        'corridor': """
###########
#@$   .   #
###########""",
        
        'multiple_boxes': """
##########
#.   .   #
#  $ $   #
#   @    #
#  $ $   #
#.   .   #
##########"""
    }
    
    for name, level_str in levels.items():
        env = SokobanRLEnv()
        env.game.load_level(level_str)
        env.visualizer.render_static(f'training_level_{name}.png')
        print(f"Created {name} level: training_level_{name}.png")

def main():
    """Main demonstration function."""
    print("🚀 Sokoban RL Visualization Demo")
    print("="*40)
    
    print("\n1. Creating training level examples...")
    create_training_level_examples()
    
    print("\n2. Running random agent demo...")
    random_agent_demo(episodes=15, max_steps=50, visualize_every=3)
    
    print("\n3. Running heuristic agent demo...")
    simple_heuristic_agent_demo(episodes=3)
    
    print("\n✅ All demonstrations completed!")
    print("\n📁 Generated files:")
    files = [
        "rl_training_progress.png - Training progress charts",
        "training_level_*.png - Various difficulty levels", 
        "rl_episode_*_final.png - Episode completion states",
        "heuristic_episode_*_solved.png - Heuristic agent solutions"
    ]
    
    for file in files:
        print(f"   • {file}")
    
    print("\n🎯 Use this as a template for your RL experiments!")

if __name__ == "__main__":
    main()