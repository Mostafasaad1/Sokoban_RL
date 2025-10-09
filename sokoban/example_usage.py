#!/usr/bin/env python3
"""
Example usage of the Sokoban engine for reinforcement learning.

This demonstrates how to use the C++ Sokoban engine from Python,
including basic gameplay and integration patterns for RL frameworks.
"""

import numpy as np
try:
    import sokoban_engine  # The compiled pybind11 module
except ImportError:
    print("Warning: sokoban_engine module not found. This example requires the compiled C++ module.")
    print("To compile: c++ -O3 -Wall -shared -std=c++20 -fPIC `python3 -m pybind11 --includes` python_bindings.cpp sokoban.cpp -o sokoban_engine`python3-config --extension-suffix`")
    exit(1)

def render_game(game):
    """Simple ASCII rendering of the game state."""
    grid = game.get_grid()
    symbols = {
        game.WALL: '#',
        game.EMPTY: ' ',
        game.PLAYER: '@',
        game.BOX: '$',
        game.TARGET: '.',
        game.BOX_ON_TARGET: '*',
        game.PLAYER_ON_TARGET: '+'
    }
    
    for row in grid:
        print(''.join(symbols[cell] for cell in row))
    print()

def play_interactive():
    """Interactive gameplay demo."""
    game = sokoban_engine.Sokoban()
    
    action_map = {
        'w': game.UP,
        's': game.DOWN,
        'a': game.LEFT,
        'd': game.RIGHT,
        'r': -1  # Reset
    }
    
    print("Sokoban Game - Use WASD to move, R to reset, Q to quit")
    print("Goal: Push all boxes ($) onto targets (.)")
    print()
    
    while True:
        render_game(game)
        
        if game.is_solved():
            print("🎉 Level solved! Press R to reset or Q to quit.")
        
        try:
            action = input("Action (wasd/r/q): ").lower().strip()
            
            if action == 'q':
                break
            elif action == 'r':
                game.reset()
                continue
            elif action in action_map:
                obs, reward, done = game.step(action_map[action])
                if reward > 0:
                    print(f"Reward: {reward}")
            else:
                print("Invalid action. Use w/a/s/d/r/q")
                
        except KeyboardInterrupt:
            break
    
    print("Thanks for playing!")

def benchmark_performance():
    """Performance benchmark for RL training."""
    import time
    
    game = sokoban_engine.Sokoban()
    num_steps = 100000
    
    print(f"Running {num_steps} random steps...")
    
    start_time = time.time()
    
    for i in range(num_steps):
        action = np.random.randint(0, 4)
        obs, reward, done = game.step(action)
        
        if done or i % 1000 == 0:
            game.reset()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Completed {num_steps} steps in {duration:.3f} seconds")
    print(f"Performance: {num_steps/duration:.1f} steps/second")
    print(f"Average step time: {duration/num_steps*1000:.3f} ms")

def demonstrate_gym_interface():
    """Demonstrate a Gym-like interface wrapper."""
    
    class SokobanEnv:
        """OpenAI Gym-like wrapper for the Sokoban engine."""
        
        def __init__(self, width=10, height=10):
            self.game = sokoban_engine.Sokoban(width, height)
            self.action_space_n = 4
            self.observation_space_shape = (width * height,)
        
        def reset(self):
            self.game.reset()
            return np.array(self.game.get_observation(), dtype=np.int32)
        
        def step(self, action):
            obs, reward, done = self.game.step(action)
            obs = np.array(obs, dtype=np.int32)
            info = {
                'player_pos': self.game.get_player_position(),
                'is_solved': self.game.is_solved()
            }
            return obs, reward, done, info
        
        def load_level(self, level_str):
            self.game.load_level(level_str)
            return self.reset()
    
    # Demo usage
    env = SokobanEnv()
    
    print("Gym-like interface demo:")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(10):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: action={action}, reward={reward}, done={done}")
        
        if done:
            obs = env.reset()
            print("Environment reset")

def test_custom_level():
    """Test loading and playing a custom level."""
    game = sokoban_engine.Sokoban()
    
    # A simple custom level
    custom_level = """
########
#   .  #
#  $@$ #
#   .  #
########"""
    
    print("Loading custom level:")
    game.load_level(custom_level)
    render_game(game)
    
    print(f"Level size: {game.get_width()}x{game.get_height()}")
    print(f"Player position: {game.get_player_position()}")
    print(f"Solved: {game.is_solved()}")

if __name__ == "__main__":
    print("Sokoban Engine Demo")
    print("==================")
    
    # Run demonstrations
    test_custom_level()
    print()
    
    demonstrate_gym_interface()
    print()
    
    benchmark_performance()
    print()
    
    # Uncomment for interactive play
    # play_interactive()