# Sokoban Reinforcement Learning Environment

A high-performance Sokoban environment built in C++ with Python bindings, designed specifically for reinforcement learning research and training.

## 🚀 Features

- **High-Performance Engine**: Optimized C++20 core with Python bindings via pybind11
- **Reinforcement Learning Ready**: Gym-like interface with multi-level support
- **Advanced Visualization**: Real-time gameplay visualization with PyGame
- **Curriculum Learning**: Progressive difficulty with 50+ levels across 4 difficulty tiers
- **PPO Implementation**: Complete Proximal Policy Optimization training pipeline
- **Multi-Level Support**: Random, sequential, and curriculum-based level selection

## 📁 Project Structure

```
sokoban-rl/
├── 🏗️ Core Engine
│   ├── sokoban.h/cpp          # C++ game engine
│   ├── python_bindings.cpp    # Python interface
│   └── test_sokoban.cpp       # Unit tests
├── 🧠 Reinforcement Learning
│   ├── sokoban_env.py         # Gym environment wrapper
│   ├── Train_sokoban_ppo.py   # PPO training script
│   ├── sokoban_levels.py      # 50+ level collection
│   └── Model_player.py        # Trained model visualizer
├── 🎨 Visualization
│   ├── visualize_sokoban.py   # Matplotlib visualization
│   ├── rl_visualization_demo.py # Training visualization
│   └── demo_visualization.py  # Demo scripts
└── 📚 Examples & Tests
    ├── example_usage.py       # Basic usage examples
    └── test_improvements.py   # Environment testing
```

## 🛠️ Installation & Setup

### Prerequisites

```bash
# C++ Compiler (C++20 support)
sudo apt install g++-11  # Ubuntu/Debian
brew install gcc         # macOS

# Python dependencies
pip install torch gymnasium pygame matplotlib numpy
pip install pybind11
```

### Building the C++ Engine

```bash
# Compile Python bindings
cd sokoban
make all 

# Run tests
./test_sokoban
```

## 🎮 Quick Start

### Basic Usage

```python
import sokoban_engine

# Create game instance
game = sokoban_engine.Sokoban()

# Basic gameplay loop
obs, reward, done = game.step(game.RIGHT)
print(f"Reward: {reward}, Solved: {done}")
```

### RL Environment Usage

```python
from sokoban_env import OptimizedSokobanEnv

env = OptimizedSokobanEnv(
    max_episode_steps=200,
    level_selection_mode='random'
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info, action_tracker = env.step(action)
```

## 🏋️ Training the Agent

### PPO Training

```bash
python Train_sokoban_ppo.py \
    --num-envs 8 \
    --num-steps 256 \
    --total-timesteps 2000000 \
    --learning-rate 3e-4 \
    --level_mode random
```

**Training Features:**
- 🎯 Multi-environment parallel training
- 📚 Automatic curriculum learning
- 🧠 Fixed network architecture (256 hidden units)
- 📊 Comprehensive training statistics
- 💾 Automatic model checkpointing

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-envs` | 8 | Number of parallel environments |
| `--num-steps` | 256 | Steps per rollout |
| `--total-timesteps` | 2M | Total training steps |
| `--learning-rate` | 3e-4 | Optimizer learning rate |
| `--level_mode` | random | Level selection strategy |

## 🎨 Visualization

### Real-time Gameplay

```bash
python Model_player.py --model-path trained_model.pt --level-mode random
```

**Visualization Features:**
- 🎮 Interactive gameplay with keyboard controls
- 📊 Real-time statistics and performance metrics
- 🗺️ Multi-level progression tracking
- ⚡ Turbo mode for fast evaluation

### Visualization Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/resume |
| `R` | Reset current level |
| `N` | Next level |
| `T` | Toggle turbo mode |
| `+/-` | Adjust simulation speed |
| `ESC` | Quit |

## 📊 Level System

### Difficulty Tiers

- **Easy (8 levels)**: Simple puzzles, 1-2 boxes
- **Medium (10 levels)**: Strategic planning required
- **Hard (8 levels)**: Complex box arrangements
- **Expert (4 levels)**: Advanced puzzle solving
- **Mixed (10 levels)**: Varied layouts and challenges

### Level Selection Modes

- **Random**: Random level selection each episode
- **Sequential**: Progress through levels in order
- **Curriculum**: Adaptive selection based on performance

## 🧠 RL Algorithm Details

### Network Architecture

```python
FixedPPONetwork(
    obs_shape=(200,),      # Flattened grid observation
    num_actions=4,         # UP, DOWN, LEFT, RIGHT
    hidden_size=256        # Optimized for Sokoban
)
```

### Reward Structure

- **Base Reward**: +1.0 for solving the level
- **Progress Reward**: Incremental rewards for box placement
- **Distance Reward**: Shaping based on box-target distances
- **Efficiency Bonus**: Reward for solving in fewer steps
- **Anti-hacking**: Penalties for repetitive/oscillating behavior

## 📈 Performance

- **Engine**: >100,000 steps/second on modern hardware
- **Training**: ~1-2 million steps for competent performance
- **Memory**: Efficient C++ backend with minimal Python overhead

## 🔧 Customization

### Adding Custom Levels

```python
custom_levels = [
    """
    ########
    #  .   #
    # $@$  #
    #  .   #
    ########
    """
]

env = OptimizedSokobanEnv(custom_levels=custom_levels)
```

### Modifying Reward Structure

Override the `_compute_advanced_reward` method in `sokoban_env.py` to implement custom reward functions.

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: No module named 'sokoban_engine'**
   - Ensure C++ bindings are compiled correctly
   - Check Python path includes the build directory

2. **Poor Training Performance**
   - Adjust `--learning-rate` (try 1e-4 to 1e-3)
   - Increase `--num-envs` for more parallel environments
   - Modify network architecture in `FixedPPONetwork`

3. **Visualization Issues**
   - Ensure PyGame and matplotlib are installed
   - Check display permissions for GUI applications
