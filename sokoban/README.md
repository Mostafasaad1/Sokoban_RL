# Sokoban Engine

A high-performance, C++20 Sokoban game engine designed for reinforcement learning applications.

## Features

- **Modern C++20**: Uses `std::span`, `std::ranges`, `constexpr`, and other C++20 features
- **Performance Optimized**: Minimal allocations in game loop, optimized for RL training
- **Python Integration**: Ready for `pybind11` bindings
- **Configurable**: Customizable grid sizes and level loading
- **Complete Game Logic**: Full Sokoban rules implementation
- **Sparse Rewards**: Designed for reinforcement learning with +1.0 reward on completion

## Building

### C++ Only
```bash
# Compile the engine
g++ -std=c++20 -O3 -Wall sokoban.cpp -c -o sokoban.o

# Run tests
g++ -std=c++20 -O3 -Wall test_sokoban.cpp sokoban.cpp -o test_sokoban
./test_sokoban
```

### Python Bindings
```bash
# Install pybind11
pip install pybind11

# Compile Python module
c++ -O3 -Wall -shared -std=c++20 -fPIC `python3 -m pybind11 --includes` \
    python_bindings.cpp sokoban.cpp -o sokoban_engine`python3-config --extension-suffix`

# Run Python example
python3 example_usage.py
```

### Visualization
```bash
# Install visualization dependencies
pip install matplotlib pillow

# Generate demo visualizations
make demo-viz

# Interactive gameplay with graphics
make interactive

# RL training visualization demo
make rl-demo
```

## API Reference

### Core Methods

```cpp
class Sokoban {
public:
    Sokoban();  // Creates default 10x10 level
    Sokoban(int width, int height = 10);
    
    void reset();
    void load_level(const std::string& level_str);
    std::tuple<std::vector<int>, float, bool> step(int action);
    std::vector<int> get_observation() const;
    std::vector<std::vector<int>> get_grid() const;
    bool is_solved() const;
    bool is_valid_action(int action) const;
};
```

### Tile Types
- `WALL = 0`: Impassable wall
- `EMPTY = 1`: Empty floor
- `PLAYER = 2`: Player character
- `BOX = 3`: Moveable box
- `TARGET = 4`: Target location for boxes
- `BOX_ON_TARGET = 5`: Box correctly placed
- `PLAYER_ON_TARGET = 6`: Player standing on target

### Actions
- `UP = 0`: Move up
- `DOWN = 1`: Move down
- `LEFT = 2`: Move left
- `RIGHT = 3`: Move right

## Level Format

Levels are defined using ASCII strings:
```
"########\n"
"# .  @ #\n"
"# $    #\n"
"#   .  #\n"
"########"
```

Symbols:
- `#`: Wall
- ` `: Empty space
- `@`: Player
- `$`: Box
- `.`: Target
- `*`: Box on target
- `+`: Player on target

## Python Usage

```python
import sokoban_engine

# Create game
game = sokoban_engine.Sokoban()

# Game loop
while True:
    obs, reward, done = game.step(action)
    if done:
        game.reset()
```

### Visualization Usage

```python
from visualize_sokoban import SokobanVisualizer
import sokoban_engine

# Create game and visualizer
game = sokoban_engine.Sokoban()
visualizer = SokobanVisualizer(game)

# Render static image
visualizer.render_static('game_state.png')

# ASCII output
visualizer.print_ascii()

# Interactive gameplay
visualizer.interactive_play()

# Animate action sequence
actions = [game.RIGHT, game.DOWN, game.LEFT]
visualizer.animate_sequence(actions, 'gameplay.gif')
```

## Visualization Features

### 🎨 Comprehensive Visualization Suite

The engine includes powerful visualization capabilities for development, debugging, and presentation:

**Static Rendering**
- High-quality matplotlib-based graphics
- Customizable color schemes and symbols
- Automatic legend generation
- Export to PNG/JPG formats

**Animation Support**
- Action sequence animation
- GIF and MP4 export
- Configurable frame rates
- Progress indicators

**Interactive Features**
- Real-time gameplay visualization
- ASCII art rendering for terminal use
- Interactive controls (WASD movement)
- Live game statistics

**RL Integration**
- Training progress visualization
- Episode statistics tracking
- Multi-level difficulty demonstrations
- Performance benchmarking charts

### 🖼️ Visualization Scripts

<filepath>visualize_sokoban.py</filepath> - Main visualization class with full feature set
<filepath>demo_visualization.py</filepath> - Generate demonstration images
<filepath>rl_visualization_demo.py</filepath> - RL training visualization examples

**Quick Start:**
```bash
# Generate demo images
make demo-viz

# Interactive gameplay
make interactive  

# RL training demo
make rl-demo
```

**Features:**
- 🎯 Multiple tile visualization styles
- 📊 Training progress charts
- 🎮 Interactive gameplay modes
- 📹 Animation and recording
- 🔬 Performance benchmarking
- 🎨 Customizable themes and colors

## Performance

Benchmarked at >100,000 steps/second on modern hardware, making it suitable for intensive RL training.

## Design Goals

1. **Minimal Latency**: Optimized for high-frequency RL training
2. **Standard Compliance**: Uses modern C++20 features appropriately
3. **Memory Efficient**: Pre-allocated buffers, minimal dynamic allocation
4. **Deterministic**: Reproducible behavior for scientific research
5. **Extensible**: Clean interface for additional features

## Default Level

The engine includes a simple 7×7 test level:
```
#######
#  .  #
# $ @ #
#     #
# $   #
#  .  #
#######
```

## Testing

Comprehensive unit tests cover:
- Basic game mechanics
- Level loading and parsing
- Performance benchmarking
- Edge cases and error handling

Run tests with:
```bash
./test_sokoban
```