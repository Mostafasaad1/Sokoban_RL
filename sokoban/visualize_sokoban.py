#!/usr/bin/env python3
"""
Sokoban Game Visualization Script

Provides comprehensive visualization capabilities for the Sokoban engine including:
- Static game state rendering
- Animated gameplay sequences  
- Interactive gameplay with visual feedback
- Custom level visualization
- Game recording and playback

Requirements: matplotlib, numpy, sokoban_engine (compiled C++ module)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional, Callable
import time

try:
    import sokoban_engine
except ImportError:
    print("Error: sokoban_engine module not found!")
    print("Please compile the module first: make sokoban_engine$(python3-config --extension-suffix)")
    exit(1)

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import warnings
    
    # Ensure warnings are printed
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Configure platform-appropriate fonts for cross-platform compatibility
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class SokobanVisualizer:
    """Enhanced Sokoban game visualizer with multiple rendering modes."""
    
    # Color scheme for different tile types
    COLORS = {
        sokoban_engine.Sokoban.WALL: '#8B4513',           # Brown
        sokoban_engine.Sokoban.EMPTY: '#F5F5DC',         # Beige
        sokoban_engine.Sokoban.PLAYER: '#FF6B6B',        # Red
        sokoban_engine.Sokoban.BOX: '#4ECDC4',           # Teal
        sokoban_engine.Sokoban.TARGET: '#45B7D1',        # Blue
        sokoban_engine.Sokoban.BOX_ON_TARGET: '#96CEB4', # Green
        sokoban_engine.Sokoban.PLAYER_ON_TARGET: '#FECA57' # Yellow
    }
    
    # Unicode symbols for text mode
    SYMBOLS = {
        sokoban_engine.Sokoban.WALL: '█',
        sokoban_engine.Sokoban.EMPTY: '·',
        sokoban_engine.Sokoban.PLAYER: '☺',
        sokoban_engine.Sokoban.BOX: '■',
        sokoban_engine.Sokoban.TARGET: '○',
        sokoban_engine.Sokoban.BOX_ON_TARGET: '●',
        sokoban_engine.Sokoban.PLAYER_ON_TARGET: '☻'
    }
    
    def __init__(self, game: sokoban_engine.Sokoban, cell_size: float = 0.8):
        """
        Initialize the visualizer.
        
        Args:
            game: Sokoban game instance
            cell_size: Size of each cell relative to grid unit (0.0 to 1.0)
        """
        self.game = game
        self.cell_size = cell_size
        self.fig = None
        self.ax = None
        self.patches = []
        
    def setup_plot(self, figsize: Tuple[int, int] = (10, 10)) -> Tuple[plt.Figure, plt.Axes]:
        """Setup matplotlib figure and axes."""
        setup_matplotlib_for_plotting()
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.ax.set_xlim(-0.5, self.game.get_width() - 0.5)
        self.ax.set_ylim(-0.5, self.game.get_height() - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Make (0,0) top-left like game grid
        
        # Remove ticks and labels for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add grid lines
        for i in range(self.game.get_width() + 1):
            self.ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
        for i in range(self.game.get_height() + 1):
            self.ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
            
        return self.fig, self.ax
    
    def render_static(self, save_path: Optional[str] = None, show_stats: bool = True) -> None:
        """
        Render current game state as static image.
        
        Args:
            save_path: Path to save image (optional)
            show_stats: Whether to show game statistics
        """
        if self.fig is None:
            self.setup_plot()
        
        # Clear previous patches
        for patch in self.patches:
            patch.remove()
        self.patches.clear()
        
        # Get current grid state
        grid = self.game.get_grid()
        
        # Render each cell
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                cell_type = grid[y][x]
                color = self.COLORS[cell_type]
                
                # Create rectangle for this cell
                rect = patches.Rectangle(
                    (x - self.cell_size/2, y - self.cell_size/2),
                    self.cell_size, self.cell_size,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1
                )
                self.ax.add_patch(rect)
                self.patches.append(rect)
                
                # Add symbols for better clarity
                if cell_type == sokoban_engine.Sokoban.PLAYER:
                    self.ax.text(x, y, '♦', ha='center', va='center', 
                               fontsize=20, color='white', weight='bold')
                elif cell_type == sokoban_engine.Sokoban.BOX:
                    self.ax.text(x, y, '▦', ha='center', va='center', 
                               fontsize=16, color='white', weight='bold')
                elif cell_type == sokoban_engine.Sokoban.TARGET:
                    self.ax.text(x, y, '◎', ha='center', va='center', 
                               fontsize=14, color='white', weight='bold')
                elif cell_type == sokoban_engine.Sokoban.BOX_ON_TARGET:
                    self.ax.text(x, y, '✓', ha='center', va='center', 
                               fontsize=18, color='white', weight='bold')
                elif cell_type == sokoban_engine.Sokoban.PLAYER_ON_TARGET:
                    self.ax.text(x, y, '♦', ha='center', va='center', 
                               fontsize=20, color='blue', weight='bold')
        
        # Add title and stats
        if show_stats:
            px, py = self.game.get_player_position()
            solved = self.game.is_solved()
            title = f"Sokoban Game - Player: ({px},{py}) - {'SOLVED!' if solved else 'Playing'}"
            self.ax.set_title(title, fontsize=14, pad=20)
        
        # Add legend
        self._add_legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Game state saved to {save_path}")
        else:
            plt.show()
    
    def _add_legend(self) -> None:
        """Add legend explaining the tile types."""
        legend_elements = [
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.WALL], label='Wall'),
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.EMPTY], label='Empty'),
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.PLAYER], label='Player'),
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.BOX], label='Box'),
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.TARGET], label='Target'),
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.BOX_ON_TARGET], label='Box on Target'),
            patches.Patch(color=self.COLORS[sokoban_engine.Sokoban.PLAYER_ON_TARGET], label='Player on Target')
        ]
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def print_ascii(self) -> None:
        """Print ASCII representation of current game state."""
        grid = self.game.get_grid()
        px, py = self.game.get_player_position()
        
        print("\n" + "="*50)
        print(f"Sokoban Game State - Player at ({px}, {py})")
        print(f"Solved: {'YES' if self.game.is_solved() else 'NO'}")
        print("="*50)
        
        for row in grid:
            print(''.join(self.SYMBOLS[cell] for cell in row))
        print()
    
    def animate_sequence(self, actions: List[int], save_path: Optional[str] = None, 
                        interval: int = 500) -> None:
        """
        Animate a sequence of actions.
        
        Args:
            actions: List of action integers
            save_path: Path to save animation (optional, .gif or .mp4)
            interval: Milliseconds between frames
        """
        if self.fig is None:
            self.setup_plot()
        
        # Store initial state
        initial_grid = self.game.get_grid()
        states = [np.array(initial_grid)]
        
        # Execute actions and store states
        total_reward = 0
        for action in actions:
            obs, reward, done = self.game.step(action)
            total_reward += reward
            states.append(np.array(self.game.get_grid()))
            if done:
                break
        
        def animate_frame(frame_num):
            # Clear previous patches
            for patch in self.patches:
                patch.remove()
            self.patches.clear()
            
            # Render current state
            grid = states[frame_num]
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    cell_type = grid[y, x]
                    color = self.COLORS[cell_type]
                    
                    rect = patches.Rectangle(
                        (x - self.cell_size/2, y - self.cell_size/2),
                        self.cell_size, self.cell_size,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1
                    )
                    self.ax.add_patch(rect)
                    self.patches.append(rect)
            
            # Update title
            self.ax.set_title(f"Step {frame_num}/{len(states)-1} - Total Reward: {total_reward}", 
                             fontsize=14, pad=20)
        
        # Create animation
        anim = FuncAnimation(self.fig, animate_frame, frames=len(states), 
                           interval=interval, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            else:
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()
    
    def interactive_play(self) -> None:
        """Interactive gameplay with visual updates."""
        print("\n🎮 Interactive Sokoban Game")
        print("Controls: W/A/S/D (move), R (reset), Q (quit)")
        print("Goal: Push all boxes (■) onto targets (○)")
        
        action_map = {
            'w': sokoban_engine.Sokoban.UP,
            'a': sokoban_engine.Sokoban.LEFT, 
            's': sokoban_engine.Sokoban.DOWN,
            'd': sokoban_engine.Sokoban.RIGHT
        }
        
        while True:
            # Show current state
            self.print_ascii()
            
            if self.game.is_solved():
                print("🎉 Congratulations! Level solved!")
                self.render_static()
                break
            
            try:
                action = input("Action (wasd/r/q): ").lower().strip()
                
                if action == 'q':
                    break
                elif action == 'r':
                    self.game.reset()
                    print("Game reset!")
                elif action in action_map:
                    obs, reward, done = self.game.step(action_map[action])
                    if reward > 0:
                        print(f"🎯 Reward earned: {reward}")
                else:
                    print("❌ Invalid action. Use w/a/s/d/r/q")
                    
            except KeyboardInterrupt:
                break
        
        print("Thanks for playing! 👋")

def demo_random_gameplay(num_steps: int = 50) -> None:
    """Demonstrate random gameplay with visualization."""
    game = sokoban_engine.Sokoban()
    visualizer = SokobanVisualizer(game)
    
    print(f"🎲 Random gameplay demo ({num_steps} steps)")
    
    # Generate random actions
    actions = [np.random.randint(0, 4) for _ in range(num_steps)]
    
    # Show initial state
    print("Initial state:")
    visualizer.render_static("initial_state.png")
    
    # Animate the sequence
    visualizer.animate_sequence(actions, "random_gameplay.gif", interval=300)
    
    print("Demo completed! Files saved: initial_state.png, random_gameplay.gif")

def demo_custom_level() -> None:
    """Demonstrate custom level loading and visualization."""
    game = sokoban_engine.Sokoban()
    
    # Create a more complex custom level
    custom_level = """
##########
#        #
# .$   $ #
# .@   . #
# .$   $ #
#        #
##########"""
    
    game.load_level(custom_level)
    visualizer = SokobanVisualizer(game)
    
    print("🏗️  Custom level demonstration")
    visualizer.print_ascii()
    visualizer.render_static("custom_level.png", show_stats=True)
    print("Custom level rendered and saved as custom_level.png")

def benchmark_visualization() -> None:
    """Benchmark visualization performance."""
    game = sokoban_engine.Sokoban()
    visualizer = SokobanVisualizer(game)
    
    print("⚡ Visualization performance benchmark")
    
    num_renders = 100
    start_time = time.time()
    
    for i in range(num_renders):
        # Make a random move
        action = np.random.randint(0, 4)
        game.step(action)
        
        # Render without displaying
        visualizer.setup_plot()
        for patch in visualizer.patches:
            patch.remove()
        visualizer.patches.clear()
        
        grid = game.get_grid()
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                cell_type = grid[y][x]
                color = visualizer.COLORS[cell_type]
                rect = patches.Rectangle((x, y), 1, 1, facecolor=color)
                visualizer.ax.add_patch(rect)
                visualizer.patches.append(rect)
        
        plt.close(visualizer.fig)
        visualizer.fig = None
        
        if i % 20 == 0:
            game.reset()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Rendered {num_renders} frames in {duration:.3f} seconds")
    print(f"Average: {duration/num_renders*1000:.2f} ms per frame")
    print(f"FPS capability: {num_renders/duration:.1f}")

def main():
    """Main function demonstrating various visualization capabilities."""
    print("🎮 Sokoban Visualization Demo")
    print("============================")
    
    while True:
        print("\nSelect demo:")
        print("1. Interactive gameplay")
        print("2. Random gameplay animation")
        print("3. Custom level visualization")
        print("4. Performance benchmark")
        print("5. Static rendering demo")
        print("6. Exit")
        
        try:
            choice = input("Enter choice (1-6): ").strip()
            
            if choice == '1':
                game = sokoban_engine.Sokoban()
                visualizer = SokobanVisualizer(game)
                visualizer.interactive_play()
                
            elif choice == '2':
                demo_random_gameplay()
                
            elif choice == '3':
                demo_custom_level()
                
            elif choice == '4':
                benchmark_visualization()
                
            elif choice == '5':
                game = sokoban_engine.Sokoban()
                visualizer = SokobanVisualizer(game)
                visualizer.render_static("demo_render.png")
                print("Static render saved as demo_render.png")
                
            elif choice == '6':
                print("Goodbye! 👋")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()