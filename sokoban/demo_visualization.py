#!/usr/bin/env python3
"""
Quick visualization demo script to showcase Sokoban engine visualization capabilities.
This script creates several demonstration images without requiring user interaction.
"""

import sys
sys.path.append('.')

from visualize_sokoban import SokobanVisualizer, setup_matplotlib_for_plotting
import sokoban_engine
import numpy as np

def create_demo_visualizations():
    """Create demonstration visualizations and save them as images."""
    
    print("🎨 Creating Sokoban visualization demonstrations...")
    
    # 1. Default level visualization
    print("1. Rendering default level...")
    game = sokoban_engine.Sokoban()
    visualizer = SokobanVisualizer(game)
    visualizer.render_static('demo_01_default_level.png')
    
    # 2. After a few moves
    print("2. Rendering after some gameplay...")
    game.step(game.LEFT)   # Move left
    game.step(game.UP)     # Move up  
    game.step(game.RIGHT)  # Move right
    visualizer.render_static('demo_02_after_moves.png')
    
    # 3. Custom level demonstration
    print("3. Creating custom level...")
    custom_level = """
##########
#.   .   #
# $ $ $ $#
#        #
#@       #
#        #
# $ $ $ $#
#.   .   #
##########"""
    
    game.load_level(custom_level)
    visualizer = SokobanVisualizer(game)
    visualizer.render_static('demo_03_custom_level.png')
    
    # 4. Small level for algorithm testing
    print("4. Creating small test level...")
    small_level = """
#####
#@$.#
#####"""
    
    game.load_level(small_level)
    visualizer = SokobanVisualizer(game, cell_size=0.9)
    visualizer.render_static('demo_04_small_level.png')
    
    # 5. Solved state demonstration
    print("5. Creating solved state example...")
    solved_level = """
######
#@*. #
######"""
    
    game.load_level(solved_level)
    visualizer = SokobanVisualizer(game)
    visualizer.render_static('demo_05_solved_state.png')
    
    print("\n✅ All demonstration images created successfully!")
    print("📁 Generated files:")
    files = [
        'demo_01_default_level.png - Default 7x7 level',
        'demo_02_after_moves.png - After some player moves', 
        'demo_03_custom_level.png - Complex custom level',
        'demo_04_small_level.png - Minimal test level',
        'demo_05_solved_state.png - Example solved state'
    ]
    
    for file in files:
        print(f"   • {file}")
    
    # 6. Print ASCII examples
    print("\n📝 ASCII Visualization Examples:")
    print("=" * 60)
    
    # Show default level
    game = sokoban_engine.Sokoban()
    visualizer = SokobanVisualizer(game)
    print("Default Level:")
    visualizer.print_ascii()
    
    # Show custom level
    game.load_level(custom_level)
    visualizer = SokobanVisualizer(game)
    print("Custom Level:")
    visualizer.print_ascii()
    
    print("🎮 Try running: python3 visualize_sokoban.py")
    print("   for interactive gameplay and more demonstrations!")

if __name__ == "__main__":
    create_demo_visualizations()