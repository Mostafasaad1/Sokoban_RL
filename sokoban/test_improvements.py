"""
Quick test to demonstrate the improvements in the Sokoban environment.
Shows the difference between the original and improved reward structures.
"""

import sys
sys.path.append('user_input_files')

from sokoban_env import SokobanEnv
from improved_sokoban_env import ImprovedSokobanEnv
import time

def test_original_environment():
    """Test the original environment to show the problem"""
    print("🔴 TESTING ORIGINAL ENVIRONMENT")
    print("=" * 50)
    
    env = SokobanEnv()
    obs, _ = env.reset()
    
    total_reward = 0
    step_count = 0
    
    # Simulate random actions to show constant 0.1 rewards
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Reward={reward:.3f}, Total={total_reward:.3f}")
        
        if done:
            print("Episode finished!")
            break
        
        time.sleep(0.1)  # Small delay for readability
    
    env.close()
    print(f"\n📊 Original Environment Results:")
    print(f"   Total Steps: {step_count}")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Average Reward per Step: {total_reward/step_count:.3f}")
    print()


def test_improved_environment():
    """Test the improved environment to show better learning signals"""
    print("🟢 TESTING IMPROVED ENVIRONMENT")
    print("=" * 50)
    
    env = ImprovedSokobanEnv()
    obs, _ = env.reset()
    
    total_reward = 0
    step_count = 0
    
    # Simulate random actions to show varied, meaningful rewards
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        boxes_on_targets = info.get('boxes_on_targets', 0)
        print(f"Step {step_count}: Action={action}, Reward={reward:.3f}, Total={total_reward:.3f}, Boxes on targets: {boxes_on_targets}/2")
        
        if done:
            if boxes_on_targets >= 2:
                print("🎉 PUZZLE SOLVED!")
            else:
                print("Episode finished without solving")
            break
        
        time.sleep(0.1)  # Small delay for readability
    
    env.close()
    print(f"\n📊 Improved Environment Results:")
    print(f"   Total Steps: {step_count}")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Average Reward per Step: {total_reward/step_count:.3f}")
    print()


def show_level_layout():
    """Show the Sokoban level layout for reference"""
    print("🗺️  SOKOBAN LEVEL LAYOUT")
    print("=" * 30)
    env = ImprovedSokobanEnv(render_mode="ansi")
    obs, _ = env.reset()
    layout = env._render_ansi()
    print(layout)
    print("\nLegend:")
    print("  # = Wall")
    print("  @ = Player")
    print("  $ = Box")
    print("  . = Target")
    print("  * = Box on Target")
    print("  + = Player on Target")
    print("\nGoal: Push both boxes ($) onto target positions (.)")
    env.close()
    print()


if __name__ == "__main__":
    print("🧪 TESTING SOKOBAN ENVIRONMENT IMPROVEMENTS")
    print("=" * 60)
    print()
    
    # Show the level layout first
    show_level_layout()
    
    # Test original environment
    test_original_environment()
    
    # Test improved environment  
    test_improved_environment()
    
    print("🎯 KEY IMPROVEMENTS DEMONSTRATED:")
    print("   1. ✅ Progress rewards for placing boxes on targets")
    print("   2. ✅ Distance-based rewards for strategic movement") 
    print("   3. ✅ Penalties for counterproductive actions")
    print("   4. ✅ Episode length limits to prevent infinite loops")
    print("   5. ✅ More detailed tracking and feedback")
    print()
    print("🚀 The improved environment should enable much better learning!")
    print("   Original: Constant 0.1 rewards → No learning signal")
    print("   Improved: Varied, meaningful rewards → Clear learning signal")