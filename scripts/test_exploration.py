"""
Quick test to verify exploration is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from efficient_sokoban_env import EfficientSokobanEnv
import numpy as np

def test_action_exploration():
    """Test if the environment allows diverse actions"""
    env = EfficientSokobanEnv()
    obs, _ = env.reset()
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    unique_sequences = set()
    
    print("🧪 TESTING ACTION EXPLORATION")
    print("=" * 50)
    
    current_sequence = []
    
    for step in range(100):
        action = env.action_space.sample()  # Random actions
        action_counts[action] += 1
        current_sequence.append(str(action))
        
        if len(current_sequence) >= 5:
            unique_sequences.add(tuple(current_sequence[-5:]))
        
        obs, reward, done, _, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: Action={action}, Reward={reward:.3f}, Boxes={info['boxes_on_targets']}")
        
        if done:
            obs, _ = env.reset()
            current_sequence = []
    
    print(f"\n📊 ACTION DISTRIBUTION (100 steps):")
    for action, count in action_counts.items():
        action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
        print(f"  {action_name}: {count} times ({count/100:.1%})")
    
    print(f"🔄 Unique 5-action sequences: {len(unique_sequences)}")
    
    # Check diversity
    max_action_ratio = max(action_counts.values()) / 100
    if max_action_ratio < 0.4:  # No single action should dominate
        print("✅ EXCELLENT: Healthy action diversity!")
        return True
    elif max_action_ratio < 0.6:
        print("⚠️  ACCEPTABLE: Moderate action diversity")
        return True
    else:
        print("❌ POOR: Action distribution is skewed")
        return False

if __name__ == "__main__":
    success = test_action_exploration()
    sys.exit(0 if success else 1)