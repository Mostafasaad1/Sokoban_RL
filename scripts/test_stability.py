"""
Quick test to verify the stable training system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_train_sokoban_ppo import StableSokobanNetwork
import torch
import numpy as np

def test_network_stability():
    """Test if the stable network produces bounded values"""
    print("🧪 TESTING NETWORK STABILITY")
    print("=" * 50)
    
    # Create network
    network = StableSokobanNetwork((49,), 4)
    
    # Test with random inputs
    test_inputs = [
        torch.randn(1, 49),           # Single observation
        torch.randn(32, 49),          # Batch of observations
        torch.randn(1, 49) * 10,      # Large values
        torch.randn(1, 49) * 0.01,    # Small values
    ]
    
    for i, test_input in enumerate(test_inputs):
        with torch.no_grad():
            logits, values = network(test_input)
        
        print(f"Test {i+1}: Input shape {test_input.shape}")
        print(f"  Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
        print(f"  Values range: {values.min().item():.3f} to {values.max().item():.3f}")
        
        # Check value bounds
        assert values.min() >= -10.1 and values.max() <= 10.1, f"Values out of bounds in test {i+1}"
        print(f"  ✅ Values within expected bounds [-10, 10]")
    
    print("\n🎉 NETWORK STABILITY TEST PASSED!")

def test_reward_scaling():
    """Test reward scaling consistency"""
    print("\n🧪 TESTING REWARD SCALING")
    print("=" * 50)
    
    from stable_train_sokoban_ppo import StableSokobanEnv
    
    env = StableSokobanEnv()
    obs, _ = env.reset()
    
    reward_sum = 0
    max_reward = -float('inf')
    min_reward = float('inf')
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        
        reward_sum += reward
        max_reward = max(max_reward, reward)
        min_reward = min(min_reward, reward)
        
        if done:
            obs, _ = env.reset()
    
    print(f"Reward statistics over 100 steps:")
    print(f"  Total reward: {reward_sum:.2f}")
    print(f"  Average reward: {reward_sum/100:.3f}")
    print(f"  Max reward: {max_reward:.2f}")
    print(f"  Min reward: {min_reward:.2f}")
    
    # Check if rewards are reasonably scaled
    assert abs(max_reward) < 200, "Rewards too large - scaling may be insufficient"
    print("  ✅ Rewards are reasonably scaled")

if __name__ == "__main__":
    test_network_stability()
    test_reward_scaling()
    print("\n🎉 ALL STABILITY TESTS PASSED!")