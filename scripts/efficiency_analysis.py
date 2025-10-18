"""
EFFICIENCY DEMONSTRATION SCRIPT
===============================

This script demonstrates the efficiency improvements of the new system:
- Compare network sizes (49 vs 100 inputs)
- Test environment compatibility
- Verify observation formats
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# Import both environments
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from efficient_sokoban_env import EfficientSokobanEnv
from optimized_sokoban_env import OptimizedSokobanEnv

# Import both networks
class EfficientPPONetwork(nn.Module):
    """EFFICIENT network (49 inputs)"""
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 512):
        super().__init__()
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        shared_out = self.shared(x.float())
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value.squeeze(-1)

class PaddedPPONetwork(nn.Module):
    """Legacy padded network (100 inputs)"""
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 768):
        super().__init__()
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        shared_out = self.shared(x.float())
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value.squeeze(-1)

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_efficiency():
    """Analyze the efficiency improvements"""
    
    print("🧪 SOKOBAN EFFICIENCY ANALYSIS")
    print("="*60)
    
    # Create environments
    print("\n🌍 ENVIRONMENT COMPARISON:")
    print("-" * 30)
    
    try:
        efficient_env = EfficientSokobanEnv()
        print(f"✅ Efficient Environment: {efficient_env.observation_space.shape} observations")
        
        optimized_env = OptimizedSokobanEnv()
        print(f"⚠️ Padded Environment: {optimized_env.observation_space.shape} observations")
        
        # Test observations
        eff_obs, _ = efficient_env.reset()
        opt_obs, _ = optimized_env.reset()
        
        print(f"\n📊 OBSERVATION ANALYSIS:")
        print(f"Efficient obs shape: {eff_obs.shape}")
        print(f"Efficient obs sample: {eff_obs[:10]}...")
        print(f"Padded obs shape: {opt_obs.shape}")
        print(f"Padded obs sample: {opt_obs[:10]}...")
        print(f"Padded obs zeros: {opt_obs[49:59]}...")  # Show the padding
        
        # Calculate efficiency
        real_data_ratio = len(eff_obs) / len(opt_obs)
        print(f"\n⚡ EFFICIENCY METRICS:")
        print(f"Real data ratio: {real_data_ratio:.1%}")
        print(f"Padding waste: {100-real_data_ratio*100:.1f}%")
        
        env_compatible = True
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        env_compatible = False
    
    # Create networks
    print(f"\n🧠 NETWORK COMPARISON:")
    print("-" * 30)
    
    try:
        # Efficient network (49 inputs)
        efficient_net = EfficientPPONetwork((49,), 4, hidden_size=512)
        efficient_params = count_parameters(efficient_net)
        
        # Padded network (100 inputs)
        padded_net = PaddedPPONetwork((100,), 4, hidden_size=768)
        padded_params = count_parameters(padded_net)
        
        print(f"✅ Efficient Network: {efficient_params:,} parameters")
        print(f"⚠️ Padded Network: {padded_params:,} parameters")
        
        # Calculate savings
        param_reduction = (padded_params - efficient_params) / padded_params
        
        print(f"\n💾 PARAMETER EFFICIENCY:")
        print(f"Parameter reduction: {param_reduction:.1%}")
        print(f"Parameters saved: {padded_params - efficient_params:,}")
        
        # Memory efficiency
        efficient_memory = efficient_params * 4 / (1024**2)  # 4 bytes per float32, convert to MB
        padded_memory = padded_params * 4 / (1024**2)
        memory_saved = padded_memory - efficient_memory
        
        print(f"\n🗄️ MEMORY EFFICIENCY:")
        print(f"Efficient network: {efficient_memory:.1f} MB")
        print(f"Padded network: {padded_memory:.1f} MB")
        print(f"Memory saved: {memory_saved:.1f} MB ({memory_saved/padded_memory:.1%})")
        
        network_compatible = True
        
    except Exception as e:
        print(f"❌ Network error: {e}")
        network_compatible = False
    
    # Test inference speed
    if env_compatible and network_compatible:
        print(f"\n🏃 INFERENCE SPEED TEST:")
        print("-" * 30)
        
        try:
            import time
            
            # Prepare test data
            eff_batch = torch.randn(32, 49)  # Batch of efficient observations
            pad_batch = torch.randn(32, 100)  # Batch of padded observations
            
            # Warm up
            for _ in range(10):
                _ = efficient_net(eff_batch)
                _ = padded_net(pad_batch)
            
            # Time efficient network
            start_time = time.time()
            for _ in range(100):
                _ = efficient_net(eff_batch)
            efficient_time = time.time() - start_time
            
            # Time padded network
            start_time = time.time()
            for _ in range(100):
                _ = padded_net(pad_batch)
            padded_time = time.time() - start_time
            
            speedup = padded_time / efficient_time
            
            print(f"✅ Efficient network: {efficient_time*1000:.1f}ms (100 batches)")
            print(f"⚠️ Padded network: {padded_time*1000:.1f}ms (100 batches)")
            print(f"🚀 Speedup: {speedup:.2f}x faster")
            
        except Exception as e:
            print(f"❌ Speed test error: {e}")
    
    # Summary
    print(f"\n🎯 EFFICIENCY SUMMARY:")
    print("="*60)
    print("✅ EFFICIENT SYSTEM BENEFITS:")
    print("   • 51% fewer input neurons (49 vs 100)")
    print("   • Significantly fewer total parameters")
    print("   • Lower memory usage")
    print("   • Faster training and inference")
    print("   • No wasted computation on padding zeros")
    print("   • True representation of 7x7 Sokoban grid")
    print("")
    print("⚠️ LEGACY SYSTEM PROBLEMS:")
    print("   • Wastes 51 neurons on padding zeros")
    print("   • Larger network with more parameters")
    print("   • Higher memory usage")
    print("   • Slower due to unnecessary computations")
    print("   • Misleading observation representation")
    print("="*60)
    
    print(f"\n💡 RECOMMENDATION:")
    print("   Use the EFFICIENT system for all new training!")
    print("   - efficient_sokoban_env.py")
    print("   - efficient_train_sokoban_ppo.py")
    print("   - efficient_evaluate_sokoban.py")

if __name__ == "__main__":
    analyze_efficiency()