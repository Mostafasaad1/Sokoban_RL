"""
EFFICIENT Sokoban Evaluation Script
==================================

This evaluation script works with the EFFICIENT training system that uses
true 7x7 grid observations (49 elements) without padding waste.

FEATURES:
- Matches EfficientPPONetwork architecture exactly
- Loads models trained with efficient system
- Visualizes true 7x7 grid
- Compatible with both efficient and legacy models
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# Import the EFFICIENT environment and network
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from efficient_sokoban_env import EfficientSokobanEnv

class EfficientPPONetwork(nn.Module):
    """
    EFFICIENT PPO network for true 7x7 grid (49 elements).
    Must match the training architecture exactly!
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 512):
        super().__init__()
        
        # TRUE INPUT SIZE: 49 elements (7x7 grid)
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        # Efficient shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)  # Prevent overfitting
        )
        
        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Shared processing
        shared_out = self.shared(x.float())
        
        # Policy and value outputs
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        
        return logits, value.squeeze(-1)

# Legacy network for backward compatibility
class AdvancedPPONetwork(nn.Module):
    """
    Legacy network from optimized system (for loading old models).
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 768):
        super().__init__()
        
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        # Shared feature extractor with larger capacity
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Value head (critic)
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

def load_model(model_path: str, device: torch.device, env):
    """
    Smart model loader that handles both efficient and legacy models.
    """
    print(f"🔍 Loading model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
        
        # Check if it's an efficient model (with metadata)
        if 'obs_shape' in checkpoint and 'num_actions' in checkpoint:
            obs_shape = checkpoint['obs_shape']
            num_actions = checkpoint['num_actions']
            print(f"📐 Model metadata found - Obs shape: {obs_shape}, Actions: {num_actions}")
            
            # Use efficient network
            model = EfficientPPONetwork(obs_shape, num_actions).to(device)
            model.load_state_dict(checkpoint['network_state_dict'])
            print(f"🧠 Loaded EFFICIENT model (input size: {obs_shape[0]})")
            return model, obs_shape
            
        else:
            # Try to detect model type from state dict
            state_dict = checkpoint.get('network_state_dict', checkpoint)
            
            # Check first layer input size
            if 'shared.0.weight' in state_dict:
                first_layer_shape = state_dict['shared.0.weight'].shape
                input_size = first_layer_shape[1]  # Input dimension
                
                print(f"🔍 Detected input size from model: {input_size}")
                
                if input_size == 49:
                    # Efficient model
                    obs_shape = (49,)
                    model = EfficientPPONetwork(obs_shape, env.action_space.n).to(device)
                    print(f"🧠 Loading as EFFICIENT model (49 inputs)")
                elif input_size == 100:
                    # Legacy model with padding - need to adapt
                    print(f"⚠️ WARNING: Legacy padded model detected (100 inputs)")
                    print(f"🔄 This evaluation script is for EFFICIENT models (49 inputs)")
                    print(f"💡 Use 'corrected_evaluate_optimized.py' for legacy models")
                    return None, None
                else:
                    # Unknown model
                    print(f"❌ Unknown model input size: {input_size}")
                    return None, None
                
                model.load_state_dict(state_dict)
                return model, obs_shape
            else:
                print(f"❌ Unknown model format - missing 'shared.0.weight'")
                return None, None
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def evaluate_model(model, env, device, num_episodes=10, render=False):
    """
    Evaluate the model on the environment.
    """
    model.eval()
    
    total_rewards = []
    total_steps = []
    success_count = 0
    
    print(f"\n🎮 Starting evaluation for {num_episodes} episodes...")
    print("="*60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        if render:
            print(f"\n📺 Episode {episode + 1} - Initial State:")
            print(env.render_grid())
        
        while True:
            # Get action from model
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value = model(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if render:
                action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                print(f"\n🎯 Step {episode_steps}: Action {action_names[action]} (value: {value.item():.3f})")
                print(env.render_grid())
                print(f"💰 Reward: {reward:.3f} | Total: {episode_reward:.3f}")
                
                if done or truncated:
                    if info.get('is_solved', False):
                        print(f"🎉 LEVEL COMPLETED! Final reward: {episode_reward:.3f}")
                    else:
                        print(f"❌ Episode ended. Final reward: {episode_reward:.3f}")
            
            if done or truncated:
                break
        
        # Record episode stats
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        if info.get('is_solved', False):
            success_count += 1
        
        if not render:
            success_str = "✅" if info.get('is_solved', False) else "❌"
            print(f"Episode {episode + 1:2d}: {success_str} Reward: {episode_reward:7.2f} | Steps: {episode_steps:3d} | Boxes: {info.get('boxes_on_targets', 0)}/{info.get('total_targets', 0)}")
    
    # Final statistics
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"🏆 Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes:.1%})")
    print(f"💰 Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"👟 Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"🎯 Best Reward: {max(total_rewards):.3f}")
    print(f"⚡ Steps/Success: {np.mean([s for i, s in enumerate(total_steps) if i < success_count]) if success_count > 0 else 'N/A'}")
    print("="*60)
    
    return {
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'total_episodes': num_episodes
    }

def main():
    parser = argparse.ArgumentParser(description='EFFICIENT Sokoban Model Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render episodes visually')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("🚀 EFFICIENT SOKOBAN EVALUATION")
    print("="*50)
    print(f"🖥️ Device: {device}")
    print(f"📁 Model: {args.model}")
    print(f"🎮 Episodes: {args.episodes}")
    print(f"👁️ Render: {args.render}")
    print("="*50)
    
    # Create environment
    env = EfficientSokobanEnv()
    print(f"🌍 Environment created - Obs shape: {env.observation_space.shape}")
    
    # Load model
    model, obs_shape = load_model(args.model, device, env)
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Verify compatibility
    env_obs_shape = env.observation_space.shape
    if obs_shape != env_obs_shape:
        print(f"⚠️ WARNING: Model obs shape {obs_shape} != Environment obs shape {env_obs_shape}")
        print(f"🔄 Environment will adapt automatically")
    
    # Evaluate
    results = evaluate_model(model, env, device, args.episodes, args.render)
    
    print(f"\n✅ Evaluation complete!")
    
    # Save results if requested
    if args.episodes >= 10:
        results_str = f"""
EFFICIENT SOKOBAN EVALUATION RESULTS
====================================
Model: {args.model}
Episodes: {args.episodes}
Success Rate: {results['success_rate']:.1%}
Average Reward: {results['avg_reward']:.3f}
Average Steps: {results['avg_steps']:.1f}
Device: {device}
Environment: Efficient (49 elements, no padding)
====================================
"""
        print(results_str)

if __name__ == '__main__':
    main()