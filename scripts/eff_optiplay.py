"""
Enhanced visualizer for trained models
"""

import argparse
import torch
import numpy as np
import time
from efficient_sokoban_env import EfficientSokobanEnv
from efficient_train_sokoban_ppo import AdvancedSokobanNetwork
import sokoban_engine as soko

def load_model(model_path, action_dim):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Detect model type
    if 'obs_shape' in checkpoint:
        obs_shape = checkpoint['obs_shape']
    else:
        obs_shape = (49,)  # Default for efficient
    
    print(f"🧠 Loading model: {model_path}")
    print(f"📐 Observation shape: {obs_shape}")
    print(f"🎮 Action space: {action_dim}")
    
    model = AdvancedSokobanNetwork(obs_shape, action_dim)
    model.load_state_dict(checkpoint['network_state_dict'])
    model.eval()
    
    return model

def run_visualization(model_path, num_episodes=5, render=True):
    """Run visualization with performance analysis"""
    env = EfficientSokobanEnv()
    model = load_model(model_path, env.action_space.n)
    
    success_count = 0
    total_rewards = []
    step_counts = []
    action_distribution = {0:0, 1:0, 2:0, 3:0}
    
    print(f"\n🎮 STARTING VISUALIZATION ({num_episodes} episodes)")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n📺 Episode {episode + 1}")
        if render:
            print(env.render_grid())
        
        while not done:
            # Preprocess observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                logits, value = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            action_distribution[action] += 1
            
            # Step environment
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render
            if render and (steps % 10 == 0 or done):
                action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                print(f"Step {steps}: {action_names[action]} (Value: {value.item():.3f})")
                print(env.render_grid())
                print(f"Reward: {reward:.3f} | Total: {total_reward:.3f}")
                print(f"Boxes on targets: {info['boxes_on_targets']}/{info['total_targets']}")
                time.sleep(0.1)
            
            if steps >= 200:
                done = True
        
        # Episode results
        total_rewards.append(total_reward)
        step_counts.append(steps)
        
        if info.get('is_solved', False):
            success_count += 1
            status = "✅ SOLVED"
        else:
            status = "❌ TIMEOUT"
        
        print(f"Episode {episode + 1}: {status}")
        print(f"  Steps: {steps} | Reward: {total_reward:.3f}")
        print(f"  Boxes on targets: {info['boxes_on_targets']}/{info['total_targets']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes:.1%})")
    print(f"Average Reward: {np.mean(total_rewards):.3f}")
    print(f"Average Steps: {np.mean(step_counts):.1f}")
    
    print(f"\n🎯 ACTION DISTRIBUTION:")
    total_actions = sum(action_distribution.values())
    for action, count in action_distribution.items():
        action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
        percentage = count / total_actions
        print(f"  {action_name}: {count} ({percentage:.1%})")
    
    return success_count / num_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Sokoban Visualizer')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    success_rate = run_visualization(
        args.model, 
        num_episodes=args.episodes, 
        render=not args.no_render
    )
    
    print(f"\n🎯 Overall Success Rate: {success_rate:.1%}")