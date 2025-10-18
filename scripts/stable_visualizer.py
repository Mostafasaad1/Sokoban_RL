"""
Enhanced visualizer for stable trained models
"""

import argparse
import torch
import numpy as np
import time
from efficient_sokoban_env import EfficientSokobanEnv
from stable_train_sokoban_ppo import StableSokobanNetwork
import sokoban_engine as soko

def load_stable_model(model_path, action_dim):
    """Load stable trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Detect model type
    if 'obs_shape' in checkpoint:
        obs_shape = checkpoint['obs_shape']
    else:
        obs_shape = (49,)  # Default for efficient
    
    print(f"🧠 Loading stable model: {model_path}")
    print(f"📐 Observation shape: {obs_shape}")
    print(f"🎮 Action space: {action_dim}")
    print(f"🎯 Curriculum level: {checkpoint.get('curriculum_level', 'Unknown')}")
    
    model = StableSokobanNetwork(obs_shape, action_dim)
    model.load_state_dict(checkpoint['network_state_dict'])
    model.eval()
    
    return model

def run_stable_visualization(model_path, num_episodes=5, render=True):
    """Run visualization with stability analysis"""
    env = EfficientSokobanEnv()
    model = load_stable_model(model_path, env.action_space.n)
    
    success_count = 0
    total_rewards = []
    step_counts = []
    action_distribution = {0:0, 1:0, 2:0, 3:0}
    value_ranges = []
    
    print(f"\n🎮 STABLE VISUALIZATION ({num_episodes} episodes)")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n📺 Episode {episode + 1}")
        if render:
            print(env.render_grid())
        
        episode_values = []
        
        while not done:
            # Preprocess observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action and value
            with torch.no_grad():
                logits, value = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            action_distribution[action] += 1
            episode_values.append(value.item())
            
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
        value_ranges.append((min(episode_values), max(episode_values)))
        
        if info.get('is_solved', False):
            success_count += 1
            status = "✅ SOLVED"
        else:
            status = "❌ TIMEOUT"
        
        print(f"Episode {episode + 1}: {status}")
        print(f"  Steps: {steps} | Reward: {total_reward:.3f}")
        print(f"  Boxes on targets: {info['boxes_on_targets']}/{info['total_targets']}")
        print(f"  Value range: {min(episode_values):.3f} to {max(episode_values):.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 STABLE PERFORMANCE SUMMARY")
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
    
    print(f"\n📈 VALUE STABILITY:")
    avg_min_value = np.mean([v[0] for v in value_ranges])
    avg_max_value = np.mean([v[1] for v in value_ranges])
    print(f"  Average value range: {avg_min_value:.3f} to {avg_max_value:.3f}")
    
    # Check value stability
    if avg_max_value <= 10.0 and avg_min_value >= -10.0:
        print("  ✅ Values are stable and bounded")
    else:
        print("  ⚠️  Values may be unstable")
    
    return success_count / num_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Sokoban Visualizer')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    success_rate = run_stable_visualization(
        args.model, 
        num_episodes=args.episodes, 
        render=not args.no_render
    )
    
    print(f"\n🎯 Overall Success Rate: {success_rate:.1%}")