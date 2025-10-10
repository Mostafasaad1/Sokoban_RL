import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple

# Import the optimized environment (assumes it's in the same directory)
from optimized_sokoban_env import OptimizedSokobanEnv

class AdvancedPPONetwork(nn.Module):
    """
    Advanced PPO network with larger capacity for complex Sokoban reasoning.
    Handles flat 100-element observations like the original environment.
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 768):
        super().__init__()
        
        # Input size should be flat observation (100 elements)
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        # Shared feature extractor with larger capacity
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

class CurriculumManager:
    """
    Manages curriculum learning progression based on success rate.
    """
    
    def __init__(self, success_threshold: float = 0.15, evaluation_window: int = 100):
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.current_difficulty = 1
        self.success_history = deque(maxlen=evaluation_window)
        self.episodes_at_current_level = 0
        self.min_episodes_per_level = 200  # Minimum episodes before considering upgrade
        
    def update(self, success: bool) -> bool:
        """Update curriculum based on episode success. Returns True if difficulty changed."""
        self.success_history.append(success)
        self.episodes_at_current_level += 1
        
        # Only consider progression after minimum episodes and full evaluation window
        if (len(self.success_history) >= self.evaluation_window and 
            self.episodes_at_current_level >= self.min_episodes_per_level):
            
            success_rate = np.mean(self.success_history)
            
            # Upgrade difficulty if success rate is high enough
            if success_rate >= self.success_threshold and self.current_difficulty < 3:
                self.current_difficulty += 1
                self.episodes_at_current_level = 0
                self.success_history.clear()
                print(f"\\n🚀 CURRICULUM UPGRADE! Difficulty increased to level {self.current_difficulty}")
                print(f"   Previous success rate: {success_rate:.1%}")
                return True
                
            # Downgrade if success rate is too low (except for level 1)
            elif success_rate < 0.05 and self.current_difficulty > 1:
                self.current_difficulty -= 1
                self.episodes_at_current_level = 0
                self.success_history.clear()
                print(f"\\n📉 Curriculum downgrade. Difficulty reduced to level {self.current_difficulty}")
                print(f"   Previous success rate: {success_rate:.1%}")
                return True
        
        return False

class OptimizedPPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Create vectorized environments
        self.envs = [OptimizedSokobanEnv(
            max_episode_steps=400,  # Longer episodes
            curriculum_mode=False,  # Start with default level
            difficulty_level=1,
            anti_hacking_strength=2.0
        ) for _ in range(args.num_envs)]
        
        # Initialize curriculum manager (for future use)
        self.curriculum = CurriculumManager(
            success_threshold=0.15,  # 15% success rate to advance
            evaluation_window=100
        )
        
        # Create network
        obs_shape = self.envs[0].observation_space.shape
        num_actions = self.envs[0].action_space.n
        self.network = AdvancedPPONetwork(obs_shape, num_actions, hidden_size=768).to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(self.network.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.total_timesteps // (args.num_envs * args.num_steps), eta_min=1e-6
        )
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.boxes_on_targets = deque(maxlen=100)
        
    def collect_rollouts(self) -> Dict:
        """Collect rollouts from all environments"""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        # Reset environments
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        
        for step in range(self.args.num_steps):
            # Convert observations to tensor
            obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
            
            # Get actions from policy
            with torch.no_grad():
                logits, value = self.network(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Store data
            observations.append(obs_tensor.cpu())
            actions.append(action.cpu())
            values.append(value.cpu())
            log_probs.append(log_prob.cpu())
            
            # Step environments
            next_obs = []
            step_rewards = []
            step_dones = []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, truncated, info = env.step(action[i].item())
                
                # Handle episode completion
                if done or truncated:
                    # Track statistics
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(info['step_count'])
                    self.success_rates.append(float(info['is_solved']))
                    self.boxes_on_targets.append(info['boxes_on_targets'] / max(1, info['total_targets']))
                    
                    # Update curriculum (for future use)
                    difficulty_changed = self.curriculum.update(info['is_solved'])
                    if difficulty_changed:
                        env.set_difficulty(self.curriculum.current_difficulty)
                    
                    # Reset environment
                    obs, _ = env.reset()
                
                next_obs.append(obs)
                step_rewards.append(reward)
                step_dones.append(done or truncated)
            
            rewards.append(torch.FloatTensor(step_rewards))
            dones.append(torch.BoolTensor(step_dones))
            obs_list = next_obs
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(
            rewards, values, dones
        )
        
        return {
            'observations': torch.cat(observations),
            'actions': torch.cat(actions),
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': torch.cat(log_probs),
            'old_values': torch.cat(values)
        }
    
    def _compute_gae(self, rewards: List[torch.Tensor], values: List[torch.Tensor], 
                     dones: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        gamma = 0.99
        gae_lambda = 0.95
        
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = torch.zeros_like(values[t])
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (~dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (~dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update_policy(self, rollout_data: Dict):
        """Update policy using PPO"""
        batch_size = 256
        num_updates = 4
        
        data_size = rollout_data['observations'].shape[0]
        indices = torch.randperm(data_size)
        
        for _ in range(num_updates):
            for start_idx in range(0, data_size, batch_size):
                end_idx = min(start_idx + batch_size, data_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                obs_batch = rollout_data['observations'][batch_indices].to(self.device)
                actions_batch = rollout_data['actions'][batch_indices].to(self.device)
                returns_batch = rollout_data['returns'][batch_indices].to(self.device)
                advantages_batch = rollout_data['advantages'][batch_indices].to(self.device)
                old_log_probs_batch = rollout_data['old_log_probs'][batch_indices].to(self.device)
                
                # Forward pass
                logits, values = self.network(obs_batch)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_batch)
                entropy = dist.entropy()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                
                # Entropy bonus (increased for more exploration)
                entropy_loss = -0.02 * entropy.mean()  # Higher entropy coefficient
                
                # Total loss
                total_loss = policy_loss + value_loss + entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def train(self):
        """Main training loop"""
        print("\\n🚀 Starting Optimized Sokoban PPO Training")
        print(f"📊 Environments: {self.args.num_envs}")
        print(f"🎯 Target timesteps: {self.args.total_timesteps:,}")
        print(f"🧠 Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print("\\n" + "="*80)
        
        global_step = 0
        num_updates = self.args.total_timesteps // (self.args.num_envs * self.args.num_steps)
        
        for update in range(num_updates):
            start_time = time.time()
            
            # Collect rollouts
            rollout_data = self.collect_rollouts()
            global_step += self.args.num_envs * self.args.num_steps
            
            # Update policy
            train_stats = self.update_policy(rollout_data)
            
            # Logging
            if update % 20 == 0:
                fps = (self.args.num_envs * self.args.num_steps) / (time.time() - start_time)
                
                print(f"\\n{'~'*60}")
                print(f"📈 Step {global_step:,} (Update {update})")
                print(f"🎭 Policy Loss: {train_stats['policy_loss']:.6f}")
                print(f"💎 Value Loss: {train_stats['value_loss']:.6f}")
                print(f"🎲 Entropy Loss: {train_stats['entropy_loss']:.6f}")
                print(f"🧠 Learning Rate: {train_stats['learning_rate']:.6f}")
                print(f"⚡ FPS: {fps:.0f}")
                print(f"🎯 Curriculum Level: {self.curriculum.current_difficulty}")
                
                if len(self.episode_rewards) > 0:
                    avg_reward = np.mean(list(self.episode_rewards)[-50:])
                    avg_length = np.mean(list(self.episode_lengths)[-50:])
                    success_rate = np.mean(list(self.success_rates)[-50:])
                    avg_boxes = np.mean(list(self.boxes_on_targets)[-50:])
                    
                    print(f"💰 Avg Reward (last 50): {avg_reward:.3f}")
                    print(f"📏 Avg Length (last 50): {avg_length:.1f}")
                    print(f"🏆 Success Rate (last 50): {success_rate:.1%}")
                    print(f"📦 Avg Boxes on Targets: {avg_boxes:.2f}")
                
                print(f"{'~'*60}")
            
            # Save model periodically
            if update % 100 == 0 and update > 0:
                torch.save({
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': global_step,
                    'curriculum_level': self.curriculum.current_difficulty
                }, f'optimized_sokoban_model_{global_step}.pt')
                print(f"💾 Model saved at step {global_step:,}")
        
        # Final summary
        print("\\n" + "="*80)
        print("🎉 OPTIMIZED TRAINING COMPLETE")
        if len(self.success_rates) > 0:
            final_success_rate = np.mean(list(self.success_rates)[-100:])
            final_avg_reward = np.mean(list(self.episode_rewards)[-100:])
            final_avg_boxes = np.mean(list(self.boxes_on_targets)[-100:])
            
            print(f"📊 Total Episodes: {len(self.success_rates):,}")
            print(f"🏆 Final Success Rate (last 100): {final_success_rate:.1%}")
            print(f"💰 Final Avg Reward (last 100): {final_avg_reward:.3f}")
            print(f"📦 Final Avg Boxes on Targets: {final_avg_boxes:.2f}")
            print(f"🎯 Final Curriculum Level: {self.curriculum.current_difficulty}")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Optimized Sokoban PPO Training')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=256, help='Steps per rollout')
    parser.add_argument('--total-timesteps', type=int, default=2000000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    trainer = OptimizedPPOTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()