"""
TRAINING IMPROVEMENT PATCH
=========================

Based on your training output, I see the agent is struggling. Here are the specific fixes:

PROBLEMS IDENTIFIED:
- 0.0% success rate (agent timing out at 200 steps)
- Poor reward signal (-9.99 average = step penalties)
- Very low boxes on targets (0.00-0.06)
- Agent not learning core Sokoban mechanics

SOLUTIONS PROVIDED:
1. Better reward shaping for learning progression
2. Curriculum starting with easier scenarios
3. Improved network architecture for Sokoban reasoning
4. Better exploration incentives
5. Debugging tools to monitor learning
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Any

# Import the efficient environment
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from efficient_sokoban_env import EfficientSokobanEnv
import sokoban_engine as soko


class ImprovedEfficientPPONetwork(nn.Module):
    """
    IMPROVED Efficient PPO network with better Sokoban-specific architecture.
    Designed to learn spatial reasoning more effectively.
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 512):
        super().__init__()
        
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        print(f"🧠 Improved network input size: {input_size}")
        
        # IMPROVED: Better feature extraction for spatial reasoning
        self.spatial_features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Add normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # IMPROVED: Separate reasoning pathways
        self.action_reasoning = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_actions)
        )
        
        self.value_reasoning = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights better for Sokoban
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔢 Improved network parameters: {total_params:,}")
        
    def _initialize_weights(self):
        """Better weight initialization for Sokoban learning"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Extract spatial features
        features = self.spatial_features(x.float())
        
        # Separate reasoning for actions and values
        logits = self.action_reasoning(features)
        value = self.value_reasoning(features)
        
        return logits, value.squeeze(-1)

class ImprovedCurriculumManager:
    """
    IMPROVED curriculum with better progression logic for Sokoban learning.
    """
    
    def __init__(self, success_threshold: float = 0.1, evaluation_window: int = 100):
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.current_difficulty = 1
        self.success_history = deque(maxlen=evaluation_window)
        self.reward_history = deque(maxlen=evaluation_window)
        self.episodes_at_current_level = 0
        self.min_episodes_per_level = 500  # INCREASED: More practice per level
        
    def update(self, success: bool, episode_reward: float) -> bool:
        """Update curriculum based on both success and reward progress"""
        self.success_history.append(success)
        self.reward_history.append(episode_reward)
        self.episodes_at_current_level += 1
        
        # Only consider progression after minimum episodes and full evaluation window
        if (len(self.success_history) >= self.evaluation_window and 
            self.episodes_at_current_level >= self.min_episodes_per_level):
            
            success_rate = np.mean(self.success_history)
            avg_reward = np.mean(self.reward_history)
            
            # IMPROVED: Consider both success rate AND reward improvement
            upgrade_threshold = max(self.success_threshold, 0.05)  # At least 5% success
            
            # Upgrade if doing well
            if success_rate >= upgrade_threshold and self.current_difficulty < 3 and avg_reward > -5:
                self.current_difficulty += 1
                self.episodes_at_current_level = 0
                self.success_history.clear()
                self.reward_history.clear()
                print(f"\n🚀 CURRICULUM UPGRADE! Level {self.current_difficulty}")
                print(f"   Success rate: {success_rate:.1%} | Avg reward: {avg_reward:.1f}")
                return True
                
            # Downgrade if struggling badly
            elif success_rate < 0.02 and self.current_difficulty > 1 and avg_reward < -15:
                self.current_difficulty -= 1
                self.episodes_at_current_level = 0
                self.success_history.clear()
                self.reward_history.clear()
                print(f"\n📉 CURRICULUM DOWNGRADE! Level {self.current_difficulty}")
                print(f"   Success rate: {success_rate:.1%} | Avg reward: {avg_reward:.1f}")
                return True
        
        return False

class ImprovedSokobanEnv(EfficientSokobanEnv):
    """
    IMPROVED Sokoban environment with better reward shaping for learning.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("🎮 IMPROVED Sokoban Environment with Better Reward Shaping")
        
        # Learning progress tracking
        self.best_boxes_on_targets = 0
        self.exploration_bonus_used = set()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Store state before action
        prev_boxes_on_targets = self.boxes_on_targets()
        prev_player_pos = self._get_player_position()
        
        # Execute action with base environment
        obs_data, base_reward, done, pushed_box = self.game.step({
            0: soko.UP, 1: soko.DOWN, 2: soko.LEFT, 3: soko.RIGHT
        }[action])
        
        # Update tracking
        self.step_count += 1
        current_boxes_on_targets = self.boxes_on_targets()
        current_player_pos = self._get_player_position()
        
        # IMPROVED REWARD SHAPING
        reward_components = {}
        reward = 0.0
        
        # 1. CORE PROGRESS REWARDS (most important)
        target_progress = current_boxes_on_targets - prev_boxes_on_targets
        if target_progress > 0:
            progress_bonus = target_progress * 20.0  # INCREASED: Bigger bonus for progress
            reward += progress_bonus
            reward_components['target_progress'] = progress_bonus
            print(f"🎯 BOX ON TARGET! +{progress_bonus}")
        elif target_progress < 0:
            regression_penalty = target_progress * 10.0  # Less harsh penalty
            reward += regression_penalty
            reward_components['target_regression'] = regression_penalty
        
        # 2. MILESTONE REWARDS
        if current_boxes_on_targets > self.best_boxes_on_targets:
            self.best_boxes_on_targets = current_boxes_on_targets
            milestone_bonus = 10.0
            reward += milestone_bonus
            reward_components['milestone'] = milestone_bonus
            print(f"🏆 NEW BEST: {current_boxes_on_targets} boxes on targets! +{milestone_bonus}")
        
        # 3. EXPLORATION BONUS (encourage visiting new positions)
        pos_key = f"{current_player_pos[0]},{current_player_pos[1]}"
        if pos_key not in self.exploration_bonus_used:
            self.exploration_bonus_used.add(pos_key)
            if len(self.exploration_bonus_used) <= 20:  # Only for first 20 unique positions
                exploration_bonus = 0.5
                reward += exploration_bonus
                reward_components['exploration'] = exploration_bonus
        
        # 4. SMALL POSITIVE STEP REWARD (prevent immediate timeout)
        step_reward = 0.05  # Small positive to encourage exploration
        reward += step_reward
        reward_components['step_reward'] = step_reward
        
        # 5. BOX INTERACTION REWARD
        if pushed_box:
            interaction_bonus = 2.0  # Reward for pushing boxes
            reward += interaction_bonus
            reward_components['box_interaction'] = interaction_bonus
        
        # 6. COMPLETION REWARD
        if done and self.game.is_solved():
            completion_bonus = 200.0 + max(0, 300 - self.step_count) * 1.0  # Big completion bonus
            reward += completion_bonus
            reward_components['completion'] = completion_bonus
            self.solved_step = self.step_count
            print(f"🎉 LEVEL COMPLETED! +{completion_bonus} (Steps: {self.step_count})")
        
        # 7. TIMEOUT PENALTY (only if no progress made)
        if self.step_count >= 200:
            done = True
            if not self.game.is_solved():
                # Less harsh timeout penalty if some progress was made
                if self.best_boxes_on_targets > 0:
                    timeout_penalty = -5.0  # Mild penalty if made some progress
                else:
                    timeout_penalty = -20.0  # Harsher if no progress
                reward += timeout_penalty
                reward_components['timeout'] = timeout_penalty
        
        self.total_reward += reward
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info(reward_components)
        info['best_boxes_on_targets'] = self.best_boxes_on_targets
        
        return obs, reward, done, False, info
    
    def _get_player_position(self):
        """Get current player position"""
        grid = self.game.get_grid()
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == self.PLAYER or cell == self.PLAYER_ON_TARGET:
                    return (x, y)
        return (0, 0)  # Fallback
    
    def reset(self, seed=None, options=None):
        """Reset with improved tracking"""
        result = super().reset(seed, options)
        self.best_boxes_on_targets = 0
        self.exploration_bonus_used = set()
        return result

class ImprovedPPOTrainer:
    """
    IMPROVED PPO trainer with better hyperparameters for Sokoban learning.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Create IMPROVED environments
        self.envs = [ImprovedSokobanEnv() for _ in range(args.num_envs)]
        
        # IMPROVED network
        obs_shape = self.envs[0].observation_space.shape
        num_actions = self.envs[0].action_space.n
        
        print(f"\n🔧 IMPROVED NETWORK SETUP:")
        print(f"📐 Observation shape: {obs_shape}")
        print(f"🎮 Number of actions: {num_actions}")
        
        self.network = ImprovedEfficientPPONetwork(obs_shape, num_actions).to(self.device)
        
        # IMPROVED optimizer with better learning rate scheduling
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.95)
        
        # IMPROVED PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.entropy_coef = 0.02  # INCREASED: More exploration
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # IMPROVED curriculum
        self.curriculum = ImprovedCurriculumManager()
        
        # Enhanced tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=1000)
        self.boxes_on_targets = deque(maxlen=1000)
        self.best_boxes_achieved = deque(maxlen=1000)  # Track learning progress
        
        print(f"\n⚡ IMPROVED TRAINER READY!")
        print(f"🎯 Better reward shaping")
        print(f"🧠 Improved network architecture") 
        print(f"📈 Enhanced curriculum learning")
        print(f"🔍 Better progress tracking")
    
    def collect_rollouts(self):
        """Collect rollouts with improved tracking."""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        # Reset environments if needed
        obs_list = []
        for env in self.envs:
            if not hasattr(env, '_current_obs'):
                obs, _ = env.reset()
                env._current_obs = obs
            obs_list.append(env._current_obs)
        
        # Collect steps
        for step in range(self.args.num_steps):
            # Convert observations to tensor
            obs_tensor = torch.FloatTensor(obs_list).to(self.device)
            
            # Get action distribution and value
            with torch.no_grad():
                logits, value = self.network(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Store step data
            observations.append(obs_tensor.cpu())
            actions.append(action.cpu())
            values.append(value.cpu())
            log_probs.append(log_prob.cpu())
            
            # Execute actions in environments
            next_obs_list = []
            step_rewards = []
            step_dones = []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, truncated, info = env.step(action[i].item())
                
                step_rewards.append(reward)
                step_dones.append(done or truncated)
                
                if done or truncated:
                    # Track episode statistics with IMPROVED metrics
                    self.episode_rewards.append(info.get('total_reward', reward))
                    self.episode_lengths.append(info.get('step_count', 1))
                    self.success_rates.append(1.0 if info.get('is_solved', False) else 0.0)
                    self.boxes_on_targets.append(info.get('boxes_on_targets', 0))
                    self.best_boxes_achieved.append(info.get('best_boxes_on_targets', 0))
                    
                    # Update IMPROVED curriculum
                    episode_reward = info.get('total_reward', reward)
                    success = info.get('is_solved', False)
                    self.curriculum.update(success, episode_reward)
                    
                    # Reset environment
                    obs, _ = env.reset()
                
                env._current_obs = obs
                next_obs_list.append(obs)
            
            rewards.append(step_rewards)
            dones.append(step_dones)
            obs_list = next_obs_list
        
        # Get final values for GAE
        with torch.no_grad():
            final_obs_tensor = torch.FloatTensor(obs_list).to(self.device)
            _, final_values = self.network(final_obs_tensor)
            final_values = final_values.cpu()
        
        return {
            'observations': torch.stack(observations),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards),
            'dones': torch.tensor(dones),
            'values': torch.stack(values),
            'log_probs': torch.stack(log_probs),
            'final_values': final_values
        }
    
    def compute_gae(self, rollouts):
        """Compute Generalized Advantage Estimation."""
        rewards = rollouts['rewards']
        values = rollouts['values']
        dones = rollouts['dones']
        final_values = rollouts['final_values']
        
        num_steps, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                print(dones[step])
                nextnonterminal = ~dones[step]
                nextvalues = final_values
            else:
                nextnonterminal = ~dones[step]
                nextvalues = values[step + 1]
            
            delta = rewards[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * nextnonterminal * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        return advantages, returns
    
    def update_policy(self, rollouts, advantages, returns):
        """Update policy using IMPROVED PPO."""
        observations = rollouts['observations'].view(-1, *rollouts['observations'].shape[2:])
        actions = rollouts['actions'].view(-1)
        old_log_probs = rollouts['log_probs'].view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # IMPROVED PPO update with more epochs for better learning
        for epoch in range(4):  # PPO epochs
            # Forward pass
            logits, values = self.network(observations.to(self.device))
            
            # Policy loss
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.to(self.device))
            ratio = torch.exp(new_log_probs - old_log_probs.to(self.device))
            
            surr1 = ratio * advantages.to(self.device)
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages.to(self.device)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values.squeeze(), returns.to(self.device))
            
            # Entropy loss (IMPROVED: higher coefficient for more exploration)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
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
        """Main IMPROVED training loop."""
        total_timesteps = self.args.total_timesteps
        num_updates = total_timesteps // (self.args.num_envs * self.args.num_steps)
        global_step = 0
        
        print(f"\n🚀 STARTING IMPROVED PPO TRAINING")
        print(f"📊 Total timesteps: {total_timesteps:,}")
        print(f"🔄 Number of updates: {num_updates:,}")
        print(f"🌍 Environments: {self.args.num_envs}")
        print(f"👟 Steps per rollout: {self.args.num_steps}")
        print("="*80)
        
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # Collect rollouts
            rollouts = self.collect_rollouts()
            global_step += self.args.num_envs * self.args.num_steps
            
            # Compute advantages
            advantages, returns = self.compute_gae(rollouts)
            
            # Update policy
            loss_info = self.update_policy(rollouts, advantages, returns)
            
            # Update curriculum difficulty for all environments
            for env in self.envs:
                env.set_difficulty(self.curriculum.current_difficulty)
            
            # IMPROVED logging with more useful metrics
            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                
                print(f"\n📈 UPDATE {update:,} | STEP {global_step:,}")
                print(f"⏱️ Time: {elapsed_time:.1f}s | Speed: {steps_per_sec:.0f} steps/s")
                print(f"🎯 Curriculum Level: {self.curriculum.current_difficulty}")
                print(f"📚 Learning Rate: {loss_info['learning_rate']:.6f}")
                print(f"📉 Policy Loss: {loss_info['policy_loss']:.4f}")
                print(f"📊 Value Loss: {loss_info['value_loss']:.4f}")
                print(f"🎲 Entropy Loss: {loss_info['entropy_loss']:.4f}")
                
                if len(self.episode_rewards) > 0:
                    avg_reward = np.mean(list(self.episode_rewards)[-50:])
                    avg_length = np.mean(list(self.episode_lengths)[-50:])
                    success_rate = np.mean(list(self.success_rates)[-50:])
                    avg_boxes = np.mean(list(self.boxes_on_targets)[-50:])
                    avg_best_boxes = np.mean(list(self.best_boxes_achieved)[-50:])
                    
                    print(f"💰 Avg Reward (last 50): {avg_reward:.3f}")
                    print(f"📏 Avg Length (last 50): {avg_length:.1f}")
                    print(f"🏆 Success Rate (last 50): {success_rate:.1%}")
                    print(f"📦 Avg Boxes on Targets: {avg_boxes:.2f}")
                    print(f"🎯 Avg Best Boxes Achieved: {avg_best_boxes:.2f}")  # NEW METRIC
                
                print(f"{'~'*60}")
            
            # Save model periodically
            if update % 100 == 0 and update > 0:
                torch.save({
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'global_step': global_step,
                    'curriculum_level': self.curriculum.current_difficulty,
                    'obs_shape': self.envs[0].observation_space.shape,
                    'num_actions': self.envs[0].action_space.n
                }, f'improved_sokoban_model_{global_step}.pt')
                print(f"💾 IMPROVED Model saved at step {global_step:,}")
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 IMPROVED TRAINING COMPLETE")
        if len(self.success_rates) > 0:
            final_success_rate = np.mean(list(self.success_rates)[-100:])
            final_avg_reward = np.mean(list(self.episode_rewards)[-100:])
            final_avg_boxes = np.mean(list(self.boxes_on_targets)[-100:])
            final_best_boxes = np.mean(list(self.best_boxes_achieved)[-100:])
            
            print(f"📊 Total Episodes: {len(self.success_rates):,}")
            print(f"🏆 Final Success Rate (last 100): {final_success_rate:.1%}")
            print(f"💰 Final Avg Reward (last 100): {final_avg_reward:.3f}")
            print(f"📦 Final Avg Boxes on Targets: {final_avg_boxes:.2f}")
            print(f"🎯 Final Avg Best Boxes: {final_best_boxes:.2f}")
            print(f"🎯 Final Curriculum Level: {self.curriculum.current_difficulty}")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='IMPROVED Efficient Sokoban PPO Training')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=256, help='Steps per rollout')
    parser.add_argument('--total-timesteps', type=int, default=2000000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate (IMPROVED: lower for stability)')
    
    args = parser.parse_args()
    
    print("🚀 IMPROVED EFFICIENT SOKOBAN PPO TRAINER")
    print("="*50)
    print("✅ BETTER REWARD SHAPING for faster learning")
    print("🧠 IMPROVED NETWORK ARCHITECTURE for spatial reasoning")
    print("📈 ENHANCED CURRICULUM LEARNING")
    print("🔍 BETTER PROGRESS TRACKING")
    print("="*50)
    
    trainer = ImprovedPPOTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()