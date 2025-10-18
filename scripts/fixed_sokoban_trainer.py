"""
FIXED SOKOBAN PPO TRAINER - CORRECT LEARNING RATE
=================================================

Critical fixes:
- Learning rate: 0.0003 (was 0.003 - 10x too high!)
- Stable network architecture
- Value loss guards
- Conservative hyperparameters
- Enhanced monitoring
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import environments
try:
    from efficient_sokoban_env import EfficientSokobanEnv
    import sokoban_engine as soko
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Environment import failed: {e}")
    ENV_AVAILABLE = False

# =============================================================================
# STABLE NETWORK ARCHITECTURE
# =============================================================================

class StableSokobanNetwork(nn.Module):
    """
    Stable network with bounded value predictions and conservative architecture.
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 512):
        super().__init__()
        
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        print(f"🧠 STABLE NETWORK:")
        print(f"   Input: {input_size}, Hidden: {hidden_size}, Actions: {num_actions}")
        
        # Enhanced input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Conservative shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_actions)
        )
        
        # Conservative value head with bounded output
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # Bound between -1 and 1
        )
        
        # Conservative weight initialization
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Parameters: {total_params:,}")
        print(f"   Value range: [-10, 10]")
        
    def _initialize_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)  # Reduced gain
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Input processing
        x = self.input_norm(x.float())
        features = self.shared_layers(x)
        
        # Policy and value
        logits = self.actor_head(features)
        value = self.critic_head(features) * 10.0  # Scale to [-10, 10]
        
        return logits, value.squeeze(-1)

# =============================================================================
# ENHANCED EXPLORATION MANAGER
# =============================================================================

class SmartExplorationManager:
    """
    Smart exploration with smooth decay and diversity tracking.
    """
    
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.05, decay_steps=100000):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.step_count = 0
        self.action_history = deque(maxlen=100)
        
    def get_epsilon(self):
        """Linear epsilon decay"""
        if self.step_count >= self.decay_steps:
            return self.min_epsilon
        
        fraction = self.step_count / self.decay_steps
        epsilon = self.initial_epsilon - fraction * (self.initial_epsilon - self.min_epsilon)
        return max(epsilon, self.min_epsilon)
    
    def record_action(self, action):
        """Record action for diversity tracking"""
        self.action_history.append(action)
    
    def get_action_diversity(self):
        """Calculate action diversity score"""
        if len(self.action_history) == 0:
            return 0.0
        unique_actions = len(set(self.action_history))
        return unique_actions / len(self.action_history)
    
    def step(self):
        """Increment step counter"""
        self.step_count += 1

# =============================================================================
# ENHANCED SOKOBAN ENVIRONMENT
# =============================================================================

class EnhancedSokobanEnv(EfficientSokobanEnv):
    """
    Enhanced Sokoban environment with better reward shaping and exploration incentives.
    """
    
    def __init__(self, **kwargs):
        if not ENV_AVAILABLE:
            raise ImportError("Sokoban environment not available")
            
        super().__init__(**kwargs)
        
        # Enhanced tracking
        self.best_boxes_on_targets = 0
        self.exploration_bonus_used = set()
        self.last_actions = deque(maxlen=5)
        self.consecutive_same_actions = 0
        self.last_action = None
        
        print("🎮 ENHANCED SOKOBAN ENVIRONMENT")
        print("   ✓ Better reward shaping")
        print("   ✓ Action diversity tracking")
    
    def _get_player_position(self):
        """Get current player position"""
        grid = self.game.get_grid()
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == self.PLAYER or cell == self.PLAYER_ON_TARGET:
                    return (x, y)
        return (0, 0)
    
    def _calculate_box_proximity_bonus(self, player_pos):
        """Calculate bonus for being close to boxes"""
        boxes = self.get_box_positions()
        if not boxes:
            return 0.0
        
        min_distance = float('inf')
        for box_x, box_y in boxes:
            distance = abs(player_pos[0] - box_x) + abs(player_pos[1] - box_y)
            min_distance = min(min_distance, distance)
        
        return max(0, (6 - min_distance) * 0.1)
    
    def _calculate_action_diversity_bonus(self, action):
        """Calculate bonus for using diverse actions"""
        if not self.last_actions:
            return 0.0
        
        # Bonus for changing actions
        if action != self.last_actions[-1]:
            return 0.1
        else:
            # Small penalty for repeating same action too much
            self.consecutive_same_actions += 1
            if self.consecutive_same_actions > 3:
                return -0.05 * (self.consecutive_same_actions - 3)
        
        return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Store pre-action state
        prev_boxes_on_targets = self.boxes_on_targets()
        prev_player_pos = self._get_player_position()
        
        # Execute action
        action_map = {0: soko.UP, 1: soko.DOWN, 2: soko.LEFT, 3: soko.RIGHT}
        obs_data, base_reward, done, pushed_box = self.game.step(action_map[action])
        
        # Update tracking
        self.step_count += 1
        current_boxes_on_targets = self.boxes_on_targets()
        current_player_pos = self._get_player_position()
        
        # ==================== ENHANCED REWARD CALCULATION ====================
        reward_components = {}
        reward = 0.0
        
        # 1. BASE STEP REWARD
        step_penalty = -0.01
        reward += step_penalty
        reward_components['step_penalty'] = step_penalty
        
        # 2. ACTION DIVERSITY BONUS
        diversity_bonus = self._calculate_action_diversity_bonus(action)
        if diversity_bonus != 0:
            reward += diversity_bonus
            reward_components['diversity'] = diversity_bonus
        
        # Update action tracking
        self.last_actions.append(action)
        if self.last_action == action:
            self.consecutive_same_actions += 1
        else:
            self.consecutive_same_actions = 1
        self.last_action = action
        
        # 3. TARGET PROGRESS (primary learning signal)
        target_progress = current_boxes_on_targets - prev_boxes_on_targets
        if target_progress > 0:
            progress_bonus = target_progress * 10.0  # Reasonable bonus
            reward += progress_bonus
            reward_components['target_progress'] = progress_bonus
            # print(f"🎯 BOX ON TARGET! +{progress_bonus:.1f}")
        elif target_progress < 0:
            regression_penalty = target_progress * 3.0
            reward += regression_penalty
            reward_components['target_regression'] = regression_penalty
        
        # 4. MILESTONE REWARDS
        if current_boxes_on_targets > self.best_boxes_on_targets:
            old_best = self.best_boxes_on_targets
            self.best_boxes_on_targets = current_boxes_on_targets
            milestone_bonus = 8.0 * (current_boxes_on_targets - old_best)
            reward += milestone_bonus
            reward_components['milestone'] = milestone_bonus
            # print(f"🏆 NEW BEST: {current_boxes_on_targets} boxes! +{milestone_bonus:.1f}")
        
        # 5. EXPLORATION BONUS
        pos_key = f"{current_player_pos[0]},{current_player_pos[1]}"
        if pos_key not in self.exploration_bonus_used:
            self.exploration_bonus_used.add(pos_key)
            if len(self.exploration_bonus_used) <= 20:
                exploration_bonus = 0.1
                reward += exploration_bonus
                reward_components['exploration'] = exploration_bonus
        
        # 6. BOX INTERACTION REWARD
        if pushed_box:
            interaction_bonus = 1.0
            reward += interaction_bonus
            reward_components['box_interaction'] = interaction_bonus
        
        # 7. BOX PROXIMITY BONUS
        proximity_bonus = self._calculate_box_proximity_bonus(current_player_pos)
        reward += proximity_bonus
        reward_components['proximity'] = proximity_bonus
        
        # 8. COMPLETION BONUS
        if done and self.game.is_solved():
            efficiency_bonus = max(0, 200 - self.step_count) * 0.5
            completion_bonus = 15000.0 + efficiency_bonus
            reward += completion_bonus
            reward_components['completion'] = completion_bonus
            self.solved_step = self.step_count
            print(f"🎉 LEVEL SOLVED! +{completion_bonus:.1f} (Steps: {self.step_count})")
        
        # 9. TIMEOUT MANAGEMENT
        if self.step_count >= 200:
            done = True
            if not self.game.is_solved():
                # Scale penalty based on progress
                if self.best_boxes_on_targets > 0:
                    timeout_penalty = -2.0 * (4 - self.best_boxes_on_targets)
                else:
                    timeout_penalty = -10.0
                reward += timeout_penalty
                reward_components['timeout'] = timeout_penalty
                print(f"👎 LEVEL NOT SOLVED! {timeout_penalty:.1f} (Steps: {self.step_count})")
                
        
        self.total_reward += reward
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info(reward_components)
        info['best_boxes_on_targets'] = self.best_boxes_on_targets
        info['action_diversity'] = len(set(self.last_actions)) / max(1, len(self.last_actions))
        
        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment with enhanced tracking"""
        result = super().reset(seed, options)
        self.best_boxes_on_targets = 0
        self.exploration_bonus_used = set()
        self.last_actions.clear()
        self.consecutive_same_actions = 0
        self.last_action = None
        return result

# =============================================================================
# SMART CURRICULUM MANAGER
# =============================================================================

class SmartCurriculumManager:
    """
    Smart curriculum learning with multiple progression criteria.
    """
    
    def __init__(self, success_threshold=0.15, evaluation_window=100):
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.current_difficulty = 1
        self.success_history = deque(maxlen=evaluation_window)
        self.reward_history = deque(maxlen=evaluation_window)
        self.boxes_history = deque(maxlen=evaluation_window)
        self.episodes_at_current_level = 0
        self.min_episodes_per_level = 400
        
        print("📈 SMART CURRICULUM MANAGER")
        print(f"   Starting at level {self.current_difficulty}")
        print(f"   Success threshold: {success_threshold:.1%}")
    
    def update(self, success: bool, episode_reward: float, boxes_on_targets: int) -> bool:
        """Update curriculum based on multiple performance metrics"""
        self.success_history.append(success)
        self.reward_history.append(episode_reward)
        self.boxes_history.append(boxes_on_targets)
        self.episodes_at_current_level += 1
        
        # Only evaluate after minimum episodes and full window
        if (len(self.success_history) >= self.evaluation_window and 
            self.episodes_at_current_level >= self.min_episodes_per_level):
            
            success_rate = np.mean(self.success_history)
            avg_reward = np.mean(self.reward_history)
            avg_boxes = np.mean(self.boxes_history)
            
            upgrade_ready = False
            
            # Multiple upgrade criteria
            if success_rate >= self.success_threshold and avg_reward > 0:
                upgrade_ready = True
                reason = f"success rate ({success_rate:.1%})"
            elif avg_boxes >= 1.5 and avg_reward > -2.0:
                upgrade_ready = True
                reason = f"box progress ({avg_boxes:.1f} boxes)"
            elif avg_reward > 5.0:
                upgrade_ready = True
                reason = f"high reward ({avg_reward:.1f})"
            
            # Upgrade if ready
            if upgrade_ready and self.current_difficulty < 3:
                old_level = self.current_difficulty
                self.current_difficulty += 1
                self._reset_tracking()
                print(f"\n🚀 CURRICULUM UPGRADE: Level {old_level} → {self.current_difficulty}")
                print(f"   Reason: {reason}")
                return True
            
            # Downgrade if struggling
            elif (success_rate < 0.02 and avg_reward < -15.0 and 
                  self.current_difficulty > 1 and self.episodes_at_current_level > 800):
                old_level = self.current_difficulty
                self.current_difficulty -= 1
                self._reset_tracking()
                print(f"\n📉 CURRICULUM DOWNGRADE: Level {old_level} → {self.current_difficulty}")
                print(f"   Reason: poor performance")
                return True
        
        return False
    
    def _reset_tracking(self):
        """Reset tracking variables"""
        self.episodes_at_current_level = 0
        self.success_history.clear()
        self.reward_history.clear()
        self.boxes_history.clear()

# =============================================================================
# FIXED PPO TRAINER
# =============================================================================

class FixedPPOTrainer:
    """
    Fixed PPO trainer with correct learning rate and stable training.
    """
    
    def __init__(self, args):
        if not ENV_AVAILABLE:
            raise ImportError("Cannot create environments - dependencies missing")
            
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Create enhanced environments
        self.envs = [EnhancedSokobanEnv() for _ in range(args.num_envs)]
        
        # Initialize components
        obs_shape = self.envs[0].observation_space.shape
        num_actions = self.envs[0].action_space.n
        
        # 🔥 CRITICAL FIX: Correct learning rate
        self.learning_rate = 0.0003  # Was 0.003 (10x too high!)
        
        self.network = StableSokobanNetwork(obs_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.96)
        
        # Enhanced components
        self.exploration_manager = SmartExplorationManager()
        self.curriculum = SmartCurriculumManager()
        
        # Conservative PPO parameters
        self.gamma = 0.98
        self.gae_lambda = 0.92
        self.clip_range = 0.15
        self.entropy_coef = 0.015
        self.value_coef = 0.3
        self.max_grad_norm = 0.3
        
        # Comprehensive tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=1000)
        self.boxes_on_targets = deque(maxlen=1000)
        self.best_boxes_achieved = deque(maxlen=1000)
        self.action_diversities = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        
        print(f"\n⚡ FIXED PPO TRAINER INITIALIZED!")
        print(f"🎯 Environments: {args.num_envs}")
        print(f"📚 Learning rate: {self.learning_rate} (FIXED - was 0.003)")
        print(f"🎲 Entropy coefficient: {self.entropy_coef}")
        print(f"📉 Value coefficient: {self.value_coef}")
        print(f"🔒 Max grad norm: {self.max_grad_norm}")
        print("=" * 60)
    
    def collect_rollouts(self):
        """Collect rollouts with smart exploration"""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        # Initialize environments
        obs_list = []
        for env in self.envs:
            if not hasattr(env, '_current_obs'):
                obs, _ = env.reset()
                env._current_obs = obs
            obs_list.append(env._current_obs)
        
        # Get current exploration rate
        epsilon = self.exploration_manager.get_epsilon()
        
        # Collect experience
        for step in range(self.args.num_steps):
            obs_tensor = torch.FloatTensor(obs_list).to(self.device)
            
            with torch.no_grad():
                logits, value = self.network(obs_tensor)
                
                # Smart exploration: epsilon-greedy + policy sampling
                if np.random.random() < epsilon:
                    # Random exploration
                    action = torch.randint(0, self.envs[0].action_space.n, (len(self.envs),))
                    log_prob = torch.zeros(len(self.envs))
                else:
                    # Policy-based action
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # Record actions for diversity tracking
                for a in action.cpu().numpy():
                    self.exploration_manager.record_action(a)
            
            # Store data
            observations.append(obs_tensor.cpu())
            actions.append(action.cpu())
            values.append(value.cpu())
            log_probs.append(log_prob.cpu())
            
            # Execute actions
            next_obs_list = []
            step_rewards = []
            step_dones = []
            
            for i, env in enumerate(self.envs):
                obs, reward, terminated, truncated, info = env.step(action[i].item())
                
                step_rewards.append(reward)
                step_dones.append(terminated or truncated)
                
                if terminated or truncated:
                    # Track comprehensive metrics
                    self.episode_rewards.append(info.get('total_reward', reward))
                    self.episode_lengths.append(info.get('step_count', 1))
                    self.success_rates.append(1.0 if info.get('is_solved', False) else 0.0)
                    self.boxes_on_targets.append(info.get('boxes_on_targets', 0))
                    self.best_boxes_achieved.append(info.get('best_boxes_on_targets', 0))
                    self.action_diversities.append(info.get('action_diversity', 0.0))
                    
                    # Update curriculum
                    episode_reward = info.get('total_reward', reward)
                    success = info.get('is_solved', False)
                    boxes = info.get('boxes_on_targets', 0)
                    self.curriculum.update(success, episode_reward, boxes)
                    
                    # Reset environment
                    obs, _ = env.reset()
                
                env._current_obs = obs
                next_obs_list.append(obs)
            
            rewards.append(step_rewards)
            dones.append(step_dones)
            obs_list = next_obs_list
        
        # Update exploration
        self.exploration_manager.step()
        
        # Final values for GAE
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
        """Compute Generalized Advantage Estimation"""
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
                next_nonterminal = 1.0 - dones[step].float()
                next_values = final_values
            else:
                next_nonterminal = 1.0 - dones[step].float()
                next_values = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_values * next_nonterminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[step] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, rollouts, advantages, returns):
        """Update policy with VALUE LOSS GUARD for stability"""
        observations = rollouts['observations'].view(-1, *rollouts['observations'].shape[2:])
        actions = rollouts['actions'].view(-1)
        old_log_probs = rollouts['log_probs'].view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple PPO epochs for stable learning
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(4):
            # Forward pass
            logits, values = self.network(observations.to(self.device))
            
            # Policy loss
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.to(self.device))
            ratio = torch.exp(new_log_probs - old_log_probs.to(self.device))
            
            surr1 = ratio * advantages.to(self.device)
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages.to(self.device)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 🔥 CRITICAL: Value loss with guard against explosion
            value_loss = nn.functional.mse_loss(values.squeeze(), returns.to(self.device))
            
            # Guard against value loss explosion
            if value_loss > 500:
                value_loss = torch.clamp(value_loss, max=100.0)
                print(f"🚨 VALUE GUARD: Clipped loss from {value_loss.item():.1f}")
            
            # Track value loss for monitoring
            self.value_losses.append(value_loss.item())
            
            # Entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss with stable coefficients
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimization step with conservative gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def train(self):
        """Main training loop"""
        total_timesteps = self.args.total_timesteps
        num_updates = total_timesteps // (self.args.num_envs * self.args.num_steps)
        global_step = 0
        
        print(f"\n🚀 STARTING FIXED PPO TRAINING")
        print(f"📊 Total timesteps: {total_timesteps:,}")
        print(f"🔄 Number of updates: {num_updates:,}")
        print(f"🌍 Environments: {self.args.num_envs}")
        print(f"👟 Steps per rollout: {self.args.num_steps}")
        print("=" * 80)
        
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # Set curriculum difficulty
            for env in self.envs:
                env.set_difficulty(self.curriculum.current_difficulty)
            
            # Collect experience
            rollouts = self.collect_rollouts()
            global_step += self.args.num_envs * self.args.num_steps
            
            # Compute advantages
            advantages, returns = self.compute_gae(rollouts)
            
            # Update policy
            loss_info = self.update_policy(rollouts, advantages, returns)
            
            # Enhanced logging
            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                
                # Calculate metrics
                metrics = {}
                if len(self.episode_rewards) > 0:
                    recent_idx = slice(-min(50, len(self.episode_rewards)), None)
                    metrics.update({
                        'avg_reward': np.mean(list(self.episode_rewards)[recent_idx]),
                        'avg_length': np.mean(list(self.episode_lengths)[recent_idx]),
                        'success_rate': np.mean(list(self.success_rates)[recent_idx]),
                        'avg_boxes': np.mean(list(self.boxes_on_targets)[recent_idx]),
                        'avg_best_boxes': np.mean(list(self.best_boxes_achieved)[recent_idx]),
                        'action_diversity': np.mean(list(self.action_diversities)[recent_idx]),
                        'exploration_epsilon': self.exploration_manager.get_epsilon(),
                        'avg_value_loss': np.mean(list(self.value_losses)[-50:]) if self.value_losses else 0.0
                    })
                
                # Print comprehensive update
                print(f"\n📈 UPDATE {update:,} | STEP {global_step:,}")
                print(f"⏱️  Time: {elapsed_time:.1f}s | Speed: {steps_per_sec:.0f} steps/s")
                print(f"🎯 Curriculum: Level {self.curriculum.current_difficulty}")
                print(f"📚 Learning Rate: {loss_info['learning_rate']:.6f}")
                print(f"📉 Policy Loss: {loss_info['policy_loss']:.4f}")
                print(f"📊 Value Loss: {loss_info['value_loss']:.4f}")
                print(f"🎲 Entropy: {loss_info['entropy_loss']:.4f}")
                
                if metrics:
                    print(f"💰 Avg Reward: {metrics['avg_reward']:.3f}")
                    print(f"📏 Avg Length: {metrics['avg_length']:.1f}")
                    print(f"🏆 Success Rate: {metrics['success_rate']:.1%}")
                    print(f"📦 Avg Boxes: {metrics['avg_boxes']:.2f}")
                    print(f"🎯 Avg Best Boxes: {metrics['avg_best_boxes']:.2f}")
                    print(f"🔄 Action Diversity: {metrics['action_diversity']:.1%}")
                    print(f"🔍 Exploration ε: {metrics['exploration_epsilon']:.3f}")
                    print(f"⚡ Avg Value Loss: {metrics['avg_value_loss']:.1f}")
                
                print("~" * 60)
            
            # Save model periodically
            if update % 100 == 0 and update > 0:
                checkpoint = {
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'global_step': global_step,
                    'curriculum_level': self.curriculum.current_difficulty,
                    'obs_shape': self.envs[0].observation_space.shape,
                    'num_actions': self.envs[0].action_space.n,
                    'training_args': vars(self.args),
                    'value_loss_history': list(self.value_losses)[-100:]
                }
                
                filename = f'fixed_sokoban_model_{global_step}.pt'
                torch.save(checkpoint, filename)
                print(f"💾 Model saved: {filename}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 FIXED TRAINING COMPLETE!")
        print("=" * 80)
        
        if len(self.success_rates) > 0:
            final_metrics = {
                'success_rate': np.mean(list(self.success_rates)[-100:]),
                'avg_reward': np.mean(list(self.episode_rewards)[-100:]),
                'avg_boxes': np.mean(list(self.boxes_on_targets)[-100:]),
                'final_level': self.curriculum.current_difficulty,
                'total_episodes': len(self.success_rates),
                'final_value_loss': np.mean(list(self.value_losses)[-100:]) if self.value_losses else 0.0
            }
            
            print(f"📊 Total Episodes: {final_metrics['total_episodes']:,}")
            print(f"🏆 Final Success Rate: {final_metrics['success_rate']:.1%}")
            print(f"💰 Final Avg Reward: {final_metrics['avg_reward']:.3f}")
            print(f"📦 Final Avg Boxes: {final_metrics['avg_boxes']:.2f}")
            print(f"🎯 Final Curriculum Level: {final_metrics['final_level']}")
            print(f"⚡ Final Value Loss: {final_metrics['final_value_loss']:.1f}")
        
        print("=" * 80)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fixed Sokoban PPO Training')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=256, help='Steps per rollout')
    parser.add_argument('--total-timesteps', type=int, default=2000000, help='Total training timesteps')
    
    args = parser.parse_args()
    
    print("🚀 FIXED SOKOBAN PPO TRAINING SYSTEM")
    print("=" * 50)
    print("✅ CORRECT LEARNING RATE: 0.0003 (was 0.003)")
    print("🛡️  VALUE LOSS GUARD against explosion")
    print("📉 CONSERVATIVE HYPERPARAMETERS")
    print("🎯 ENHANCED REWARD SHAPING")
    print("=" * 50)
    
    try:
        trainer = FixedPPOTrainer(args)
        trainer.train()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()