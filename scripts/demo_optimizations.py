#!/usr/bin/env python3
"""
Demo script to showcase the improvements in the optimized Sokoban environment.
Compares behavior between original and optimized reward systems.
"""

import numpy as np
from optimized_sokoban_env import OptimizedSokobanEnv

def test_reward_improvements():
    """
    Demonstrate the improved reward system with detailed analysis.
    """
    print("🧪 Testing Optimized Sokoban Environment")
    print("=" * 60)
    
    # Test different difficulty levels
    for difficulty in [1, 2, 3]:
        print(f"\n🎯 Testing Difficulty Level {difficulty}")
        print("-" * 40)
        
        env = OptimizedSokobanEnv(
            max_episode_steps=50,  # Short for demo
            curriculum_mode=True,
            difficulty_level=difficulty,
            anti_hacking_strength=2.0
        )
        
        obs, info = env.reset()
        print(f"🎲 Initial state: {info['boxes_on_targets']}/{info['total_targets']} boxes on targets")
        
        total_reward = 0
        step_count = 0
        
        # Simulate some moves
        for i in range(10):
            action = np.random.randint(0, 4)  # Random action for demo
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if reward != -0.002:  # Only show non-step rewards
                print(f"  Step {step_count}: Action {action} → Reward {reward:.3f}")
                if 'reward_components' in info:
                    components = info['reward_components']
                    print(f"    Components: {components}")
            
            if done:
                if info['is_solved']:
                    print(f"🎉 SOLVED in {step_count} steps!")
                break
                
            if truncated:
                print(f"⏰ Episode truncated at {step_count} steps")
                break
        
        print(f"💰 Total reward: {total_reward:.3f}")
        print(f"📦 Final boxes on targets: {info['boxes_on_targets']}/{info['total_targets']}")
        print(f"📉 Useless push ratio: {info['useless_push_ratio']:.2%}")

def test_anti_hacking_features():
    """
    Demonstrate anti-hacking mechanisms.
    """
    print("\n\n🛑 Testing Anti-Hacking Features")
    print("=" * 60)
    
    env = OptimizedSokobanEnv(
        max_episode_steps=100,
        curriculum_mode=True,
        difficulty_level=1,
        anti_hacking_strength=3.0  # High strength for demo
    )
    
    obs, info = env.reset()
    
    print("📦 Simulating box oscillation (moving back and forth)...")
    
    # Simulate some back-and-forth movement
    actions = [2, 3, 2, 3, 2, 3]  # LEFT, RIGHT, LEFT, RIGHT...
    
    for i, action in enumerate(actions):
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: Action {['UP','DOWN','LEFT','RIGHT'][action]} → Reward {reward:.3f}")
        
        if 'reward_components' in info and info['reward_components']:
            components = info['reward_components']
            if components.get('anti_hack', 0) < 0:
                print(f"    ⚠️  Anti-hack penalty detected: {components['anti_hack']:.3f}")
    
    print(f"\n📉 Final useless push ratio: {info['useless_push_ratio']:.2%}")
    print(f"📦 Push count: {info['push_count']}")

def test_curriculum_system():
    """
    Demonstrate curriculum learning system.
    """
    print("\n\n🎓 Testing Curriculum Learning System")
    print("=" * 60)
    
    from optimized_train_sokoban_ppo import CurriculumManager
    
    curriculum = CurriculumManager(
        success_threshold=0.2,  # 20% for demo
        evaluation_window=10    # Small window for demo
    )
    
    print(f"Starting at difficulty level: {curriculum.current_difficulty}")
    
    # Simulate training episodes with varying success
    print("\n🎮 Simulating training episodes...")
    
    # First batch: mostly failures (should stay at level 1)
    results = [False] * 8 + [True] * 2  # 20% success
    for i, success in enumerate(results):
        changed = curriculum.update(success)
        if changed:
            print(f"  Episode {i+1}: Difficulty changed to {curriculum.current_difficulty}!")
    
    print(f"After {len(results)} episodes: Level {curriculum.current_difficulty}")
    
    # Second batch: high success (should upgrade)
    results = [True] * 8 + [False] * 2  # 80% success
    for i, success in enumerate(results):
        changed = curriculum.update(success)
        if changed:
            print(f"  Episode {len(results) + i + 1}: 🚀 UPGRADE to level {curriculum.current_difficulty}!")
    
    print(f"After high success: Level {curriculum.current_difficulty}")

def main():
    """
    Run all demonstration tests.
    """
    print("🎆 OPTIMIZED SOKOBAN ENVIRONMENT DEMO")
    print("This demo showcases the advanced features and improvements.")
    print()
    
    try:
        test_reward_improvements()
        test_anti_hacking_features()
        test_curriculum_system()
        
        print("\n\n✨ Demo Complete!")
        print("🚀 Ready to train with: python optimized_train_sokoban_ppo.py")
        
    except Exception as e:
        print(f"\n⚠️  Demo error: {e}")
        print("Make sure sokoban_cpp is available and properly compiled.")

if __name__ == '__main__':
    main()
