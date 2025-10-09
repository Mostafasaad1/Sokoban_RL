# demo.py
import gymnasium as gym
from sokoban_env import SokobanEnv
import time

env = SokobanEnv(render_mode="human")
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.2)
    if done:
        obs, _ = env.reset()
env.close()