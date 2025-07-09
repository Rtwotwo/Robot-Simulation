"""
Author: Redal
Date: 2025/07/09
TODO: Implement a comparison of reinforcement learning 
      algorithms in the OpenAI Gym FrozenLake-v1 environment
HomePage: https://github.com/Rtwotwo/Robot-Simulation.git
"""
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
# 设置plt显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
plt.reParams['font.family'] = 'SimHei'
# 创建FrozenLake-v1环境
env = gym.make('FrozenLake-v1', desc=None, 
        map_name="4x4", is_slippery=True)


def Q_learning(env, learning_rate=0.1, 
               discount_factor=0.95, 
               epsilon_start=1.0, 
               epsilon_min=0.01,
               episodes=10000):
    """实现基本的Q-leanring算法
    env (gym.Env): OpenAI Gym FrozenLake-v1环境
    learning_rate (float, optional): 学习率. Defaults to 0.1.
    discount_factor (float, optional): 折扣因子. Defaults to 0.95.
    epsilon_start (float, optional): epsilon的初始值. Defaults to 1.0.
    epsilon_min (float, optional): epsilon的最小值. Defaults to 0.01.
    episodes (int, optional): 训练的轮数. Defaults to 10000"""
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    steps = []
    q_values = []
    epsilon_values = []
    
