"""
Author: Redal
Date: 2025/07/09
TODO: Implement a comparison of reinforcement learning 
      algorithms in the OpenAI Gym FrozenLake-v1 environment
HomePage: https://github.com/Rtwotwo/Robot-Simulation.git
"""
import os
import random
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym.wrappers import FrameStack, RecordVideo
from gym.envs.atari import AtariEnv
import cv2

# 设置随机种子以保证结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 自定义Atari预处理包装器
class AtariPreprocessing:
    def __init__(self, env, frame_skip=4, screen_size=84, grayscale=True):
        self.env = env
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale = grayscale
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(screen_size, screen_size), dtype=np.uint8
        )
    
    def reset(self):
        obs = self.env.reset()
        return self._process_frame(obs)
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return self._process_frame(obs), total_reward, done, info
    
    def _process_frame(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return frame
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()

# 创建环境（带预处理）
def create_env(render_mode=None, record_video=False):
    # 使用基础Atari环境创建Pong
    env = AtariEnv(
        game='pong',
        obs_type='image',  # 使用图像观测
        frameskip=1,       # 每帧都处理
        repeat_action_probability=0.0,  # 无随机重复动作
        full_action_space=False
    )
    
    # 自定义预处理
    env = AtariPreprocessing(
        env, 
        frame_skip=4,         # 每4帧执行一次动作
        screen_size=84,       # 调整到84x84
        grayscale=True        # 转换为灰度图
    )
    
    # 堆叠4帧作为状态
    env = FrameStack(env, num_stack=4)
    
    # 如果需要录制视频
    if record_video:
        if not os.path.exists("videos"):
            os.makedirs("videos")
        env = RecordVideo(env, "videos", episode_trigger=lambda x: True)
    
    return env

# Q网络模型
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 全连接层
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 通过卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 通过全连接层
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机采样一批经验
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)

# DQN智能体
class DQNAgent:
    def __init__(self, action_dim, device, gamma=0.99, lr=1e-4):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        
        # 创建策略网络和目标网络
        self.policy_net = DQN(action_dim).to(device)
        self.target_net = DQN(action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不需要梯度
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 训练步数计数器
        self.steps_done = 0

    def select_action(self, state, epsilon):
        # ε-贪婪策略选择动作
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                # 添加批次维度并转换为张量
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update_model(self, replay_buffer, batch_size):
        # 如果缓冲区中样本不足，不更新
        if len(replay_buffer) < batch_size:
            return 0.0  # 返回0损失
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        # 更新目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        # 保存模型
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        # 加载模型
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        self.update_target_net()

# 训练函数
def train(args):
    # 创建环境
    env = create_env()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化智能体
    agent = DQNAgent(
        action_dim=env.action_space.n,
        device=device,
        gamma=args.gamma,
        lr=args.lr
    )
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=args.replay_capacity)
    
    # 训练参数
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = args.epsilon_decay
    batch_size = args.batch_size
    target_update = args.target_update
    
    # 初始化状态
    state = env.reset()
    episode_reward = 0
    episode_rewards = []
    episode_losses = []
    best_reward = -np.inf
    
    # 训练循环
    for step in range(1, args.total_steps + 1):
        # 线性衰减ε
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
        
        # 选择动作
        action = agent.select_action(state, epsilon)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        replay_buffer.push(state, action, reward, next_state, done)
        
        # 更新状态
        state = next_state
        episode_reward += reward
        
        # 更新模型
        loss = agent.update_model(replay_buffer, batch_size)
        if loss > 0:
            episode_losses.append(loss)
        
        # 定期更新目标网络
        if step % target_update == 0:
            agent.update_target_net()
        
        # 重置环境（如果回合结束）
        if done:
            # 记录回合奖励
            episode_rewards.append(episode_reward)
            
            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_model("best_model.pth")
            
            # 打印进度
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else episode_reward
            avg_loss = np.mean(episode_losses[-100:]) if len(episode_losses) > 0 else 0
            
            print(f"Step: {step}/{args.total_steps} | "
                  f"Episode: {len(episode_rewards)} | "
                  f"Reward: {episode_reward} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Epsilon: {epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(replay_buffer)}/{args.replay_capacity}")
            
            # 重置回合奖励
            state = env.reset()
            episode_reward = 0
            
        # 定期保存进度
        if step % 10000 == 0:
            agent.save_model(f"dqn_pong_step_{step}.pth")
    
    # 保存最终模型
    agent.save_model("final_model.pth")
    env.close()
    
    # 返回训练结果
    return {
        "episode_rewards": episode_rewards,
        "best_reward": best_reward
    }

# 测试函数
def test(args):
    # 创建环境（带视频录制）
    env = create_env(record_video=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化智能体
    agent = DQNAgent(
        action_dim=env.action_space.n,
        device=device
    )
    
    # 加载训练好的模型
    agent.load_model(args.model_path)
    
    # 测试多个回合
    all_rewards = []
    for episode in range(args.test_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            # 选择动作（使用贪婪策略）
            action = agent.select_action(state, epsilon=0.01)
            
            # 执行动作
            state, reward, done, _ = env.step(action)
            
            total_reward += reward
            step += 1
            
            # 结束条件
            if done or step >= args.max_steps:
                break
        
        all_rewards.append(total_reward)
        print(f"Episode: {episode+1}/{args.test_episodes} | Reward: {total_reward}")
    
    # 打印平均奖励
    avg_reward = np.mean(all_rewards)
    print(f"\nAverage Reward over {args.test_episodes} episodes: {avg_reward:.2f}")
    
    env.close()
    
    return all_rewards

# 随机策略测试（用于基准比较）
def test_random_policy(args):
    env = create_env()
    
    all_rewards = []
    for episode in range(args.test_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            state, reward, done, _ = env.step(action)
            
            total_reward += reward
            step += 1
            
            # 结束条件
            if done or step >= args.max_steps:
                break
        
        all_rewards.append(total_reward)
        print(f"Episode: {episode+1}/{args.test_episodes} | Reward: {total_reward}")
    
    # 打印平均奖励
    avg_reward = np.mean(all_rewards)
    print(f"\nRandom Policy Average Reward: {avg_reward:.2f}")
    
    env.close()
    
    return all_rewards

# 主函数
def main():
    parser = argparse.ArgumentParser(description="DQN for Atari Pong")
    
    # 训练参数
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--test_random", action="store_true", help="Test random policy")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to model file")
    
    # 环境参数
    parser.add_argument("--test_episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    
    # 训练超参数
    parser.add_argument("--total_steps", type=int, default=1000000, help="Total training steps")
    parser.add_argument("--replay_capacity", type=int, default=300000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon_decay", type=float, default=100000, help="Epsilon decay rate")
    parser.add_argument("--target_update", type=int, default=10000, help="Target network update frequency")
    
    args = parser.parse_args()
    
    # 执行训练或测试
    if args.train:
        print("Starting training...")
        train_results = train(args)
        print(f"Training completed. Best reward: {train_results['best_reward']}")
    
    if args.test:
        print("Testing trained model...")
        test_results = test(args)
        print(f"Test completed. Average reward: {np.mean(test_results):.2f}")
    
    if args.test_random:
        print("Testing random policy...")
        random_results = test_random_policy(args)
        print(f"Random policy average reward: {np.mean(random_results):.2f}")

if __name__ == "__main__":
    main()