# :rocket: Project Guide

The purpose of this project is to assess students' understanding of the fundamental theories and practical capabilities in reinforcement learning through reading, experiments, and programming tasks. Reinforcement Learning (RL) is a method that involves learning to maximize cumulative rewards through interaction with the environment and a process of trial and error. It generally consists of two key elements: the agent and the environment. At each time step, the agent receives the environmental state, takes an action, and obtains a corresponding reward. Reinforcement learning is a crucial technology in embodied intelligence and large language models. To enable beginners to gradually master the core concepts and common algorithms of reinforcement learning, this assessment project is divided into three stages with progressively increasing task difficulty. All programming tasks will be implemented using the Python language.

## :mag: 1.Basic Concept Learning

## :assistant: 2.Code Practice

| assignments | requirements |
| ----------- | ------------ |
|Task A: Implement a comparison of reinforcement learning algorithms in the OpenAI Gym FrozenLake-v1 environment| Manually implement the basic Q-learning algorithm (calling RL packages is prohibited); <br> Design a dynamically decaying ε value strategy (such as linear decay from 1.0 to 0.01); <br> Visualize the changes in Q-values during the training process; <br> Compare and analyze the impact of different learning rates (α) on the convergence speed |
|Task B: Read the paper "Human-level control through deep reinforcement learning" and reproduce DQN on the Atari Game according to the content of the paper | Use the PyTorch framework; You can choose to implement one of the many Atari games; <br> The experience pool (Simulation Dataset) should accumulate at least 100,000 pieces of data, and it is recommended to have more than 300,000 pieces; <br> The performance of the trained model should be significantly better than that of a random model |

### Task A: Q-Learning
