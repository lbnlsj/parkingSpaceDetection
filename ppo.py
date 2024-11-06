import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import logging


class PPOMemory:
    """PPO的经验回放缓冲区"""

    def __init__(self, batch_size=32):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.vals = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, reward, probs, vals, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(probs)
        self.vals.append(vals)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.vals = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.dones),
                batches)


class ActorNetwork(nn.Module):
    """Actor网络：输出动作概率和位置调整"""

    def __init__(self, input_dim, n_actions, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 模型参数调整输出
        self.param_adjustment = nn.Linear(hidden_dim, n_actions)

        # 检测框调整输出
        self.box_adjustment = nn.Linear(hidden_dim, 4)  # x, y, width, height

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.param_adjustment, self.box_adjustment]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # 模型参数调整
        param_adj = torch.tanh(self.param_adjustment(x))  # 限制在[-1,1]范围

        # 检测框调整
        box_adj = torch.tanh(self.box_adjustment(x))  # 限制在[-1,1]范围

        return param_adj, box_adj


class CriticNetwork(nn.Module):
    """Critic网络：评估状态价值"""

    def __init__(self, input_dim, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value


class PPOAgent:
    """Modified PPO Agent for OpenCV's DNN model"""

    def __init__(self, net, input_dim, n_actions,
                 learning_rate=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 c1=1.0,
                 c2=0.01,
                 max_iterations=100,
                 iou_threshold=0.9
                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net  # OpenCV DNN model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.iou_threshold = iou_threshold

        # Actor-Critic Networks
        self.actor = ActorNetwork(input_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(input_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = PPOMemory()

        # 存储模型配置参数
        self.model_config = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'scale_factor': 0.007843,
            'mean': 127.5
        }

    def calculate_gae(self, rewards, values, dones):
        """计算广义优势估计(GAE)"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage

        returns = advantages + values
        return torch.tensor(advantages), torch.tensor(returns)

    def learn(self):
        """训练PPO算法"""
        states, actions, rewards, old_probs, values, dones, batches = self.memory.generate_batches()

        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)

        # 计算GAE
        advantages, returns = self.calculate_gae(rewards, values, dones)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(10):
            for batch in batches:
                # 获取新的动作分布和状态价值
                param_adj, box_adj = self.actor(states[batch])
                critic_value = self.critic(states[batch])

                # 将调整action合并
                full_adj = torch.cat([param_adj, box_adj], dim=1)

                # 计算新的动作概率
                dist = Normal(full_adj, torch.ones_like(full_adj))
                new_probs = dist.log_prob(actions[batch]).sum(dim=1)

                # 计算比率
                prob_ratio = (new_probs - old_probs[batch]).exp()

                # 计算PPO损失
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = advantages[batch] * torch.clamp(
                    prob_ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # 计算价值损失
                critic_loss = F.mse_loss(critic_value.squeeze(), returns[batch])

                # 计算熵损失
                entropy = dist.entropy().mean()

                # 总损失
                total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # 清空记忆
        self.memory.clear_memory()

    def get_model_parameter_statistics(self):
        """获取模型参数统计信息"""
        stats = [
            self.model_config['confidence_threshold'],
            self.model_config['nms_threshold'],
            self.model_config['scale_factor'],
            self.model_config['mean']
        ]
        return stats

    def get_model_weights(self):
        """获取模型状态"""
        return self.model_config.copy()

    def set_model_weights(self, weights):
        """设置模型状态"""
        self.model_config.update(weights)