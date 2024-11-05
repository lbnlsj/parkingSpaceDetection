import cv2
import numpy as np
import logging
import time
from pathlib import Path
import queue
import threading
from typing import Dict, List, Tuple
from collections import defaultdict
import datetime
import xml.etree.ElementTree as ET
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parking_detection.log'),
        logging.StreamHandler()
    ]
)


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

    def __init__(self, input_dim, n_actions, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # 模型参数调整输出
        self.param_adjustment = nn.Linear(hidden_dim, n_actions)

        # 检测框调整输出
        self.box_adjustment = nn.Linear(hidden_dim, 4)  # x, y, width, height

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.param_adjustment, self.box_adjustment]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 模型参数调整
        param_adj = torch.tanh(self.param_adjustment(x))  # 限制在[-1,1]范围

        # 检测框调整
        box_adj = torch.tanh(self.box_adjustment(x))  # 限制在[-1,1]范围

        return param_adj, box_adj


class CriticNetwork(nn.Module):
    """Critic网络：评估状态价值"""

    def __init__(self, input_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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

        # 移除net_optimizer
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

    def apply_model_adjustments(self, param_adjustments):
        """修改后的模型参数调整方法"""
        # 由于无法直接修改OpenCV模型参数，我们改为调整模型的配置参数
        config_adjustments = param_adjustments[:4]  # 假设前4个调整值用于配置

        # 调整置信度阈值
        self.model_config['confidence_threshold'] = np.clip(
            self.model_config['confidence_threshold'] + config_adjustments[0] * 0.1,
            0.1, 0.9
        )

        # 调整NMS阈值
        self.model_config['nms_threshold'] = np.clip(
            self.model_config['nms_threshold'] + config_adjustments[1] * 0.1,
            0.1, 0.9
        )

        # 调整缩放因子
        self.model_config['scale_factor'] = np.clip(
            self.model_config['scale_factor'] * (1 + config_adjustments[2] * 0.1),
            0.001, 0.01
        )

        # 调整均值
        self.model_config['mean'] = np.clip(
            self.model_config['mean'] * (1 + config_adjustments[3] * 0.1),
            100, 150
        )

    def get_current_detections(self, frame):
        """使用当前配置获取检测结果"""
        try:
            # 使用当前配置进行预处理
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                self.model_config['scale_factor'],
                (300, 300),
                self.model_config['mean'],
                swapRB=True
            )

            self.net.setInput(blob)
            detections = self.net.forward()

            # 使用当前置信度阈值过滤检测结果
            valid_detections = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.model_config['confidence_threshold']:
                    box = detections[0, 0, i, 3:7] * np.array([
                        frame.shape[1], frame.shape[0],
                        frame.shape[1], frame.shape[0]
                    ])
                    valid_detections.append(box.astype("int"))

            # 应用NMS
            if len(valid_detections) > 0:
                valid_detections = np.array(valid_detections)
                scores = detections[0, 0, :len(valid_detections), 2]
                indices = cv2.dnn.NMSBoxes(
                    valid_detections.tolist(),
                    scores.tolist(),
                    self.model_config['confidence_threshold'],
                    self.model_config['nms_threshold']
                )
                valid_detections = valid_detections[indices.flatten()]

            return np.array(valid_detections)

        except Exception as e:
            logging.error(f"Detection error: {e}")
            return np.array([])

    def get_model_parameter_statistics(self):
        """修改后的模型统计信息方法"""
        # 由于无法访问模型参数，我们返回配置参数的统计信息
        stats = [
            self.model_config['confidence_threshold'],
            self.model_config['nms_threshold'],
            self.model_config['scale_factor'],
            self.model_config['mean']
        ]
        return stats

    def get_model_weights(self):
        """修改后的获取模型状态方法"""
        return self.model_config.copy()

    def set_model_weights(self, weights):
        """修改后的设置模型状态方法"""
        self.model_config.update(weights)

    def calculate_gae(self, rewards, values, dones):
        """计算广义优势估计(GAE)"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage

        returns = advantages + values
        return torch.tensor(advantages), torch.tensor(returns)

    def optimize_detection(self, frame, ground_truth_boxes):
        """优化单帧图像的检测结果"""
        iteration = 0
        best_iou = 0
        best_weights = None
        episode_done = False

        while not episode_done and iteration < self.max_iterations:
            # 1. 获取当前检测结果
            detections = self.get_current_detections(frame)

            # 2. 计算当前状态
            state = self.extract_state_features(frame, detections, ground_truth_boxes)
            state_tensor = torch.FloatTensor(state).to(self.device)

            # 3. 获取动作
            with torch.no_grad():
                param_adj, box_adj = self.actor(state_tensor)
                value = self.critic(state_tensor)

            # 4. 应用动作
            self.apply_model_adjustments(param_adj.cpu().numpy())
            self.apply_box_adjustments(box_adj.cpu().numpy(), detections)

            # 5. 获取新的检测结果
            new_detections = self.get_current_detections(frame)

            # 6. 计算奖励
            current_iou = self.calculate_mean_iou(new_detections, ground_truth_boxes)
            reward = self.calculate_reward(current_iou, best_iou)

            # 7. 更新最佳结果
            if current_iou > best_iou:
                best_iou = current_iou
                best_weights = self.get_model_weights()

            # 8. 检查是否完成
            episode_done = current_iou >= self.iou_threshold

            # 9. 存储经验
            self.memory.store_memory(
                state,
                np.concatenate([param_adj.cpu().numpy(), box_adj.cpu().numpy()]),
                reward,
                0.0,  # 动作概率会在训练时计算
                value.item(),
                episode_done
            )

            # 10. 定期训练
            if len(self.memory.states) >= self.memory.batch_size:
                self.learn()

            iteration += 1

        # 恢复最佳权重
        if best_weights is not None:
            self.set_model_weights(best_weights)

        return best_iou

    def calculate_reward(self, current_iou, best_iou):
        """计算奖励"""
        reward = 0

        # IOU改善奖励
        iou_improvement = current_iou - best_iou
        if iou_improvement > 0:
            reward += 10 * iou_improvement  # 显著的改善给予更高奖励

        # IOU阈值达成奖励
        if current_iou >= self.iou_threshold:
            reward += 50  # 达到目标给予大奖励

        # 探索奖励（即使IOU没有提升，也给予小的正奖励以鼓励探索）
        if iou_improvement >= -0.01:  # 允许小幅度下降
            reward += 1

        return reward

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

    def extract_state_features(self, frame, detections, ground_truth_boxes):
        """提取状态特征"""
        features = []

        # 1. 检测框数量特征
        det_count = len(detections)
        gt_count = len(ground_truth_boxes)
        features.append(det_count / max(gt_count, 1))  # 归一化的数量比

        # 2. IOU特征
        max_iou = 0
        mean_iou = 0
        if det_count > 0 and gt_count > 0:
            ious = []
            for det in detections:
                best_iou = max(self.calculate_iou(det, gt) for gt in ground_truth_boxes)
                ious.append(best_iou)
            max_iou = max(ious)
            mean_iou = sum(ious) / len(ious)
        features.extend([max_iou, mean_iou])

        if det_count > 0:
            det_centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                                    for box in detections])
            det_sizes = np.array([(box[2] - box[0], box[3] - box[1])
                                  for box in detections])
            features.extend([
                np.mean(det_centers[:, 0]) / frame.shape[1],  # 平均x位置
                np.mean(det_centers[:, 1]) / frame.shape[0],  # 平均y位置
                np.mean(det_sizes[:, 0]) / frame.shape[1],  # 平均宽度
                np.mean(det_sizes[:, 1]) / frame.shape[0]  # 平均高度
            ])
        else:
            features.extend([0, 0, 0, 0])
        #
        # 4. 模型参数统计特征
        param_stats = self.get_model_parameter_statistics()
        features.extend(param_stats)

        return np.array(features, dtype=np.float32)

    def calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def calculate_mean_iou(self, detections, ground_truth_boxes):
        """计算平均IOU"""
        if len(detections) == 0 or len(ground_truth_boxes) == 0:
            return 0

        ious = []
        for det in detections:
            best_iou = max(self.calculate_iou(det, gt) for gt in ground_truth_boxes)
            ious.append(best_iou)

        return sum(ious) / len(ious)

    def apply_box_adjustments(self, box_adjustments, detections):
        """应用边界框调整"""
        if len(detections) == 0:
            return

        # 将调整应用到所有检测框
        adjustments = np.array([
            box_adjustments[0] * 10,  # x调整
            box_adjustments[1] * 10,  # y调整
            box_adjustments[2] * 5,  # 宽度调整
            box_adjustments[3] * 5  # 高度调整
        ])

        detections[:, :4] += adjustments


class ParkingDetector:
    """停车位检测器主类"""

    def __init__(self, prototxt_path: str, model_path: str,
                 input_dim=20, n_actions=10, **kwargs):
        # 初始化MobileNet-SSD
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # 初始化PPO Agent
        self.agent = PPOAgent(
            net=self.net,
            input_dim=input_dim,
            n_actions=n_actions,
            **kwargs
        )

        # 设置日志记录
        self.logger = logging.getLogger(__name__)

    def process_dataset(self, dataset_path: str):
        """处理整个数据集"""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        total_images = 0
        successful_detections = 0
        total_iou = 0

        for img_path in sorted(dataset_path.glob("**/*.jpg")):
            try:
                # 读取图像
                frame = cv2.imread(str(img_path))
                if frame is None:
                    self.logger.warning(f"Could not read image: {img_path}")
                    continue

                # 读取真实标注
                ground_truth = self.read_ground_truth(img_path)
                if ground_truth is None:
                    self.logger.warning(f"No ground truth found for: {img_path}")
                    continue

                # 优化检测
                best_iou = self.agent.optimize_detection(frame, ground_truth)

                # 更新统计信息
                total_images += 1
                if best_iou > self.agent.iou_threshold:
                    successful_detections += 1
                total_iou += best_iou

                # 记录进度
                self.logger.info(
                    f"Processed {img_path.name}: IOU = {best_iou:.3f}, "
                    f"Success Rate = {successful_detections / total_images:.3f}"
                )

                # 可视化结果
                self.visualize_results(frame, ground_truth)

            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                continue

        # 输出最终统计信息
        self.logger.info(
            f"Dataset Processing Complete:\n"
            f"Total Images: {total_images}\n"
            f"Successful Detections: {successful_detections}\n"
            f"Average IOU: {total_iou / total_images:.3f}\n"
            f"Success Rate: {successful_detections / total_images:.3f}"
        )

    def read_ground_truth(self, img_path: Path):
        """读取真实标注数据"""
        xml_path = img_path.with_suffix('.xml')
        if not xml_path.exists():
            return None

        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            ground_truth = []

            for space in root.findall('.//space'):
                points = []
                for point in space.findall('.//point'):
                    x = int(point.get('x', 0))
                    y = int(point.get('y', 0))
                    points.append([x, y])

                if len(points) >= 3:
                    # 转换多边形为边界框格式
                    points = np.array(points)
                    x, y, w, h = cv2.boundingRect(points)
                    ground_truth.append([x, y, x + w, y + h])

            return np.array(ground_truth)

        except Exception as e:
            self.logger.error(f"Error reading ground truth: {e}")
            return None

    def visualize_results(self, frame, ground_truth):
        """可视化检测结果"""
        display_frame = frame.copy()

        # 绘制地面真实框
        for box in ground_truth:
            cv2.rectangle(
                display_frame,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),  # 绿色表示真实框
                2
            )

        # 获取当前检测结果
        detections = self.agent.get_current_detections(frame)

        # 绘制检测框
        for box in detections:
            cv2.rectangle(
                display_frame,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 0, 255),  # 红色表示检测框
                2
            )

        # 显示图像
        cv2.imshow('Detection Results', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True


def main():
    try:
        # 初始化检测器
        detector = ParkingDetector(
            prototxt_path="MobileNetSSD_deploy.prototxt",
            model_path="MobileNetSSD_deploy.caffemodel",
            input_dim=20,
            n_actions=10,
            learning_rate=0.0003,
            max_iterations=100,
            iou_threshold=0.8
        )

        # 处理数据集
        detector.process_dataset("PKLot")

    except Exception as e:
        logging.critical(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
