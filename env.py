import cv2
import numpy as np
import logging
from pathlib import Path
import xml.etree.ElementTree as ET


class ParkingDetectorEnv:
    """停车位检测环境类"""

    def __init__(self, prototxt_path: str, model_path: str, input_dim=20, n_actions=10):
        if not Path(prototxt_path).exists():
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 初始化网络
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.input_dim = input_dim
        self.n_actions = n_actions

        # 配置参数
        self.model_config = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'scale_factor': 0.007843,
            'mean': (127.5, 127.5, 127.5)  # 使用元组存储RGB三通道的mean值
        }

    def extract_state_features(self, frame, detections, ground_truth_boxes):
        """提取状态特征，确保输出11维特征向量"""
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

        # 3. 位置和大小特征
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

        # 4. 模型参数特征 - 修改这里，使用mean的第一个通道值
        features.extend([
            self.model_config['confidence_threshold'],
            self.model_config['nms_threshold'],
            self.model_config['scale_factor'],
            self.model_config['mean'][0] / 255.0  # 只使用第一个通道的值进行归一化
        ])

        return np.array(features, dtype=np.float32)

    def apply_model_adjustments(self, param_adjustments):
        """应用模型参数调整"""
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

        # 调整均值 - 修改为调整所有通道
        mean_adjustment = 1 + config_adjustments[3] * 0.1
        new_mean = np.clip(np.array(self.model_config['mean']) * mean_adjustment, 100, 150)
        self.model_config['mean'] = tuple(new_mean)  # 转换回元组格式

    """停车位检测环境类"""

    def get_current_detections(self, frame):
        """获取当前检测结果"""
        try:
            # 预处理 - 确保mean参数正确传递
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                scalefactor=self.model_config['scale_factor'],
                size=(300, 300),
                mean=self.model_config['mean'],  # 使用元组格式的mean值
                swapRB=True,
                crop=False
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
                if len(indices) > 0:  # 确保indices不为空
                    indices = indices.flatten()
                    valid_detections = valid_detections[indices]

            return np.array(valid_detections)

        except Exception as e:
            logging.error(f"Detection error: {str(e)}")
            return np.array([])

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
        ], dtype=np.float32)  # 确保使用float32类型

        # 先进行调整计算，然后再转换为整数
        adjusted_detections = detections.astype(np.float32)
        adjusted_detections[:, :4] += adjustments

        # 最后转换回整数类型
        detections[:] = adjusted_detections.round().astype(np.int64)

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

    def calculate_reward(self, current_iou, best_iou):
        """计算奖励"""
        reward = 0

        # IOU改善奖励
        iou_improvement = current_iou - best_iou
        if iou_improvement > 0:
            reward += 20 * iou_improvement  # 增加奖励系数

        # IOU阈值奖励
        if current_iou >= 0.5:  # 降低阈值
            reward += current_iou * 100  # 根据IOU值给予更大奖励

        # 基础探索奖励
        if current_iou > 0:  # 只要有重叠就给予基础奖励
            reward += current_iou * 10
        else:
            reward -= 1  # 没有重叠给予小惩罚

        # 添加惩罚以避免停留在局部最优
        if current_iou == best_iou:
            reward -= 0.1

        return reward

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
            logging.error(f"Error reading ground truth: {e}")
            return None
