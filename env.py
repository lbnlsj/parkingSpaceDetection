import cv2
import numpy as np
import logging
from pathlib import Path
import xml.etree.ElementTree as ET


class ParkingDetectorEnv:
    def __init__(self, prototxt_path: str, model_path: str, input_dim=20, n_actions=10):
        if not Path(prototxt_path).exists():
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.input_dim = input_dim
        self.n_actions = n_actions

        
        self.model_config = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'scale_factor': 0.007843,
            'mean': (127.5, 127.5, 127.5)
        }

        
        self.size_limits = {
            'min_width_ratio': 0.03,  
            'max_width_ratio': 0.15,  
            'min_height_ratio': 0.03,  
            'max_height_ratio': 0.15,  
            'max_area_ratio': 0.02,  
        }

    def adjust_detection_size(self, box: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """
        调整检测框大小到合理范围，更严格的大小控制
        """
        frame_height, frame_width = frame_shape[:2]
        adjusted_box = box.copy()

        
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        
        width = box[2] - box[0]
        height = box[3] - box[1]

        
        width_ratio = width / frame_width
        height_ratio = height / frame_height
        area_ratio = (width * height) / (frame_width * frame_height)

        
        if width_ratio > self.size_limits['max_width_ratio'] or area_ratio > self.size_limits['max_area_ratio']:
            new_width = min(
                frame_width * self.size_limits['max_width_ratio'],
                np.sqrt(self.size_limits['max_area_ratio'] * frame_width * frame_height * (width / height))
            )
            adjusted_box[0] = max(0, int(center_x - new_width / 2))
            adjusted_box[2] = min(frame_width, int(center_x + new_width / 2))

        if height_ratio > self.size_limits['max_height_ratio'] or area_ratio > self.size_limits['max_area_ratio']:
            new_height = min(
                frame_height * self.size_limits['max_height_ratio'],
                np.sqrt(self.size_limits['max_area_ratio'] * frame_width * frame_height * (height / width))
            )
            adjusted_box[1] = max(0, int(center_y - new_height / 2))
            adjusted_box[3] = min(frame_height, int(center_y + new_height / 2))

        
        if width_ratio < self.size_limits['min_width_ratio']:
            new_width = frame_width * self.size_limits['min_width_ratio']
            adjusted_box[0] = max(0, int(center_x - new_width / 2))
            adjusted_box[2] = min(frame_width, int(center_x + new_width / 2))

        if height_ratio < self.size_limits['min_height_ratio']:
            new_height = frame_height * self.size_limits['min_height_ratio']
            adjusted_box[1] = max(0, int(center_y - new_height / 2))
            adjusted_box[3] = min(frame_height, int(center_y + new_height / 2))

        return adjusted_box

    def is_reasonable_detection(self, box: np.ndarray, frame_shape: tuple) -> bool:
        """
        检查检测框是否合理
        """
        frame_height, frame_width = frame_shape[:2]

        
        width = box[2] - box[0]
        height = box[3] - box[1]
        width_ratio = width / frame_width
        height_ratio = height / frame_height
        area_ratio = (width * height) / (frame_width * frame_height)

        
        return (width_ratio <= self.size_limits['max_width_ratio'] and
                height_ratio <= self.size_limits['max_height_ratio'] and
                area_ratio <= self.size_limits['max_area_ratio'] and
                width_ratio >= self.size_limits['min_width_ratio'] and
                height_ratio >= self.size_limits['min_height_ratio'])

    def get_current_detections(self, frame):
        """获取当前检测结果"""
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                scalefactor=self.model_config['scale_factor'],
                size=(300, 300),
                mean=self.model_config['mean'],
                swapRB=True,
                crop=False
            )

            self.net.setInput(blob)
            detections = self.net.forward()

            valid_detections = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.model_config['confidence_threshold']:
                    box = detections[0, 0, i, 3:7] * np.array([
                        frame.shape[1], frame.shape[0],
                        frame.shape[1], frame.shape[0]
                    ])
                    box = box.astype("int")

                    
                    adjusted_box = self.adjust_detection_size(box, frame.shape)
                    if self.is_reasonable_detection(adjusted_box, frame.shape):
                        valid_detections.append(adjusted_box)

            
            if len(valid_detections) > 0:
                valid_detections = np.array(valid_detections)
                scores = detections[0, 0, :len(valid_detections), 2]
                indices = cv2.dnn.NMSBoxes(
                    valid_detections.tolist(),
                    scores.tolist(),
                    self.model_config['confidence_threshold'],
                    self.model_config['nms_threshold']
                )
                if len(indices) > 0:
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

        
        adjustments = np.array([
            box_adjustments[0] * 10,  
            box_adjustments[1] * 10,  
            box_adjustments[2] * 5,  
            box_adjustments[3] * 5  
        ], dtype=np.float32)

        adjusted_detections = detections.astype(np.float32)

        
        for i in range(len(adjusted_detections)):
            
            adjusted_detections[i, :4] += adjustments

            
            adjusted_detections[i] = self.adjust_detection_size(
                adjusted_detections[i],
                frame_shape=(adjusted_detections.shape[0], adjusted_detections.shape[1])
            )

        
        detections[:] = adjusted_detections.round().astype(np.int64)


    def extract_state_features(self, frame, detections, ground_truth_boxes):
        """提取状态特征，确保输出11维特征向量"""
        features = []

        
        det_count = len(detections)
        gt_count = len(ground_truth_boxes)
        features.append(det_count / max(gt_count, 1))  

        
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
                np.mean(det_centers[:, 0]) / frame.shape[1],  
                np.mean(det_centers[:, 1]) / frame.shape[0],  
                np.mean(det_sizes[:, 0]) / frame.shape[1],  
                np.mean(det_sizes[:, 1]) / frame.shape[0]  
            ])
        else:
            features.extend([0, 0, 0, 0])

        
        features.extend([
            self.model_config['confidence_threshold'],
            self.model_config['nms_threshold'],
            self.model_config['scale_factor'],
            self.model_config['mean'][0] / 255.0  
        ])

        return np.array(features, dtype=np.float32)

    def apply_model_adjustments(self, param_adjustments):
        """应用模型参数调整"""
        config_adjustments = param_adjustments[:4]  

        
        self.model_config['confidence_threshold'] = np.clip(
            self.model_config['confidence_threshold'] + config_adjustments[0] * 0.1,
            0.1, 0.9
        )

        
        self.model_config['nms_threshold'] = np.clip(
            self.model_config['nms_threshold'] + config_adjustments[1] * 0.1,
            0.1, 0.9
        )

        
        self.model_config['scale_factor'] = np.clip(
            self.model_config['scale_factor'] * (1 + config_adjustments[2] * 0.1),
            0.001, 0.01
        )

        
        mean_adjustment = 1 + config_adjustments[3] * 0.1
        new_mean = np.clip(np.array(self.model_config['mean']) * mean_adjustment, 100, 150)
        self.model_config['mean'] = tuple(new_mean)  

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

        
        iou_improvement = current_iou - best_iou
        if iou_improvement > 0:
            reward += 20 * iou_improvement  

        
        if current_iou >= 0.5:  
            reward += current_iou * 100  

        
        if current_iou > 0:  
            reward += current_iou * 10
        else:
            reward -= 1  

        
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
                    
                    points = np.array(points)
                    x, y, w, h = cv2.boundingRect(points)
                    ground_truth.append([x, y, x + w, y + h])

            return np.array(ground_truth)

        except Exception as e:
            logging.error(f"Error reading ground truth: {e}")
            return None
