import cv2
import numpy as np
import logging


class ParkingVisualizer:
    """停车位检测可视化工具类"""

    def __init__(self):
        self.colors = {
            'ground_truth': (0, 255, 0),  # 绿色表示真实框
            'detection': (0, 0, 255),  # 红色表示检测框
            'text': (255, 255, 255),  # 白色文字
            'empty': (0, 255, 0),  # 绿色表示空车位
            'occupied': (0, 0, 255)  # 红色表示占用车位
        }

    def draw_boxes(self, frame, boxes, color, thickness=2):
        """绘制边界框"""
        for box in boxes:
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                thickness
            )
        return frame

    def draw_text(self, frame, text, position, scale=0.6):
        """绘制文字"""
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            self.colors['text'],
            2
        )
        return frame

    def visualize_detections(self, frame, detections, ground_truth, metrics=None):
        """可视化检测结果"""
        display_frame = frame.copy()

        # 绘制地面真实框
        display_frame = self.draw_boxes(
            display_frame,
            ground_truth,
            self.colors['ground_truth']
        )

        # 绘制检测框
        display_frame = self.draw_boxes(
            display_frame,
            detections,
            self.colors['detection']
        )

        # 添加指标信息
        if metrics is not None:
            y_offset = 30
            for key, value in metrics.items():
                text = f"{key}: {value:.3f}"
                display_frame = self.draw_text(
                    display_frame,
                    text,
                    (10, y_offset)
                )
                y_offset += 30

        return display_frame

    def visualize_parking_spaces(self, frame, spaces, occupancy_status):
        """可视化停车位状态"""
        display_frame = frame.copy()

        # 绘制空车位计数
        empty_count = sum(1 for status in occupancy_status.values() if not status)
        total_count = len(occupancy_status)

        self.draw_text(
            display_frame,
            f"Empty: {empty_count}/{total_count}",
            (10, 30)
        )

        # 绘制每个车位
        for space_id, status in occupancy_status.items():
            if space_id in spaces:
                color = self.colors['occupied'] if status else self.colors['empty']
                cv2.polylines(
                    display_frame,
                    [spaces[space_id]],
                    True,
                    color,
                    2
                )

                # 计算中心点并添加ID标签
                M = cv2.moments(spaces[space_id])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = spaces[space_id][0]

                self.draw_text(
                    display_frame,
                    str(space_id),
                    (cx - 10, cy + 10),
                    0.5
                )

        return display_frame

    def show_frame(self, frame, window_name='Parking Detection', wait_time=1):
        """显示帧"""
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait_time) & 0xFF
        return key != ord('q')

    def save_frame(self, frame, save_path):
        """保存帧"""
        try:
            cv2.imwrite(save_path, frame)
            logging.info(f"Frame saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving frame to {save_path}: {e}")