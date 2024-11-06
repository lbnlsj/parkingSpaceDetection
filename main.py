import logging
import cv2
import torch
import numpy as np
from pathlib import Path
from env import ParkingDetectorEnv
from ppo import PPOAgent
from visualization import ParkingVisualizer


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('parking_detection.log'),
            logging.StreamHandler()
        ]
    )


def process_dataset(env, agent, visualizer, dataset_path):
    """处理数据集"""
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
                logging.warning(f"Could not read image: {img_path}")
                continue

            # 读取真实标注
            ground_truth = env.read_ground_truth(img_path)
            if ground_truth is None:
                logging.warning(f"No ground truth found for: {img_path}")
                continue

            # 优化检测过程
            best_iou = 0
            best_weights = None

            for iteration in range(agent.max_iterations):
                # 获取当前检测结果
                detections = env.get_current_detections(frame)

                # 提取状态特征
                state = env.extract_state_features(frame, detections, ground_truth)
                state_tensor = torch.FloatTensor(state).to(agent.device)

                # Agent决策并执行动作
                with torch.no_grad():  # 添加这行来禁用梯度计算
                    param_adj, box_adj = agent.actor(state_tensor)
                    value = agent.critic(state_tensor)

                # 转换tensor到numpy，使用detach()
                param_adj_np = param_adj.cpu().detach().numpy()
                box_adj_np = box_adj.cpu().detach().numpy()

                # 应用动作
                env.apply_model_adjustments(param_adj_np)
                env.apply_box_adjustments(box_adj_np, detections)

                # 获取新的检测结果
                new_detections = env.get_current_detections(frame)

                # 计算新的IOU和奖励
                current_iou = env.calculate_mean_iou(new_detections, ground_truth)
                reward = env.calculate_reward(current_iou, best_iou)

                # 更新最佳结果
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_weights = agent.get_model_weights()

                # 存储经验
                agent.memory.store_memory(
                    state,
                    np.concatenate([param_adj_np, box_adj_np]),
                    reward,
                    0.0,  # 动作概率会在训练时计算
                    value.cpu().detach().item(),  # 修改这里，使用detach()
                    current_iou >= agent.iou_threshold
                )

                # 可视化当前结果
                display_frame = visualizer.visualize_detections(
                    frame,
                    new_detections,
                    ground_truth,
                    {'IOU': current_iou, 'Reward': reward}
                )

                if not visualizer.show_frame(display_frame):
                    return

                # 定期训练
                if len(agent.memory.states) >= agent.memory.batch_size:
                    agent.learn()

                # 检查是否达到目标
                if current_iou >= agent.iou_threshold:
                    break

            # 更新统计信息
            total_images += 1
            if best_iou > agent.iou_threshold:
                successful_detections += 1
            total_iou += best_iou

            # 恢复最佳权重
            if best_weights is not None:
                agent.set_model_weights(best_weights)

            # 记录进度
            logging.info(
                f"Processed {img_path.name}: "
                f"IOU = {best_iou:.3f}, "
                f"Success Rate = {successful_detections / total_images:.3f}"
            )

        except Exception as e:
            continue

    # 输出最终统计信息
    logging.info(
        f"\nDataset Processing Complete:\n"
        f"Total Images: {total_images}\n"
        f"Successful Detections: {successful_detections}\n"
        f"Average IOU: {total_iou / total_images:.3f}\n"
        f"Success Rate: {successful_detections / total_images:.3f}"
    )


def main():
    """主程序入口"""
    try:
        # 设置日志
        setup_logging()

        # 初始化环境
        env = ParkingDetectorEnv(
            prototxt_path="MobileNetSSD_deploy.prototxt",
            model_path="MobileNetSSD_deploy.caffemodel",
            input_dim=11,
            n_actions=10
        )

        # 初始化Agent
        agent = PPOAgent(
            net=env.net,
            input_dim=11,
            n_actions=10,
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            max_iterations=100,
            iou_threshold=0.8
        )

        # 初始化可视化工具
        visualizer = ParkingVisualizer()

        # 处理数据集
        process_dataset(
            env=env,
            agent=agent,
            visualizer=visualizer,
            dataset_path="PKLot"
        )

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.critical(f"Application error: {e}")
    finally:
        cv2.destroyAllWindows()
        logging.info("Processing completed")


if __name__ == "__main__":
    main()