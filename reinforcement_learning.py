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


class ParkingSpace:
    """Represents a single parking space with RL state"""

    def __init__(self, space_id: str, contour: np.ndarray):
        self.id = space_id
        self.contour = contour
        self.occupied = False
        self.last_update = datetime.datetime.now()
        self.confidence = 0.0
        # RL state features
        self.state_history = []
        self.q_values = {'occupied': 0.0, 'empty': 0.0}

    def update_status(self, occupied: bool, confidence: float):
        """Update space status and RL state"""
        self.occupied = occupied
        self.confidence = confidence
        self.last_update = datetime.datetime.now()
        self.state_history.append((occupied, confidence))
        if len(self.state_history) > 10:  # Keep last 10 states
            self.state_history.pop(0)


class RLAgent:
    """Reinforcement Learning Agent for parking detection"""

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: {'occupied': 0.0, 'empty': 0.0})

    def get_state_key(self, features: List[float]) -> str:
        """Convert continuous features to discrete state key"""
        # Discretize features into bins for Q-table
        discretized = []
        for f in features:
            if isinstance(f, float):
                # Discretize continuous values into 10 bins
                discretized.append(str(int(f * 10)))
            else:
                discretized.append(str(f))
        return '_'.join(discretized)

    def get_action(self, state_key: str) -> bool:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice([True, False])

        q_values = self.q_table[state_key]
        return q_values['occupied'] > q_values['empty']

    def update(self, state_key: str, action: bool, reward: float, next_state_key: str):
        """Update Q-values using Q-learning"""
        action_key = 'occupied' if action else 'empty'
        next_max_q = max(self.q_table[next_state_key].values())

        # Q-learning update formula
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state_key][action_key] = new_q


class ParkingDetector:
    """Main detector class with RL integration"""

    def __init__(
            self,
            prototxt_path: str,
            model_path: str,
            confidence_threshold: float = 0.5,
            use_cuda: bool = False
    ):
        # Original initialization
        if not Path(prototxt_path).exists():
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logging.info("Successfully initialized CUDA backend")
            except Exception as e:
                logging.warning(f"Failed to initialize CUDA backend: {e}. Using CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.confidence_threshold = confidence_threshold
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False

        # RL components
        self.rl_agent = RLAgent()
        self.previous_states = {}

    def extract_features(self, frame: np.ndarray, space: ParkingSpace, vehicles: List[dict]) -> List[float]:
        """Extract features for RL state"""
        # Create mask for parking space
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [space.contour], 255)
        space_area = cv2.countNonZero(mask)

        # 1. Vehicle detection overlap
        max_overlap_ratio = 0
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            bbox_mask = np.zeros_like(mask)
            cv2.rectangle(bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
            overlap = cv2.bitwise_and(mask, bbox_mask)
            overlap_ratio = cv2.countNonZero(overlap) / space_area
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        # 2. Edge density
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / space_area

        # 3. Color features
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        shadow_ratio = np.sum(v_channel[mask > 0] < 100) / space_area

        # 4. Historical information
        recent_occupancy = sum(1 for state in space.state_history[-5:] if state[0]) / max(1,
                                                                                          len(space.state_history[-5:]))

        return [max_overlap_ratio, edge_ratio, shadow_ratio, recent_occupancy]

    def calculate_reward(self, prediction: bool, confidence: float, previous_prediction: bool) -> float:
        """Calculate reward for RL agent"""
        # Base reward based on confidence
        reward = confidence - 0.5  # Reward scale: -0.5 to 0.5

        # Penalty for rapid switching
        if previous_prediction is not None and prediction != previous_prediction:
            reward -= 0.2  # Penalty for changing prediction

        return reward

    def check_space_occupancy(
            self,
            frame: np.ndarray,
            space: ParkingSpace,
            detected_vehicles: List[dict]
    ) -> Tuple[bool, float]:
        """Check if a parking space is occupied using RL"""
        try:
            # Extract features for RL state
            features = self.extract_features(frame, space, detected_vehicles)
            state_key = self.rl_agent.get_state_key(features)

            # Get previous prediction and state
            previous_prediction = space.occupied if space.id in self.previous_states else None
            previous_state = self.previous_states.get(space.id)

            # Get RL action (prediction)
            is_occupied = self.rl_agent.get_action(state_key)

            # Calculate confidence based on Q-values
            q_values = self.rl_agent.q_table[state_key]
            confidence = abs(q_values['occupied'] - q_values['empty']) / (
                    max(abs(q_values['occupied']), abs(q_values['empty'])) + 1e-6
            )

            # Calculate reward and update Q-values if we have previous state
            if previous_state is not None:
                reward = self.calculate_reward(is_occupied, confidence, previous_prediction)
                self.rl_agent.update(previous_state, previous_prediction, reward, state_key)

            # Store current state for next update
            self.previous_states[space.id] = state_key

            return is_occupied, confidence

        except Exception as e:
            logging.error(f"Error in space occupancy check: {e}")
            return False, 0.0

    # Rest of the class methods remain the same
    def detect_vehicles(self, frame: np.ndarray) -> List[dict]:
        """Detect vehicles in the frame using MobileNet-SSD"""
        if frame is None or frame.size == 0:
            return []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5,
            swapRB=True
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        vehicles = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] in ["car", "bus", "motorbike", "truck"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    area = (endX - startX) * (endY - startY)
                    if area > 100:
                        vehicles.append({
                            'bbox': (startX, startY, endX, endY),
                            'confidence': float(confidence),
                            'area': area
                        })

        return vehicles

    def start_processing(self):
        """Start the processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()

    def stop_processing(self):
        """Stop the processing thread"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                frame, spaces = self.processing_queue.get(timeout=1.0)
                results = self._process_frame(frame, spaces)
                self.results_queue.put(results)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {e}")

    def _process_frame(self, frame: np.ndarray, spaces: Dict[str, ParkingSpace]) -> dict:
        """Process a single frame"""
        vehicles = self.detect_vehicles(frame)
        results = {}

        for space_id, space in spaces.items():
            is_occupied, confidence = self.check_space_occupancy(
                frame,
                space,
                vehicles
            )
            results[space_id] = (is_occupied, confidence)

        return results

    def visualize_empty_spaces(
            self,
            frame: np.ndarray,
            spaces: Dict[str, ParkingSpace]
    ) -> np.ndarray:
        """Visualize empty parking spaces"""
        display_frame = frame.copy()

        empty_spaces = [space for space in spaces.values() if not space.occupied]
        empty_count = len(empty_spaces)

        cv2.putText(
            display_frame,
            f"Empty Parking Spaces: {empty_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        for space in empty_spaces:
            cv2.polylines(
                display_frame,
                [space.contour],
                True,
                (0, 255, 0),
                2
            )

            M = cv2.moments(space.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = tuple(space.contour[0])

            cv2.putText(
                display_frame,
                str(space.id),
                (cx - 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return display_frame


def main():
    try:
        detector = ParkingDetector(
            prototxt_path="MobileNetSSD_deploy.prototxt",
            model_path="MobileNetSSD_deploy.caffemodel",
            confidence_threshold=0.4,
            use_cuda=True
        )

        detector.start_processing()

        base_dir = Path("PKLot")
        if not base_dir.exists():
            logging.error(f"Dataset directory not found: {base_dir}")
            return

        for img_path in sorted(base_dir.glob("**/*.jpg")):
            xml_path = img_path.with_suffix('.xml')
            if not xml_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            try:
                parking_data = ParkingSpaceData(str(xml_path))
                detector.processing_queue.put((frame, parking_data.spaces))

                try:
                    results = detector.results_queue.get(timeout=5.0)
                    for space_id, (occupied, confidence) in results.items():
                        if space_id in parking_data.spaces:
                            parking_data.spaces[space_id].update_status(
                                occupied,
                                confidence
                            )

                    display_frame = detector.visualize_empty_spaces(
                        frame,
                        parking_data.spaces
                    )

                    cv2.imshow('Empty Parking Spaces', display_frame)

                except queue.Empty:
                    logging.warning(f"Processing timeout for {img_path}")
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # Add delay to control processing speed
                time.sleep(0.1)

                # Log RL metrics periodically
                if detector.rl_agent.q_table:
                    avg_q_value = np.mean([max(q_values.values())
                                           for q_values in detector.rl_agent.q_table.values()])
                    logging.info(f"Average Q-value: {avg_q_value:.4f}")

            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                continue

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.error(f"Processing error: {e}")
    finally:
        # Save Q-table before exit
        try:
            q_table_file = "q_table.pkl"
            import pickle
            with open(q_table_file, 'wb') as f:
                pickle.dump(dict(detector.rl_agent.q_table), f)
            logging.info(f"Q-table saved to {q_table_file}")
        except Exception as e:
            logging.error(f"Error saving Q-table: {e}")

        detector.stop_processing()
        cv2.destroyAllWindows()
        logging.info("Processing completed")


class ParkingSpaceData:
    """Handles parking space data from XML files"""

    def __init__(self, xml_path: str):
        if not Path(xml_path).exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        try:
            self.tree = ET.parse(xml_path)
            self.root = self.tree.getroot()
            self.spaces: Dict[str, ParkingSpace] = self.parse_spaces()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format in {xml_path}: {e}")

    def parse_spaces(self) -> Dict[str, ParkingSpace]:
        """Parse parking spaces from XML data"""
        spaces = {}
        for space in self.root.findall('.//space'):
            try:
                space_id = space.get('id')
                if space_id is None:
                    continue

                contour = []
                for point in space.findall('.//point'):
                    try:
                        x = int(point.get('x', 0))
                        y = int(point.get('y', 0))
                        contour.append([x, y])
                    except (ValueError, TypeError):
                        continue

                if len(contour) >= 3:  # Need at least 3 points for a valid contour
                    spaces[space_id] = ParkingSpace(
                        space_id=space_id,
                        contour=np.array(contour)
                    )
            except Exception as e:
                logging.warning(f"Error parsing space {space_id}: {e}")
                continue

        return spaces


if __name__ == "__main__":
    try:
        # Load saved Q-table if exists
        q_table_file = "q_table.pkl"
        if os.path.exists(q_table_file):
            try:
                import pickle

                with open(q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                logging.info(f"Loaded Q-table from {q_table_file}")
            except Exception as e:
                logging.error(f"Error loading Q-table: {e}")
                loaded_q_table = None
        else:
            loaded_q_table = None

        # Initialize and run main program
        main()

    except Exception as e:
        logging.critical(f"Application error: {e}")
