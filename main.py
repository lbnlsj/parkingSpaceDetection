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
    """Represents a single parking space"""

    def __init__(self, space_id: str, contour: np.ndarray):
        self.id = space_id
        self.contour = contour
        self.occupied = False
        self.last_update = datetime.datetime.now()
        self.confidence = 0.0

    def update_status(self, occupied: bool, confidence: float):
        """Update space status"""
        self.occupied = occupied
        self.confidence = confidence
        self.last_update = datetime.datetime.now()


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


class ParkingDetector:
    """Main detector class for parking space monitoring"""

    def __init__(
            self,
            prototxt_path: str,
            model_path: str,
            confidence_threshold: float = 0.5,
            use_cuda: bool = False
    ):
        if not Path(prototxt_path).exists():
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Try CUDA if requested
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

    def detect_vehicles(self, frame: np.ndarray) -> List[dict]:
        """Detect vehicles in the frame using MobileNet-SSD"""
        if frame is None or frame.size == 0:
            return []

        (h, w) = frame.shape[:2]
        # Adjust blob parameters for better detection
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
                # Focus on vehicle classes
                if self.CLASSES[idx] in ["car", "bus", "motorbike", "truck"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Validate detection coordinates
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # Calculate detection area
                    area = (endX - startX) * (endY - startY)
                    if area > 100:  # Minimum area threshold
                        vehicles.append({
                            'bbox': (startX, startY, endX, endY),
                            'confidence': float(confidence),
                            'area': area
                        })

        return vehicles

    def _process_frame(self, frame: np.ndarray, spaces: Dict[str, ParkingSpace]) -> dict:
        """Process a single frame"""
        vehicles = self.detect_vehicles(frame)
        results = {}

        for space_id, space in spaces.items():
            is_occupied, confidence = self.check_space_occupancy(
                frame,
                space.contour,
                vehicles
            )
            results[space_id] = (is_occupied, confidence)

        return results

    def check_space_occupancy(
            self,
            frame: np.ndarray,
            space_contour: np.ndarray,
            detected_vehicles: List[dict]
    ) -> Tuple[bool, float]:
        """Check if a parking space is occupied using multiple methods"""
        try:
            # 1. Create mask for parking space
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [space_contour], 255)
            space_area = cv2.countNonZero(mask)

            # 2. Check vehicle detection overlaps
            max_overlap_ratio = 0
            max_confidence = 0

            for vehicle in detected_vehicles:
                bbox = vehicle['bbox']
                bbox_mask = np.zeros_like(mask)
                cv2.rectangle(bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)

                overlap = cv2.bitwise_and(mask, bbox_mask)
                overlap_ratio = cv2.countNonZero(overlap) / space_area

                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    max_confidence = vehicle['confidence']

            # 3. Traditional CV-based detection
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert to grayscale
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )

            # Calculate the percentage of occupied pixels
            occupied_pixels = cv2.countNonZero(thresh)
            occupation_ratio = occupied_pixels / space_area

            # 4. Edge detection based check
            edges = cv2.Canny(blur, 50, 150)
            edge_pixels = cv2.countNonZero(edges)
            edge_ratio = edge_pixels / space_area

            # 5. Color-based detection (for vehicle shadows)
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            shadow_threshold = np.mean(v_channel[mask > 0]) < 100

            # Combined decision making
            is_occupied = False
            confidence = 0.0

            # Vehicle detection based decision
            if max_overlap_ratio > 0.3:
                is_occupied = True
                confidence = max(confidence, max_confidence)

            # Traditional CV based decision
            if occupation_ratio > 0.5:
                is_occupied = True
                confidence = max(confidence, occupation_ratio)

            # Edge detection based decision
            if edge_ratio > 0.1:
                is_occupied = True
                confidence = max(confidence, edge_ratio)

            # Shadow detection based decision
            if shadow_threshold:
                is_occupied = True
                confidence = max(confidence, 0.6)

            return is_occupied, min(confidence, 1.0)

        except Exception as e:
            logging.error(f"Error in space occupancy check: {e}")
            return False, 0.0

    def visualize_empty_spaces(
            self,
            frame: np.ndarray,
            spaces: Dict[str, ParkingSpace]
    ) -> np.ndarray:
        """Visualize empty parking spaces"""
        display_frame = frame.copy()

        # Count and sort empty spaces
        empty_spaces = [space for space in spaces.values() if not space.occupied]
        empty_count = len(empty_spaces)

        # Draw count
        cv2.putText(
            display_frame,
            f"Empty Parking Spaces: {empty_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Draw empty spaces
        for space in empty_spaces:
            # Draw green rectangle for empty space
            cv2.polylines(
                display_frame,
                [space.contour],
                True,
                (0, 255, 0),
                2
            )

            # Calculate centroid for text placement
            M = cv2.moments(space.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = tuple(space.contour[0])

            # Draw space ID
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
        # Initialize detector
        detector = ParkingDetector(
            prototxt_path="MobileNetSSD_deploy.prototxt",
            model_path="MobileNetSSD_deploy.caffemodel",
            confidence_threshold=0.4,  # Lowered threshold for better detection
            use_cuda=True
        )

        detector.start_processing()

        # Process PKLot dataset
        base_dir = Path("PKLot")
        if not base_dir.exists():
            logging.error(f"Dataset directory not found: {base_dir}")
            return

        # Process all images in the dataset
        for img_path in sorted(base_dir.glob("**/*.jpg")):
            # Get corresponding XML file
            xml_path = img_path.with_suffix('.xml')
            if not xml_path.exists():
                continue

            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            try:
                # Parse parking space data
                parking_data = ParkingSpaceData(str(xml_path))

                # Process frame
                detector.processing_queue.put((frame, parking_data.spaces))

                # Get results with timeout
                try:
                    results = detector.results_queue.get(timeout=5.0)
                    # Update space status
                    for space_id, (occupied, confidence) in results.items():
                        if space_id in parking_data.spaces:
                            parking_data.spaces[space_id].update_status(
                                occupied,
                                confidence
                            )

                    # Visualize results
                    display_frame = detector.visualize_empty_spaces(
                        frame,
                        parking_data.spaces
                    )

                    # Show frame
                    cv2.imshow('Empty Parking Spaces', display_frame)

                except queue.Empty:
                    logging.warning(f"Processing timeout for {img_path}")
                    continue

                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                time.sleep(0.1)  # Control processing speed

            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                continue

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.error(f"Processing error: {e}")
    finally:
        detector.stop_processing()
        cv2.destroyAllWindows()
        logging.info("Processing completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Application error: {e}")