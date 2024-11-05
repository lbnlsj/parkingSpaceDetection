import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import json
import datetime
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import threading
import queue
import time

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
    """Represents a single parking space with its properties and history"""

    def __init__(self, space_id: str, contour: np.ndarray, occupied: bool = False):
        self.id = space_id
        self.contour = contour
        self.occupied = occupied
        self.occupation_history = []
        self.last_update = datetime.datetime.now()
        self.confidence = 0.0

    def update_status(self, occupied: bool, confidence: float):
        """Update the occupancy status and confidence of the parking space"""
        self.occupied = occupied
        self.confidence = confidence
        self.last_update = datetime.datetime.now()
        self.occupation_history.append((self.last_update, occupied, confidence))

    def get_occupation_rate(self, time_window: datetime.timedelta) -> float:
        """Calculate occupation rate over a specific time window"""
        current_time = datetime.datetime.now()
        relevant_history = [x for x in self.occupation_history
                            if current_time - x[0] <= time_window]
        if not relevant_history:
            return 0.0
        return sum(1 for x in relevant_history if x[1]) / len(relevant_history)


class ParkingSpaceData:
    """Handles parsing and management of parking space data from XML files"""

    def __init__(self, xml_path: str):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        try:
            self.tree = ET.parse(xml_path)
            self.root = self.tree.getroot()
            self.spaces: Dict[str, ParkingSpace] = self.parse_spaces()
            self.metadata = self._extract_metadata()
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

                # Handle missing or invalid occupied attribute
                occupied_str = space.get('occupied', '0')
                try:
                    occupied = bool(int(occupied_str))
                except (ValueError, TypeError):
                    occupied = False

                contour = []
                for point in space.findall('.//contour/point'):
                    try:
                        x = int(point.get('x', 0))
                        y = int(point.get('y', 0))
                        contour.append([x, y])
                    except (ValueError, TypeError):
                        continue

                if len(contour) >= 3:  # Need at least 3 points for a valid contour
                    spaces[space_id] = ParkingSpace(
                        space_id=space_id,
                        contour=np.array(contour),
                        occupied=occupied
                    )
            except Exception as e:
                logging.warning(f"Error parsing space {space_id}: {e}")
                continue

        return spaces

    def _extract_metadata(self) -> dict:
        """Extract metadata from XML file"""
        metadata = {
            'parking_id': self.root.get('id', 'unknown'),
            'date': None,
            'weather': None,
            'total_spaces': len(self.spaces)
        }

        # Try to extract optional metadata
        date_elem = self.root.find('.//date')
        if date_elem is not None:
            metadata['date'] = date_elem.text

        weather_elem = self.root.find('.//weather')
        if weather_elem is not None:
            metadata['weather'] = weather_elem.text

        return metadata


class ParkingDetector:
    """Handles vehicle detection and parking space occupancy analysis"""

    def __init__(self,
                 prototxt_path: str,
                 model_path: str,
                 confidence_threshold: float = 0.5,
                 use_cuda: bool = False):
        if not os.path.exists(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Try CUDA if requested
        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logging.info("Successfully initialized CUDA backend")
            except Exception as e:
                logging.warning(f"Failed to initialize CUDA backend: {e}. Falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.confidence_threshold = confidence_threshold
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        self.stats = defaultdict(int)
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False

    def start_processing_thread(self):
        """Start background processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()

    def stop_processing_thread(self):
        """Stop background processing thread"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

    def _processing_loop(self):
        """Background processing loop"""
        while self.is_running:
            try:
                frame, spaces = self.processing_queue.get(timeout=1.0)
                results = self._process_frame(frame, spaces)
                self.results_queue.put(results)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in processing loop: {str(e)}")

    def detect_vehicles(self, frame: np.ndarray) -> List[dict]:
        """Detect vehicles in frame using MobileNet-SSD"""
        if frame is None or frame.size == 0:
            logging.error("Invalid frame provided")
            return []

        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)

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
                        vehicles.append({
                            'bbox': (startX, startY, endX, endY),
                            'confidence': float(confidence),
                            'class': self.CLASSES[idx]
                        })
            return vehicles

        except Exception as e:
            logging.error(f"Error in vehicle detection: {e}")
            return []

    def _process_frame(self, frame: np.ndarray, spaces: Dict[str, ParkingSpace]) -> dict:
        """Process a single frame"""
        vehicles = self.detect_vehicles(frame)
        results = {}

        for space_id, space in spaces.items():
            is_occupied = self.check_space_occupancy(frame, space.contour, vehicles)
            confidence = self._calculate_confidence(frame, space.contour, vehicles)
            results[space_id] = (is_occupied, confidence)

        return results

    def check_space_occupancy(self,
                              frame: np.ndarray,
                              space_contour: np.ndarray,
                              detected_vehicles: List[dict]) -> bool:
        """Determine if a parking space is occupied"""
        try:
            # Create mask for the parking space
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [space_contour], 255)

            # Check for vehicle detections overlapping with space
            for vehicle in detected_vehicles:
                bbox = vehicle['bbox']
                bbox_mask = np.zeros_like(mask)
                cv2.rectangle(bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)

                overlap = cv2.bitwise_and(mask, bbox_mask)
                if cv2.countNonZero(overlap) > 0.3 * cv2.countNonZero(mask):
                    return True

            # Additional analysis using traditional CV
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)[1]

            total_pixels = cv2.countNonZero(mask)
            occupied_pixels = cv2.countNonZero(thresh)
            occupation_ratio = occupied_pixels / total_pixels if total_pixels > 0 else 0

            return occupation_ratio > 0.3

        except Exception as e:
            logging.error(f"Error checking space occupancy: {e}")
            return False

    def _calculate_confidence(self,
                              frame: np.ndarray,
                              space_contour: np.ndarray,
                              detected_vehicles: List[dict]) -> float:
        """Calculate confidence score for occupancy detection"""
        try:
            confidence_scores = []

            # Vehicle detection confidence
            space_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(space_mask, [space_contour], 255)

            for vehicle in detected_vehicles:
                bbox = vehicle['bbox']
                bbox_mask = np.zeros_like(space_mask)
                cv2.rectangle(bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)

                overlap = cv2.bitwise_and(space_mask, bbox_mask)
                overlap_ratio = cv2.countNonZero(overlap) / cv2.countNonZero(space_mask)

                if overlap_ratio > 0:
                    confidence_scores.append(vehicle['confidence'] * overlap_ratio)

            # Traditional CV confidence
            masked_frame = cv2.bitwise_and(frame, frame, mask=space_mask)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            edge_confidence = min(1.0, cv2.countNonZero(edges) / 1000)
            confidence_scores.append(edge_confidence)

            return max(confidence_scores) if confidence_scores else 0.0

        except Exception as e:
            logging.error(f"Error calculating confidence: {e}")
            return 0.0


class ParkingAnalytics:
    """Handles analytics and reporting for parking space usage"""

    def __init__(self, save_dir: str = 'parking_analytics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.data = defaultdict(list)

    def update(self, parking_id: str, spaces: Dict[str, ParkingSpace]):
        """Update analytics with current parking status"""
        timestamp = datetime.datetime.now()
        occupied_count = sum(1 for space in spaces.values() if space.occupied)
        total_spaces = len(spaces)

        if total_spaces > 0:  # Avoid division by zero
            self.data['timestamp'].append(timestamp)
            self.data['parking_id'].append(parking_id)
            self.data['occupied_spaces'].append(occupied_count)
            self.data['total_spaces'].append(total_spaces)
            self.data['occupancy_rate'].append(occupied_count / total_spaces)

    def generate_report(self) -> dict:
        """Generate analytics report"""
        if not self.data['timestamp']:  # Check if we have any data
            return {
                'total_observations': 0,
                'average_occupancy_rate': 0,
                'peak_occupancy_rate': 0,
                'peak_hours': [],
                'peak_rates': []
            }

        df = pd.DataFrame(self.data)

        try:
            hourly_stats = df.set_index('timestamp').resample('1H').mean()
            peak_hours = hourly_stats['occupancy_rate'].nlargest(3)

            report = {
                'total_observations': len(df),
                'average_occupancy_rate': float(df['occupancy_rate'].mean()),
                'peak_occupancy_rate': float(df['occupancy_rate'].max()),
                'peak_hours': peak_hours.index.strftime('%H:00').tolist(),
                'peak_rates': peak_hours.tolist()
            }
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            report = {
                'error': str(e),
                'total_observations': len(df)
            }

        # Save report
        try:
            with open(self.save_dir / 'report.json', 'w') as f:
                json.dump(report, f, indent=4, default=str)
        except Exception as e:
            logging.error(f"Error saving report: {e}")

        return report

    def plot_occupancy_trends(self, save_plot: bool = True):
        """Plot occupancy trends over time"""
        try:
            df = pd.DataFrame(self.data)
            if df.empty:
                logging.warning("No data available for plotting")
                return

            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['occupancy_rate'])
            plt.title('Parking Lot Occupancy Over Time')
            plt.xlabel('Time')
            plt.ylabel('Occupancy Rate')
            plt.grid(True)

            if save_plot:
                plt.savefig(self.save_dir / 'occupancy_trends.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting occupancy trends: {e}")


def main():
    """Main function to run the parking detection system"""
    try:
        # Initialize detector with error handling for model files
        detector = None
        try:
            detector = ParkingDetector(
                prototxt_path="MobileNetSSD_deploy.prototxt",
                model_path="MobileNetSSD_deploy.caffemodel",
                confidence_threshold=0.5,
                use_cuda=True
            )
        except FileNotFoundError as e:
            logging.error(f"Model file error: {e}")
            return
        except Exception as e:
            logging.error(f"Detector initialization error: {e}")
            return

        analytics = ParkingAnalytics()
        detector.start_processing_thread()

        # Process images
        try:
            base_dir = Path("PKLot")
            if not base_dir.exists():
                logging.error(f"Dataset directory not found: {base_dir}")
                return

            for img_file in base_dir.glob("**/*.jpg"):
                xml_file = img_file.with_suffix('.xml')
                if not xml_file.exists():
                    logging.warning(f"No XML file found for {img_file}")
                    continue

                # Read and validate image
                frame = cv2.imread(str(img_file))
                if frame is None:
                    logging.warning(f"Failed to read image: {img_file}")
                    continue

                # Process parking data
                try:
                    parking_data = ParkingSpaceData(str(xml_file))
                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f"Error loading parking data from {xml_file}: {e}")
                    continue

                # Queue frame for processing
                detector.processing_queue.put((frame, parking_data.spaces))

                # Get results with timeout
                try:
                    results = detector.results_queue.get(timeout=5.0)
                    for space_id, (occupied, confidence) in results.items():
                        parking_data.spaces[space_id].update_status(occupied, confidence)
                except queue.Empty:
                    logging.warning(f"Processing timeout for {img_file}")
                    continue

                # Update analytics
                analytics.update(parking_data.metadata['parking_id'],
                                 parking_data.spaces)

                # Visualize results
                display_frame = frame.copy()
                for space in parking_data.spaces.values():
                    # Color based on occupancy (red for occupied, green for empty)
                    color = (0, 0, 255) if space.occupied else (0, 255, 0)

                    # Draw parking space contour
                    cv2.polylines(display_frame, [space.contour], True, color, 2)

                    # Draw space ID and confidence
                    text_pos = tuple(space.contour[0])
                    cv2.putText(display_frame,
                                f"{space.id}: {space.confidence:.2f}",
                                text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2)

                # Show frame
                cv2.imshow('Parking Detection', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # Optional delay to control processing speed
                time.sleep(0.1)

        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
        except Exception as e:
            logging.error(f"Processing error: {e}")
        finally:
            # Cleanup
            detector.stop_processing_thread()
            cv2.destroyAllWindows()

            # Generate final analytics
            try:
                report = analytics.generate_report()
                logging.info(f"Analytics report generated: {report}")
                analytics.plot_occupancy_trends()
            except Exception as e:
                logging.error(f"Error generating analytics: {e}")

    except Exception as e:
        logging.error(f"Main function error: {e}")
    finally:
        logging.info("Processing completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Application error: {e}")
