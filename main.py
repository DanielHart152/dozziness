"""
Main driver drowsiness detection system for Raspberry Pi.
Coordinates camera capture, vision processing, scoring, alerts, and logging.
"""
import cv2
import time
import sys
import numpy as np
from typing import Optional, Dict

from utils_config import load_config, validate_config
from vision import FaceDetector, RoadDetector
from score import RiskScorer
from io_alerts import AlertManager
from logger import DataLogger


class DrowsinessMonitor:
    """Main drowsiness monitoring system for Raspberry Pi."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize drowsiness monitor."""
        self.config = load_config(config_path)
        validate_config(self.config)
        
        self.face_detector = FaceDetector(self.config)
        self.road_detector = RoadDetector(self.config)
        self.scorer = RiskScorer(self.config)
        self.alert_manager = AlertManager(self.config)
        self.logger = DataLogger(self.config)
        
        self.driver_cam_index = self.config['cameras']['driver_cam_index']
        self.road_cam_index = self.config['cameras']['road_cam_index']
        self.width = self.config['cameras']['resolution']['width']
        self.height = self.config['cameras']['resolution']['height']
        self.target_fps = self.config['cameras']['fps']
        self.frame_time = 1.0 / self.target_fps
        
        self.driver_cam: Optional[cv2.VideoCapture] = None
        self.road_cam: Optional[cv2.VideoCapture] = None
        
        self.show_hud = self.config['display']['show_hud']
        self.font_scale = self.config['display']['font_scale']
        self.font_thickness = self.config['display']['font_thickness']
        
        self.running = False
        self.frame_count = 0
    
    def initialize_cameras(self) -> bool:
        """Initialize camera captures."""
        print("Initializing cameras...")
        
        try:
            self.driver_cam = cv2.VideoCapture(self.driver_cam_index)
            time.sleep(0.5)
            
            if not self.driver_cam.isOpened():
                print(f"❌ ERROR: Could not open driver camera at index {self.driver_cam_index}")
                print("\nTroubleshooting:")
                print("  1. Check camera is connected: ls -l /dev/video*")
                print("  2. Try different camera index in config.json")
                print("  3. Check camera permissions")
                return False
            
            self.driver_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.driver_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.driver_cam.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            ret, test_frame = self.driver_cam.read()
            if not ret or test_frame is None:
                print("❌ ERROR: Camera opened but cannot read frames")
                return False
            
            actual_width = int(self.driver_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.driver_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ Driver camera initialized: {actual_width}x{actual_height} @ {self.target_fps} FPS")
            
            if self.road_cam_index != self.driver_cam_index:
                try:
                    self.road_cam = cv2.VideoCapture(self.road_cam_index)
                    time.sleep(0.3)
                    
                    if self.road_cam.isOpened():
                        self.road_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.road_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self.road_cam.set(cv2.CAP_PROP_FPS, self.target_fps)
                        road_w = int(self.road_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                        road_h = int(self.road_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"✅ Road camera initialized: {road_w}x{road_h}")
                    else:
                        self.road_cam = None
                        print("⚠️  Road camera not available - continuing with driver camera only")
                except Exception as e:
                    self.road_cam = None
                    print(f"⚠️  Road camera error: {e}")
            else:
                self.road_cam = None
                print("   (Using same camera for driver and road - road detection disabled)")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR initializing cameras: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def draw_tracking_visuals(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Draw face bounding box and tracking line."""
        # Always draw face bounding box when face is detected
        face_detected = metrics.get('face_detected', False)
        face_bbox = metrics.get('face_bbox')
        print(f"DISPLAY DEBUG: face_detected={face_detected}, face_bbox={face_bbox}")
        
        if face_detected:
            if face_bbox is not None:
                try:
                    x, y, w, h = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
                    print(f"DISPLAY DEBUG: Drawing bbox at ({x}, {y}, {w}, {h})")
                    # Green box for active tracking, blue for detection only
                    color = (0, 255, 0) if metrics.get('face_tracked', False) else (255, 0, 0)
                    thickness = 2 if metrics.get('face_tracked', False) else 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # Add label
                    label = "TRACKING" if metrics.get('face_tracked', False) else "DETECTED"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Always draw center point
                    center_x, center_y = x + w//2, y + h//2
                    cv2.circle(frame, (center_x, center_y), 3, color, -1)
                except (ValueError, TypeError, IndexError) as e:
                    print(f"DISPLAY DEBUG: Exception drawing bbox: {e}")
                    # Fallback: draw a simple indicator if bbox format is wrong
                    cv2.putText(frame, "BBOX FORMAT ERROR", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                print("DISPLAY DEBUG: face_bbox is None")
                # Fallback: draw a simple indicator if face detected but no bbox
                cv2.putText(frame, "FACE DETECTED (NO BBOX)", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            print("DISPLAY DEBUG: No face detected")
        
        # Draw tracking line (movement trail)
        current_center = metrics.get('tracking_center')
        last_center = metrics.get('last_center')
        
        if current_center and last_center and current_center != last_center:
            # Draw line from last position to current position
            cv2.line(frame, last_center, current_center, (0, 255, 255), 2)
            # Draw movement indicator
            cv2.putText(frame, "MOVEMENT", (current_center[0] - 30, current_center[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def draw_hud(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Draw HUD overlay on frame."""
        # First draw tracking visuals
        frame = self.draw_tracking_visuals(frame, metrics)
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Position HUD in top-left corner with larger size for bigger display
        hud_x = 5
        hud_y = 5
        hud_w = 350
        hud_h = 280
        
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 25
        line_height = 30
        
        # Title with larger font
        cv2.putText(frame, "DROWSINESS MONITOR", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height + 5
        
        # Risk Score - most prominent display
        score = metrics.get('risk_score', 0.0)
        score_color = (0, 255, 0) if score < 50 else (0, 165, 255) if score < 75 else (0, 0, 255)
        cv2.putText(frame, f"Risk Score: {score:.1f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 3)
        y_offset += line_height + 5
        
        # PERCLOS with larger font
        perclos = metrics.get('perclos', 0.0)
        cv2.putText(frame, f"PERCLOS: {perclos:.3f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # Blinks per minute with larger font
        bpm = metrics.get('blinks_per_min', 0.0)
        cv2.putText(frame, f"Blinks/min: {bpm:.1f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # Speed display (placeholder for future sensor integration)
        cv2.putText(frame, "Speed: 0 km/h (No sensor)", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        y_offset += line_height
        
        # Face detection status with larger font
        face_status = "Face: YES" if metrics.get('face_detected', False) else "Face: NO"
        face_color = (0, 255, 0) if metrics.get('face_detected', False) else (0, 0, 255)
        cv2.putText(frame, face_status, (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        y_offset += line_height
        
        # Face tracking status
        if metrics.get('face_tracked', False):
            track_color = (0, 255, 255)  # Yellow for tracking
            cv2.putText(frame, "Tracking: ON", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
        else:
            cv2.putText(frame, "Tracking: OFF", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        y_offset += line_height
        
        # Face movement detection
        if metrics.get('face_movement_detected', False):
            cv2.putText(frame, "FACE MOVEMENT!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            y_offset += line_height
        
        if metrics.get('yawn_detected', False):
            cv2.putText(frame, "YAWN DETECTED!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += line_height
        
        if metrics.get('nod_detected', False):
            cv2.putText(frame, "HEAD NOD!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += line_height
        
        if metrics.get('texting_detected', False):
            cv2.putText(frame, "TEXTING DETECTED!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += line_height
        
        if metrics.get('car_close', False):
            cv2.putText(frame, "CAR CLOSE!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += line_height
        
        if metrics.get('alert_triggered', False):
            cv2.putText(frame, "*** ALERT ***", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        if self.alert_manager.is_muted:
            mute_remaining = max(0, int(self.alert_manager.mute_until - time.time()))
            cv2.putText(frame, f"MUTED: {mute_remaining}s", (hud_x + 5, hud_y + hud_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        cv2.putText(frame, "Press 'q' to quit, 'm' to mute", (hud_x + 5, hud_y + hud_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_road_hud(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Draw HUD overlay on road camera frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        hud_x = 10
        hud_y = 40  # Start below the "ROAD CAM" label
        hud_w = 250
        hud_h = 180
        
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 60
        line_height = 25
        
        cv2.putText(frame, "ROAD MONITOR", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        
        # Car proximity status
        car_close = metrics.get('car_close', False)
        car_status = "CAR: CLOSE!" if car_close else "CAR: SAFE"
        car_color = (0, 0, 255) if car_close else (0, 255, 0)
        cv2.putText(frame, car_status, (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, car_color, self.font_thickness)
        y_offset += line_height
        
        # Texting detection
        texting_detected = metrics.get('texting_detected', False)
        texting_status = "TEXTING: YES!" if texting_detected else "TEXTING: NO"
        texting_color = (0, 0, 255) if texting_detected else (0, 255, 0)
        cv2.putText(frame, texting_status, (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, texting_color, self.font_thickness)
        y_offset += line_height
        
        # Overall risk score (also show on road cam)
        score = metrics.get('risk_score', 0.0)
        score_color = (0, 255, 0) if score < 50 else (0, 165, 255) if score < 75 else (0, 0, 255)
        cv2.putText(frame, f"Risk: {score:.1f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, score_color, self.font_thickness)
        y_offset += line_height
        
        # Alert status
        if metrics.get('alert_triggered', False):
            cv2.putText(frame, "*** ALERT ***", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += line_height
        
        return frame
    
    def add_object_detection_marks(self, frame: np.ndarray) -> np.ndarray:
        """Add yellow object detection marks to road camera frame."""
        if frame is None:
            return frame
            
        # Simple object detection using background subtraction and contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use background subtractor for motion detection
        if not hasattr(self, 'bg_subtractor_objects'):
            self.bg_subtractor_objects = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=False)
        
        fg_mask = self.bg_subtractor_objects.apply(blurred)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw yellow bounding boxes around detected objects
        h, w = frame.shape[:2]
        min_area = (w * h) * 0.001  # Minimum 0.1% of frame area
        max_area = (w * h) * 0.3    # Maximum 30% of frame area
        
        object_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Draw yellow bounding box
                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 255), 2)
                
                # Add object label
                cv2.putText(frame, f"OBJ{object_count+1}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Draw center point
                center_x, center_y = x + cw//2, y + ch//2
                cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
                
                object_count += 1
        
        # Add object count to frame
        cv2.putText(frame, f"Objects: {object_count}", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def process_frame(self, driver_frame: np.ndarray, road_frame: Optional[np.ndarray] = None) -> Dict:
        """Process frames and return metrics."""
        driver_result = self.face_detector.process_frame(driver_frame)
        perclos = self.face_detector.get_perclos()
        blinks_per_min = self.face_detector.get_blinks_per_minute()
        
        car_close = False
        texting_detected = False
        
        if road_frame is not None:
            car_close = self.road_detector.detect_car_close(road_frame)
        
        texting_detected = self.road_detector.detect_texting(driver_frame)
        
        risk_score = self.scorer.calculate_score(
            perclos=perclos, blinks_per_min=blinks_per_min,
            yawn_detected=driver_result.get('yawn_detected', False),
            nod_detected=driver_result.get('nod_detected', False),
            texting_detected=texting_detected, car_close=car_close
        )
        
        should_alert = self.scorer.should_alert(score=risk_score)
        alert_triggered = False
        
        if should_alert:
            alert_triggered = self.alert_manager.trigger_alert(risk_score=risk_score)
        
        metrics = {
            'perclos': perclos, 'blinks_per_min': blinks_per_min, 'risk_score': risk_score,
            'alert_triggered': alert_triggered, 'yawn_detected': driver_result.get('yawn_detected', False),
            'nod_detected': driver_result.get('nod_detected', False), 'texting_detected': texting_detected,
            'car_close': car_close, 'left_ear': driver_result.get('left_ear', 0.0),
            'right_ear': driver_result.get('right_ear', 0.0), 'mar': driver_result.get('mar', 0.0),
            'head_pitch': driver_result.get('head_pitch', 0.0), 'face_detected': driver_result.get('face_detected', False),
            'face_tracked': driver_result.get('face_tracked', False), 'face_movement_detected': driver_result.get('face_movement_detected', False),
            'face_bbox': driver_result.get('face_bbox'), 'tracking_center': driver_result.get('tracking_center'), 'last_center': driver_result.get('last_center')
        }
        
        return metrics
    
    def run(self):
        """Main processing loop."""
        if not self.initialize_cameras():
            print("Failed to initialize cameras. Exiting.")
            return
        
        print("\n" + "="*50)
        print("=== Drowsiness Monitor Started ===")
        print(f"Driver Camera: Index {self.driver_cam_index}")
        if self.road_cam is not None:
            print(f"Road Camera: Index {self.road_cam_index}")
        print("Press 'q' to quit, 'm' to mute alerts for 30s")
        print("="*50 + "\n")
        
        self.running = True
        
        try:
            while self.running:
                frame_start = time.time()
                
                ret_driver, driver_frame = self.driver_cam.read()
                if not ret_driver or driver_frame is None:
                    print("Error: Failed to read from driver camera")
                    break
                
                h, w = driver_frame.shape[:2]
                if w != self.width or h != self.height:
                    driver_frame = cv2.resize(driver_frame, (self.width, self.height))
                
                road_frame = None
                if self.road_cam is not None:
                    ret_road, road_frame = self.road_cam.read()
                    if ret_road and road_frame is not None:
                        h, w = road_frame.shape[:2]
                        if w != self.width or h != self.height:
                            road_frame = cv2.resize(road_frame, (self.width, self.height))
                    else:
                        road_frame = None
                
                metrics = self.process_frame(driver_frame, road_frame)
                
                if self.show_hud:
                    driver_frame = self.draw_hud(driver_frame, metrics)
                    # Also draw HUD on road camera if available
                    if road_frame is not None:
                        road_frame = self.draw_road_hud(road_frame, metrics)
                
                self.logger.log(metrics)
                
                # Display cameras in separate windows
                try:
                    # Always show driver camera
                    cv2.imshow('Driver Camera - Drowsiness Monitor', driver_frame)
                    
                    # Show road camera with object detection if available
                    if road_frame is not None:
                        # Add yellow object detection marks
                        road_frame_with_objects = self.add_object_detection_marks(road_frame.copy())
                        cv2.imshow('Road Camera - Object Detection', road_frame_with_objects)
                    
                    # Position windows side by side (optional)
                    cv2.moveWindow('Driver Camera - Drowsiness Monitor', 50, 50)
                    if road_frame is not None:
                        cv2.moveWindow('Road Camera - Object Detection', 700, 50)
                        
                except Exception as e:
                    print(f"Display error: {e}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('m'):
                    self.alert_manager.mute()
                
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.frame_count += 1
                
                if self.frame_count % (self.target_fps * 5) == 0:
                    print(f"Frame: {self.frame_count} | Score: {metrics['risk_score']:.1f} | "
                          f"PERCLOS: {metrics['perclos']:.3f} | BPM: {metrics['blinks_per_min']:.1f} | "
                          f"Alert: {metrics['alert_triggered']}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up...")
        self.running = False
        
        if self.driver_cam is not None:
            self.driver_cam.release()
        
        if self.road_cam is not None:
            self.road_cam.release()
        
        try:
            cv2.destroyWindow('Driver Camera - Drowsiness Monitor')
            cv2.destroyWindow('Road Camera - Object Detection')
            cv2.destroyAllWindows()
        except:
            pass
        
        self.alert_manager.cleanup()
        print("Cleanup complete.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection System - Raspberry Pi')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file (default: config.json)')
    
    args = parser.parse_args()
    
    try:
        monitor = DrowsinessMonitor(args.config)
        monitor.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

