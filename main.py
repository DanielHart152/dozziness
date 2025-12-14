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
        if metrics.get('face_detected', False):
            face_bbox = metrics.get('face_bbox')
            if face_bbox is not None:
                try:
                    x, y, w, h = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
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
                except (ValueError, TypeError, IndexError):
                    # Fallback: draw a simple indicator if bbox format is wrong
                    cv2.putText(frame, "BBOX FORMAT ERROR", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # Fallback: draw a simple indicator if face detected but no bbox
                cv2.putText(frame, "FACE DETECTED (NO BBOX)", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
        
        hud_x = 10
        hud_y = 10
        hud_w = 280
        hud_h = 245
        
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 30
        line_height = 25
        
        cv2.putText(frame, "DROWSINESS MONITOR", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        score = metrics.get('risk_score', 0.0)
        score_color = (0, 255, 0) if score < 50 else (0, 165, 255) if score < 75 else (0, 0, 255)
        cv2.putText(frame, f"Risk Score: {score:.1f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, score_color, self.font_thickness)
        y_offset += line_height
        
        perclos = metrics.get('perclos', 0.0)
        cv2.putText(frame, f"PERCLOS: {perclos:.3f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness)
        y_offset += line_height
        
        bpm = metrics.get('blinks_per_min', 0.0)
        cv2.putText(frame, f"Blinks/min: {bpm:.1f}", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness)
        y_offset += line_height
        
        # Speed display (placeholder for future sensor integration)
        cv2.putText(frame, "Speed: 0 km/h (No sensor)", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (128, 128, 128), self.font_thickness)
        y_offset += line_height
        
        face_status = "Face: YES" if metrics.get('face_detected', False) else "Face: NO"
        face_color = (0, 255, 0) if metrics.get('face_detected', False) else (0, 0, 255)
        cv2.putText(frame, face_status, (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        y_offset += line_height
        
        # Face tracking status
        if metrics.get('face_tracked', False):
            track_color = (0, 255, 255)  # Yellow for tracking
            cv2.putText(frame, "Tracking: ON", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
        else:
            cv2.putText(frame, "Tracking: OFF", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        y_offset += line_height
        
        # Face movement detection
        if metrics.get('face_movement_detected', False):
            cv2.putText(frame, "FACE MOVEMENT!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            y_offset += line_height
        
        if metrics.get('yawn_detected', False):
            cv2.putText(frame, "YAWN DETECTED!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += line_height
        
        if metrics.get('nod_detected', False):
            cv2.putText(frame, "HEAD NOD!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += line_height
        
        if metrics.get('texting_detected', False):
            cv2.putText(frame, "TEXTING DETECTED!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            y_offset += line_height
        
        if metrics.get('car_close', False):
            cv2.putText(frame, "CAR CLOSE!", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            y_offset += line_height
        
        if metrics.get('alert_triggered', False):
            cv2.putText(frame, "*** ALERT ***", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.alert_manager.is_muted:
            mute_remaining = max(0, int(self.alert_manager.mute_until - time.time()))
            cv2.putText(frame, f"MUTED: {mute_remaining}s", (hud_x + 5, hud_y + hud_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        cv2.putText(frame, "Press 'q' to quit, 'm' to mute", (hud_x + 5, hud_y + hud_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
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
    
    def combine_camera_views(self, driver_frame: np.ndarray, road_frame: np.ndarray) -> np.ndarray:
        """Combine driver and road camera frames side by side."""
        h1, w1 = driver_frame.shape[:2]
        h2, w2 = road_frame.shape[:2]
        
        # Resize both frames to same height for side-by-side display
        target_height = max(h1, h2)
        driver_resized = cv2.resize(driver_frame, (int(w1 * target_height / h1), target_height))
        road_resized = cv2.resize(road_frame, (int(w2 * target_height / h2), target_height))
        
        # Add labels
        cv2.putText(driver_resized, "DRIVER CAM", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(road_resized, "ROAD CAM", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Combine horizontally
        combined = np.hstack([driver_resized, road_resized])
        return combined
    
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
            alert_triggered = self.alert_manager.trigger_alert()
        
        metrics = {
            'perclos': perclos, 'blinks_per_min': blinks_per_min, 'risk_score': risk_score,
            'alert_triggered': alert_triggered, 'yawn_detected': driver_result.get('yawn_detected', False),
            'nod_detected': driver_result.get('nod_detected', False), 'texting_detected': texting_detected,
            'car_close': car_close, 'left_ear': driver_result.get('left_ear', 0.0),
            'right_ear': driver_result.get('right_ear', 0.0), 'mar': driver_result.get('mar', 0.0),
            'head_pitch': driver_result.get('head_pitch', 0.0), 'face_detected': driver_result.get('face_detected', False),
            'face_tracked': driver_result.get('face_tracked', False), 'face_movement_detected': driver_result.get('face_movement_detected', False)
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
                
                # Display both cameras
                try:
                    if road_frame is not None:
                        # Combine both frames side by side
                        combined_frame = self.combine_camera_views(driver_frame, road_frame)
                        cv2.imshow('Driver Drowsiness Monitor - Driver | Road', combined_frame)
                    else:
                        # Only driver camera available
                        cv2.imshow('Driver Drowsiness Monitor', driver_frame)
                except:
                    pass
                
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

