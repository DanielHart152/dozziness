#!/usr/bin/env python3
"""
Video demo for drowsiness detection system - uses video file instead of camera
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


class VideoDrowsinessMonitor:
    """Video-based drowsiness monitoring system for testing."""
    
    def __init__(self, config_path: str = "config.json", video_path: str = None, debug_mode: bool = False):
        """Initialize video drowsiness monitor."""
        self.config = load_config(config_path)
        validate_config(self.config)
        
        self.face_detector = FaceDetector(self.config)
        self.road_detector = RoadDetector(self.config)
        self.scorer = RiskScorer(self.config)
        self.alert_manager = AlertManager(self.config)
        self.logger = DataLogger(self.config)
        
        self.video_path = video_path
        self.debug_mode = debug_mode
        self.width = self.config['cameras']['resolution']['width']
        self.height = self.config['cameras']['resolution']['height']
        self.target_fps = self.config['cameras']['fps']
        self.frame_time = 1.0 / self.target_fps
        
        self.video_cap: Optional[cv2.VideoCapture] = None
        
        self.show_hud = self.config['display']['show_hud']
        self.font_scale = self.config['display']['font_scale']
        self.font_thickness = self.config['display']['font_thickness']
        
        self.running = False
        self.frame_count = 0
        self.paused = debug_mode
    
    def initialize_video(self) -> bool:
        """Initialize video capture."""
        print("Initializing video...")
        
        if not self.video_path:
            print("❌ ERROR: No video path provided")
            print("Usage: python3 video_demo.py path/to/video.mp4")
            return False
        
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            
            if not self.video_cap.isOpened():
                print(f"❌ ERROR: Could not open video file: {self.video_path}")
                return False
            
            # Get video properties
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            print(f"✅ Video loaded: {self.video_path}")
            print(f"   Frames: {total_frames}, FPS: {video_fps:.1f}, Duration: {duration:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR initializing video: {e}")
            return False
    
    def draw_tracking_visuals(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Draw face bounding box and tracking line."""
        # Draw face bounding box
        face_bbox = metrics.get('face_bbox')
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Green box for active tracking, blue for detection only
            color = (0, 255, 0) if metrics.get('face_tracked', False) else (255, 0, 0)
            thickness = 2 if metrics.get('face_tracked', False) else 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label = "TRACKING" if metrics.get('face_tracked', False) else "DETECTED"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw tracking line (movement trail)
        current_center = metrics.get('tracking_center')
        last_center = metrics.get('last_center')
        
        if current_center and last_center and current_center != last_center:
            # Draw line from last position to current position
            cv2.line(frame, last_center, current_center, (0, 255, 255), 2)
            # Draw current center point
            cv2.circle(frame, current_center, 3, (0, 255, 255), -1)
            # Draw last center point
            cv2.circle(frame, last_center, 2, (128, 128, 128), -1)
        elif current_center:
            # Just draw current center if no movement
            cv2.circle(frame, current_center, 3, (0, 255, 0), -1)
        
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
        
        # Face tracking status with type indicator
        if metrics.get('face_tracked', False):
            # Check what type of tracker is being used
            tracker_type = "UNKNOWN"
            track_color = (128, 128, 128)
            
            if hasattr(self.face_detector, 'face_tracker'):
                if self.face_detector.face_tracker == "simple":
                    tracker_type = "FALLBACK"
                    track_color = (0, 165, 255)  # Orange for fallback
                elif self.face_detector.face_tracker is not None:
                    tracker_name = type(self.face_detector.face_tracker).__name__
                    tracker_type = f"REAL-{tracker_name}"
                    track_color = (0, 255, 255)  # Yellow for real tracking
                else:
                    tracker_type = "DETECTION"
                    track_color = (255, 0, 0)  # Blue for detection only
            
            cv2.putText(frame, f"Track: {tracker_type}", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
        else:
            cv2.putText(frame, "Track: OFF", (hud_x + 5, hud_y + y_offset),
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
        
        if metrics.get('alert_triggered', False):
            cv2.putText(frame, "*** ALERT ***", (hud_x + 5, hud_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.alert_manager.is_muted:
            mute_remaining = max(0, int(self.alert_manager.mute_until - time.time()))
            cv2.putText(frame, f"MUTED: {mute_remaining}s", (hud_x + 5, hud_y + hud_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        if self.debug_mode:
            cv2.putText(frame, "DEBUG: Left/Right arrows, Space=pause, 'q'=quit", (hud_x + 5, hud_y + hud_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            if self.paused:
                cv2.putText(frame, "PAUSED", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 'q' to quit, 'm' to mute", (hud_x + 5, hud_y + hud_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame and return metrics."""
        driver_result = self.face_detector.process_frame(frame)
        perclos = self.face_detector.get_perclos()
        blinks_per_min = self.face_detector.get_blinks_per_minute()
        
        # Simulate texting detection on video
        texting_detected = self.road_detector.detect_texting(frame)
        
        risk_score = self.scorer.calculate_score(
            perclos=perclos, blinks_per_min=blinks_per_min,
            yawn_detected=driver_result.get('yawn_detected', False),
            nod_detected=driver_result.get('nod_detected', False),
            texting_detected=texting_detected, car_close=False
        )
        
        should_alert = self.scorer.should_alert(score=risk_score)
        alert_triggered = False
        
        if should_alert:
            alert_triggered = self.alert_manager.trigger_alert()
        
        metrics = {
            'perclos': perclos, 'blinks_per_min': blinks_per_min, 'risk_score': risk_score,
            'alert_triggered': alert_triggered, 'yawn_detected': driver_result.get('yawn_detected', False),
            'nod_detected': driver_result.get('nod_detected', False), 'texting_detected': texting_detected,
            'car_close': False, 'left_ear': driver_result.get('left_ear', 0.0),
            'right_ear': driver_result.get('right_ear', 0.0), 'mar': driver_result.get('mar', 0.0),
            'head_pitch': driver_result.get('head_pitch', 0.0), 'face_detected': driver_result.get('face_detected', False),
            'face_tracked': driver_result.get('face_tracked', False), 'face_movement_detected': driver_result.get('face_movement_detected', False),
            'face_bbox': driver_result.get('face_bbox'), 'tracking_center': driver_result.get('tracking_center'),
            'last_center': driver_result.get('last_center')
        }
        
        return metrics
    
    def run(self):
        """Main processing loop."""
        if not self.initialize_video():
            print("Failed to initialize video. Exiting.")
            return
        
        print("\\n" + "="*50)
        if self.debug_mode:
            print("=== Video Drowsiness Monitor (DEBUG MODE) ===")
            print("Controls: Left/Right arrows = prev/next frame, Space = play/pause")
        else:
            print("=== Video Drowsiness Monitor Started ===")
            print("Press 'q' to quit, 'r' to restart video")
        print(f"Video: {self.video_path}")
        print("="*50 + "\\n")
        
        self.running = True
        current_frame = None
        
        # Read first frame for debug mode
        if self.debug_mode:
            ret, frame = self.video_cap.read()
            if ret:
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                metrics = self.process_frame(frame)
                if self.show_hud:
                    frame = self.draw_hud(frame, metrics)
                current_frame = frame.copy()
                
                print(f"\\n=== FRAME 0 DEBUG ===")
                print(f"Face detected: {metrics['face_detected']}")
                print(f"PERCLOS: {metrics['perclos']:.4f}")
                print(f"Blinks/min: {metrics['blinks_per_min']:.1f}")
                print(f"Risk score: {metrics['risk_score']:.2f}")
                print("Press 'a'/'d' to navigate, Space to play, 'q' to quit")
        
        try:
            while self.running:
                frame_start = time.time()
                
                if self.debug_mode and self.paused:
                    if current_frame is not None:
                        try:
                            cv2.imshow('Video Drowsiness Monitor', current_frame)
                            cv2.waitKey(1)  # Refresh display
                        except:
                            pass
                    
                    key = cv2.waitKey(0)
                    print(f"Key pressed: {key}")  # Debug key codes
                    
                    if key == ord('q') or key == 27:  # q or ESC
                        print("Quit requested by user")
                        break
                    elif key == ord(' ') or key == 32:  # Space
                        self.paused = False
                        print("Resuming playback...")
                        continue
                    elif key == 2424832 or key == 65361:  # Left arrow (different systems)
                        current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                        new_pos = max(0, current_pos - 2)
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                        print(f"DEBUG: Frame {new_pos}")
                    elif key == 2555904 or key == 65363:  # Right arrow (different systems)
                        current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print(f"DEBUG: Frame {current_pos}")
                    elif key == ord('a'):  # Alternative: 'a' for left
                        current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                        new_pos = max(0, current_pos - 2)
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                        print(f"DEBUG: Frame {new_pos} (prev)")
                    elif key == ord('d'):  # Alternative: 'd' for right
                        current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print(f"DEBUG: Frame {current_pos} (next)")
                    else:
                        print(f"Unknown key. Use: Space=play, a/d=prev/next, q=quit")
                        continue
                
                ret, frame = self.video_cap.read()
                if not ret:
                    if self.debug_mode:
                        print("End of video reached.")
                        break
                    else:
                        print("End of video reached. Restarting...")
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                
                # Resize frame if needed
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                metrics = self.process_frame(frame)
                
                if self.show_hud:
                    frame = self.draw_hud(frame, metrics)
                
                current_frame = frame.copy()
                
                if self.debug_mode:
                    current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"\\n=== FRAME {current_pos} DEBUG ===")
                    print(f"Face detected: {metrics['face_detected']}")
                    print(f"PERCLOS: {metrics['perclos']:.4f}")
                    print(f"Blinks/min: {metrics['blinks_per_min']:.1f}")
                    print(f"Risk score: {metrics['risk_score']:.2f}")
                    print(f"Alert: {metrics['alert_triggered']}")
                    if metrics['yawn_detected']:
                        print("YAWN DETECTED!")
                    if metrics['nod_detected']:
                        print("NOD DETECTED!")
                
                self.logger.log(metrics)
                
                # Display frame
                try:
                    cv2.imshow('Video Drowsiness Monitor', frame)
                    if self.debug_mode:
                        cv2.waitKey(1)  # Ensure frame is displayed
                except:
                    pass
                
                if not self.debug_mode or not self.paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('r'):
                        print("Restarting video...")
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    elif key == ord('m'):
                        self.alert_manager.mute()
                    elif (key == ord(' ') or key == 32) and self.debug_mode:  # Space
                        self.paused = True
                        print("Paused. Use a/d keys to navigate frames.")
                
                if not self.debug_mode:
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, self.frame_time - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                self.frame_count += 1
                
                if not self.debug_mode and self.frame_count % (self.target_fps * 5) == 0:
                    print(f"Frame: {self.frame_count} | Score: {metrics['risk_score']:.1f} | "
                          f"PERCLOS: {metrics['perclos']:.3f} | BPM: {metrics['blinks_per_min']:.1f}")
        
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("\\nCleaning up...")
        self.running = False
        
        if self.video_cap is not None:
            self.video_cap.release()
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        self.alert_manager.cleanup()
        print("Cleanup complete.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Drowsiness Detection Demo')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file (default: config.json)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (frame-by-frame control)')
    
    args = parser.parse_args()
    
    try:
        monitor = VideoDrowsinessMonitor(args.config, args.video, args.debug)
        monitor.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()