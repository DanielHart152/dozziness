"""
Demo mode for drowsiness detection system - shows interface without camera
"""
import cv2
import numpy as np
import time
import math

def create_demo_frame(width=640, height=480):
    """Create a demo frame simulating a driver."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(height):
        intensity = int(30 + (y / height) * 50)
        frame[y, :] = [intensity, intensity//2, intensity//3]
    
    # Add "DEMO MODE" text
    cv2.putText(frame, "DEMO MODE - No Camera", (width//2 - 120, height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
    
    return frame

def draw_demo_hud(frame, time_elapsed):
    """Draw HUD with simulated data."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    hud_x = 10
    hud_y = 10
    hud_w = 280
    hud_h = 245
    
    # Semi-transparent background
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 30
    line_height = 25
    
    # Title
    cv2.putText(frame, "DROWSINESS MONITOR", (hud_x + 5, hud_y + y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_offset += line_height
    
    # Simulated risk score (oscillates)
    risk_score = 25 + 20 * math.sin(time_elapsed * 0.5)
    score_color = (0, 255, 0) if risk_score < 50 else (0, 165, 255) if risk_score < 75 else (0, 0, 255)
    cv2.putText(frame, f"Risk Score: {risk_score:.1f}", (hud_x + 5, hud_y + y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)
    y_offset += line_height
    
    # Simulated PERCLOS
    perclos = 0.05 + 0.03 * math.sin(time_elapsed * 0.3)
    cv2.putText(frame, f"PERCLOS: {perclos:.3f}", (hud_x + 5, hud_y + y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    # Simulated blinks per minute
    bpm = 15 + 5 * math.sin(time_elapsed * 0.2)
    cv2.putText(frame, f"Blinks/min: {bpm:.1f}", (hud_x + 5, hud_y + y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    # Speed display - THIS IS THE NEW FEATURE
    cv2.putText(frame, "Speed: 0 km/h (No sensor)", (hud_x + 5, hud_y + y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    y_offset += line_height
    
    # Face detection status
    cv2.putText(frame, "Face: DEMO", (hud_x + 5, hud_y + y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += line_height
    
    # Simulate occasional alerts
    if int(time_elapsed) % 10 == 0 and (time_elapsed % 1) < 0.5:
        cv2.putText(frame, "YAWN DETECTED!", (hud_x + 5, hud_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        y_offset += line_height
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit, 'm' to mute", (hud_x + 5, hud_y + hud_h + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def main():
    """Run demo mode."""
    print("=" * 50)
    print("=== DROWSINESS MONITOR DEMO ===")
    print("Shows interface with speed display")
    print("Press 'q' to quit")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        while True:
            time_elapsed = time.time() - start_time
            
            # Create demo frame
            frame = create_demo_frame(640, 480)
            
            # Add HUD with speed display
            frame = draw_demo_hud(frame, time_elapsed)
            
            # Display
            cv2.imshow('Driver Drowsiness Monitor - DEMO', frame)
            
            # Handle input
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                print("Mute button pressed (demo)")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    finally:
        cv2.destroyAllWindows()
        print("Demo complete!")

if __name__ == "__main__":
    main()