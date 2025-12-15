#!/usr/bin/env python3
"""
Test script for dual camera setup with object detection.
Tests both driver and road cameras with separate windows.
"""
import cv2
import numpy as np
import time

def add_object_detection_marks(frame):
    """Add yellow object detection marks to frame."""
    if frame is None:
        return frame
        
    # Simple object detection using background subtraction and contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use background subtractor for motion detection
    if not hasattr(add_object_detection_marks, 'bg_subtractor'):
        add_object_detection_marks.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=30, detectShadows=False)
    
    fg_mask = add_object_detection_marks.bg_subtractor.apply(blurred)
    
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

def test_dual_cameras():
    """Test dual camera setup."""
    print("Testing dual camera setup...")
    
    # Try to open both cameras
    driver_cam = cv2.VideoCapture(0)  # Driver camera
    road_cam = cv2.VideoCapture(1)    # Road camera
    
    if not driver_cam.isOpened():
        print("ERROR: Driver camera (index 0) not available")
        return False
    
    print("SUCCESS: Driver camera opened successfully")
    
    road_cam_available = road_cam.isOpened()
    if road_cam_available:
        print("SUCCESS: Road camera opened successfully")
    else:
        print("WARNING: Road camera (index 1) not available - will use driver camera only")
        road_cam.release()
        road_cam = None
    
    # Set camera properties
    driver_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    driver_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    driver_cam.set(cv2.CAP_PROP_FPS, 15)
    
    if road_cam_available:
        road_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        road_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        road_cam.set(cv2.CAP_PROP_FPS, 15)
    
    print("\n" + "="*50)
    print("=== Dual Camera Test Started ===")
    print("Driver Camera: Window 1 (left)")
    if road_cam_available:
        print("Road Camera: Window 2 (right) with yellow object detection")
    print("Press 'q' to quit")
    print("="*50 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            # Read from driver camera
            ret_driver, driver_frame = driver_cam.read()
            if not ret_driver:
                print("ERROR: Failed to read from driver camera")
                break
            
            # Add driver camera label
            cv2.putText(driver_frame, "DRIVER CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add frame counter
            cv2.putText(driver_frame, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Read from road camera if available
            road_frame = None
            if road_cam_available:
                ret_road, road_frame = road_cam.read()
                if ret_road:
                    # Add road camera label
                    cv2.putText(road_frame, "ROAD CAMERA", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Add object detection marks
                    road_frame = add_object_detection_marks(road_frame)
                else:
                    road_frame = None
            
            # Display cameras in separate windows
            cv2.imshow('Driver Camera - Test', driver_frame)
            
            if road_frame is not None:
                cv2.imshow('Road Camera - Object Detection Test', road_frame)
            
            # Position windows side by side
            cv2.moveWindow('Driver Camera - Test', 50, 50)
            if road_frame is not None:
                cv2.moveWindow('Road Camera - Object Detection Test', 700, 50)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            
            frame_count += 1
            
            # Print status every 5 seconds
            if frame_count % (15 * 5) == 0:  # 15 FPS * 5 seconds
                print(f"Frame: {frame_count} | Driver: OK | Road: {'OK' if road_frame is not None else 'N/A'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        driver_cam.release()
        if road_cam_available:
            road_cam.release()
        cv2.destroyAllWindows()
        print("Test complete!")

if __name__ == "__main__":
    test_dual_cameras()