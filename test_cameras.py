#!/usr/bin/env python3
"""
Test script to test both driver and road cameras from config.json.
Shows preview of both cameras simultaneously.
"""
import cv2
import sys
import time
import numpy as np
import json
import os


def test_camera(index: int, camera_name: str = "") -> tuple:
    """
    Test if camera at index is accessible and can capture frames.
    
    Returns:
        (success: bool, cap: VideoCapture or None, frame: np.ndarray or None)
    """
    print(f"\nTesting {camera_name} camera (index {index}, /dev/video{index})...")
    
    try:
        cap = cv2.VideoCapture(index)
        time.sleep(0.3)  # Give camera time to initialize
        
        if not cap.isOpened():
            print(f"  ❌ {camera_name} Camera {index}: NOT AVAILABLE")
            return (False, None, None)
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(f"  ⚠️  {camera_name} Camera {index}: OPENED but cannot read frames")
            cap.release()
            return (False, None, None)
        
        h, w = frame.shape[:2]
        print(f"  ✅ {camera_name} Camera {index}: WORKING")
        print(f"     Resolution: {w}x{h}")
        
        return (True, cap, frame)
        
    except Exception as e:
        print(f"  ❌ {camera_name} Camera {index}: ERROR - {e}")
        return (False, None, None)


def main():
    """Test both driver and road cameras from config.json."""
    import time
    
    print("=" * 60)
    print("Driver & Road Camera Test")
    print("=" * 60)
    
    # Load config
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"\n❌ ERROR: {config_path} not found!")
        print("   Make sure you're in the raspberry folder")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        driver_index = config['cameras']['driver_cam_index']
        road_index = config['cameras']['road_cam_index']
        
        print(f"\nConfig loaded from {config_path}:")
        print(f"  Driver Camera Index: {driver_index}")
        print(f"  Road Camera Index: {road_index}")
        print("\nTesting cameras...")
        
    except Exception as e:
        print(f"\n❌ ERROR loading config: {e}")
        print("   Using default indices: 0 and 1")
        driver_index = 0
        road_index = 1
    
    # Test driver camera
    driver_success, driver_cap, driver_frame = test_camera(driver_index, "Driver")
    
    # Test road camera
    road_success, road_cap, road_frame = test_camera(road_index, "Road")
    
    # If road camera failed, try alternative indices
    if not road_success and road_index != driver_index:
        print("\n⚠️  Road camera failed. Trying alternative indices...")
        alternative_indices = [1, 3, 4, 5]
        for alt_idx in alternative_indices:
            if alt_idx != driver_index:
                print(f"\nTrying alternative index {alt_idx}...")
                alt_success, alt_cap, alt_frame = test_camera(alt_idx, f"Road (Alt {alt_idx})")
                if alt_success:
                    print(f"✅ Found working road camera at index {alt_idx}!")
                    road_success = True
                    road_cap = alt_cap
                    road_frame = alt_frame
                    road_index = alt_idx  # Update to working index
                    break
                elif alt_cap:
                    alt_cap.release()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if not driver_success:
        print("\n❌ ERROR: Driver camera not working!")
        print("   This is required for the system to work.")
        if driver_cap:
            driver_cap.release()
        if road_cap:
            road_cap.release()
        sys.exit(1)
    
    if not road_success:
        print("\n⚠️  WARNING: Road camera not working!")
        print("   System will work with driver camera only.")
        print("   Road detection features will be disabled.")
    
    # Show previews
    print("\n" + "=" * 60)
    print("SHOWING PREVIEWS")
    print("=" * 60)
    print("\nBoth camera feeds will be shown for 10 seconds")
    print("Press any key to close early, or wait for auto-close")
    print()
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Read frames
            if driver_cap:
                ret_driver, driver_frame = driver_cap.read()
                if not ret_driver:
                    break
            
            if road_cap:
                ret_road, road_frame = road_cap.read()
                if not ret_road:
                    road_frame = None
            
            if driver_frame is None:
                break
            
            # Resize frames for display (if too large)
            display_driver = driver_frame.copy()
            h, w = display_driver.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w = 640
                new_h = int(h * scale)
                display_driver = cv2.resize(display_driver, (new_w, new_h))
            
            # Add labels
            cv2.putText(display_driver, f"DRIVER CAMERA (Index {driver_index})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show driver camera
            cv2.imshow('Driver Camera', display_driver)
            
            # Show road camera if available
            if road_frame is not None:
                display_road = road_frame.copy()
                h, w = display_road.shape[:2]
                if w > 640:
                    scale = 640 / w
                    new_w = 640
                    new_h = int(h * scale)
                    display_road = cv2.resize(display_road, (new_w, new_h))
                
                cv2.putText(display_road, f"ROAD CAMERA (Index {road_index})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Road Camera', display_road)
            else:
                # Show placeholder if road camera not available
                placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(placeholder, "ROAD CAMERA", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(placeholder, "NOT AVAILABLE", (30, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('Road Camera', placeholder)
            
            # Check for key press or timeout
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Any key pressed
                break
            
            elapsed = time.time() - start_time
            if elapsed >= 10:  # 10 second timeout
                break
            
            frame_count += 1
        
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during preview: {e}")
    finally:
        # Cleanup
        if driver_cap:
            driver_cap.release()
        if road_cap:
            road_cap.release()
        cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if driver_success:
        print(f"\n✅ Driver Camera (Index {driver_index}): WORKING")
    else:
        print(f"\n❌ Driver Camera (Index {driver_index}): FAILED")
    
    if road_success:
        print(f"✅ Road Camera (Index {road_index}): WORKING")
        print(f"\n📝 Your config.json is correct!")
        print(f"   Driver: {driver_index}, Road: {road_index}")
    else:
        print(f"⚠️  Road Camera: NOT AVAILABLE (optional)")
        original_road = config.get('cameras', {}).get('road_cam_index', 'unknown')
        if road_index != original_road:
            print(f"\n💡 SUGGESTION:")
            print(f"   Original config had road_cam_index: {original_road}")
            print(f"   But that camera doesn't work.")
            print(f"   You can either:")
            print(f"   1. Use driver camera for both (set road_cam_index: {driver_index})")
            print(f"   2. Try a different camera index")
            print(f"   3. Continue without road camera (system works fine)")
    
    print("\n" + "=" * 60)
    print("READY TO RUN")
    print("=" * 60)
    
    if driver_success:
        print("\n✅ System is ready!")
        print("   Run: python3 main.py")
        if not road_success:
            print("\n   Note: Road detection will be disabled")
            print("   All other features (PERCLOS, blinks, yawn, alerts) will work!")
    else:
        print("\n❌ Fix driver camera first!")
        print("   Check camera connection and permissions")


if __name__ == "__main__":
    main()

