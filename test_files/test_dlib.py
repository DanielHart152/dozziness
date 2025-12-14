#!/usr/bin/env python3
"""Test dlib installation and shape predictor."""

import cv2
import os

def test_dlib():
    try:
        import dlib
        print("✅ dlib imported successfully")
        
        # Test shape predictor file
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            print("✅ Shape predictor file found")
            try:
                predictor = dlib.shape_predictor(predictor_path)
                print("✅ Shape predictor loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to load shape predictor: {e}")
                return False
        else:
            print(f"❌ Shape predictor not found: {predictor_path}")
            print("Download with: wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            return False
            
    except ImportError:
        print("❌ dlib not installed")
        print("Install with: pip3 install dlib")
        return False

if __name__ == "__main__":
    test_dlib()