# Dual Camera Setup - Driver Drowsiness Detection

## Overview

The system now supports dual camera operation with separate windows:

1. **Driver Camera** - Monitors driver for drowsiness indicators
2. **Road Camera** - Monitors road ahead with yellow object detection marks

## Features

### Driver Camera Window
- Face detection and tracking with green/blue bounding boxes
- Real-time drowsiness metrics (PERCLOS, blinks per minute, risk score)
- Eye state monitoring and blink detection
- Yawn and head nod detection (with dlib)
- Movement tracking indicators

### Road Camera Window  
- **Yellow object detection marks** on moving objects
- Background subtraction for motion detection
- Bounding boxes around detected objects
- Object counting display
- Car proximity detection for alerts

## Configuration

Edit `config.json` to set camera indices:

```json
{
  "cameras": {
    "driver_cam_index": 0,    // Usually the built-in or first USB camera
    "road_cam_index": 1,      // Second USB camera for road view
    "resolution": {
      "width": 640,           // Reduced for better performance
      "height": 480
    },
    "fps": 15
  }
}
```

## Hardware Setup

### Camera Connections
1. **Driver Camera (Index 0)**: Connect to first USB port or use built-in camera
2. **Road Camera (Index 1)**: Connect second USB camera to different USB port

### Verify Camera Setup
```bash
# Check available cameras
ls -l /dev/video*

# Test both cameras
python3 test_dual_cameras.py
```

## Usage

### Run Full System
```bash
python3 main.py
```

### Test Cameras Only
```bash
python3 test_dual_cameras.py
```

## Window Layout

The system opens two separate windows:

- **Left Window**: "Driver Camera - Drowsiness Monitor"
  - Shows driver with face tracking
  - Displays drowsiness metrics overlay
  
- **Right Window**: "Road Camera - Object Detection"  
  - Shows road view with yellow object detection
  - Marks moving objects with yellow bounding boxes
  - Shows object count

## Object Detection Details

The road camera uses:
- **Background Subtraction**: MOG2 algorithm for motion detection
- **Morphological Operations**: Noise reduction and shape cleanup
- **Contour Detection**: Identifies object boundaries
- **Size Filtering**: Only marks objects between 0.1% and 30% of frame area
- **Yellow Markers**: Bounding boxes, labels, and center points

## Performance Tips

For Raspberry Pi Zero 2 W:
1. Use lower resolution (320x240 or 640x480)
2. Reduce FPS to 10-12
3. Consider using only one camera if performance is poor
4. Close other applications

## Troubleshooting

### Camera Issues
```bash
# Check camera permissions
sudo usermod -a -G video $USER

# Test individual cameras
ffplay /dev/video0  # Driver camera
ffplay /dev/video1  # Road camera
```

### Performance Issues
- Reduce resolution in `config.json`
- Lower FPS setting
- Use only driver camera (set `road_cam_index` same as `driver_cam_index`)

### Object Detection Not Working
- Ensure good lighting for road camera
- Check camera is pointing at area with movement
- Adjust `motion_sensitivity` in config
- Objects need to be moving to be detected

## Controls

- **'q'**: Quit application
- **'m'**: Mute alerts for 30 seconds
- **Mouse**: Click and drag windows to reposition

## Output Files

- `drowsiness_log.csv`: All detection metrics and alerts
- `scorer_debug.log`: Detailed scoring information

---

**Ready to use dual cameras!** 🎥📹