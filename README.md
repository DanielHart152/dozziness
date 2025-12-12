# Driver Drowsiness Detection - Raspberry Pi

Complete driver drowsiness detection system optimized for Raspberry Pi Zero 2 W (compatible with Pi 4).

## Quick Start

### 1. Copy Files to Raspberry Pi

Copy the entire `raspberry` folder to your Raspberry Pi.

### 2. Install Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-dev cmake build-essential
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev liblapack-dev libblas-dev

pip3 install numpy opencv-python dlib RPi.GPIO
```

### 3. Download dlib Shape Predictor (Optional but Recommended)

```bash
cd raspberry
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### 4. Configure Hardware

**Cameras:**
- Connect USB webcams
- Find camera indices: `ls -l /dev/video*`
- Update `config.json` with correct indices

**GPIO:**
- Buzzer: Connect to GPIO 18 (configurable)
- Mute button: Connect to GPIO 16 (optional, configurable)

### 5. Run

```bash
cd raspberry
python3 main.py
```

## File Structure

```
raspberry/
├── main.py              # Main entry point
├── vision.py            # Face/eye detection
├── score.py             # Risk score calculation
├── io_alerts.py         # GPIO buzzer control
├── logger.py            # CSV logging
├── utils_config.py      # Config loader
├── config.json          # Configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

Edit `config.json` to customize:

- **Camera indices**: `cameras.driver_cam_index`, `cameras.road_cam_index`
- **Resolution**: `cameras.resolution.width/height`
- **FPS**: `cameras.fps`
- **Alert threshold**: `alerts.risk_threshold` (default: 50)
- **GPIO pins**: `alerts.buzzer_gpio_pin`, `alerts.mute_button_gpio_pin`

## Hardware Setup

### Camera Connection

1. Connect USB webcams to Pi
2. Check devices: `ls -l /dev/video*`
3. Update `config.json`:
   ```json
   "driver_cam_index": 0,
   "road_cam_index": 1
   ```

### GPIO Wiring

**Buzzer (GPIO 18):**
- Positive lead → GPIO 18
- Negative lead → GND
- **Note**: Use transistor/relay if buzzer needs more current

**Mute Button (GPIO 16, optional):**
- One terminal → GPIO 16
- Other terminal → GND
- Button pulls GPIO low when pressed

## Usage

### Basic Run

```bash
python3 main.py
```

### With Custom Config

```bash
python3 main.py --config /path/to/custom_config.json
```

### Headless Operation (No Display)

```bash
# Install X virtual framebuffer
sudo apt install xvfb

# Run with virtual display
xvfb-run -a python3 main.py
```

Or set display:
```bash
export DISPLAY=:0
python3 main.py
```

## Controls

- **'q'**: Quit application
- **'m'**: Mute alerts for 30 seconds
- **Mute Button**: Physical button to mute (if connected)

## Performance Tips for Pi Zero 2 W

1. **Reduce resolution** in `config.json` (e.g., 240×180)
2. **Lower FPS** (e.g., 10-12 FPS)
3. **Close other applications**
4. **Use Bullseye Lite** (no desktop environment)
5. **Install dlib** for better detection (but heavier)

## Troubleshooting

### Camera Not Detected

```bash
# Check cameras
ls -l /dev/video*

# Test camera
ffplay /dev/video0
```

### GPIO Not Working

```bash
# Test GPIO
python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); GPIO.setup(18, GPIO.OUT); GPIO.output(18, GPIO.HIGH)"
```

### Low Performance

- Reduce resolution to 240×180
- Lower FPS to 10
- Close other apps
- Use OpenCV only (don't install dlib)

### Face Not Detected

- Ensure good lighting
- Position camera to see full face
- Adjust thresholds in `config.json`

## Output

### CSV Logging

All metrics logged to `drowsiness_log.csv`:
- Timestamp
- PERCLOS
- Blinks per minute
- Risk score
- Alert status
- All detection flags

### Console Output

- Status updates every 5 seconds
- Alert notifications
- Error messages

## Alert Conditions

Alert triggers when **Risk Score ≥ 50** (configurable).

Risk Score = (PERCLOS × 60%) + (Blinks/min × 40%) + Penalties

See `config.json` for all thresholds and weights.

## Support

For issues:
1. Check camera connections
2. Verify GPIO wiring
3. Review `config.json` settings
4. Check CSV logs for patterns

---

**Ready to deploy!** Copy files to Pi and run `python3 main.py` 🚀

