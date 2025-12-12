# Complete Workflow Explanation - Driver Drowsiness Detection System

## Overview
This document explains the complete workflow when you run `python3 main.py`. Follow along to understand how every component works together.

---

## PHASE 1: INITIALIZATION (When script starts)

### Step 1: Script Entry Point
```python
if __name__ == "__main__":
    main()
```
- Python calls the `main()` function
- Parses command line arguments (config file path)

### Step 2: Create DrowsinessMonitor Object
```python
monitor = DrowsinessMonitor("config.json")
```

**What happens inside `__init__`:**

#### 2.1 Load Configuration
- Reads `config.json` file
- Validates all required keys exist
- Stores all settings (cameras, thresholds, GPIO pins, etc.)

#### 2.2 Initialize Components (in order)

**A. FaceDetector** (`vision.py`)
- Tries to load dlib (if available)
- If dlib fails → loads OpenCV Haar cascades
- Sets up eye/mouth/head detection thresholds
- Creates empty history buffers for:
  - Eye state history (for PERCLOS calculation)
  - Blink history (for blinks per minute)

**B. RoadDetector** (`vision.py`)
- Sets up background subtractor (for car detection)
- Configures texting detection parameters

**C. RiskScorer** (`score.py`)
- Loads scoring weights and thresholds from config
- Sets up penalty values (yawn, nod, texting, car close)

**D. AlertManager** (`io_alerts.py`)
- Initializes GPIO pins (buzzer on GPIO 18, mute button on GPIO 16)
- Starts background thread to monitor mute button
- Sets up cooldown and mute duration

**E. DataLogger** (`logger.py`)
- Creates CSV file if it doesn't exist
- Writes header row with column names

#### 2.3 Store Camera Settings
- Driver camera index: 0
- Road camera index: 1 (or 2 in your case)
- Resolution: 320x240
- Target FPS: 15

---

## PHASE 2: CAMERA INITIALIZATION

### Step 3: Initialize Cameras
```python
monitor.initialize_cameras()
```

**Process:**

1. **Open Driver Camera**
   - `cv2.VideoCapture(0)` - Opens /dev/video0
   - Sets resolution to 320x240
   - Sets FPS to 15
   - Tests if it can read a frame
   - ✅ Success: "Driver camera initialized: 800x448 @ 15 FPS"

2. **Open Road Camera** (if index different from driver)
   - `cv2.VideoCapture(2)` - Opens /dev/video2
   - Same setup as driver camera
   - ✅ Success: "Road camera initialized: 800x448"
   - ⚠️ If fails: Continues without it

---

## PHASE 3: MAIN PROCESSING LOOP

### Step 4: Start Main Loop
```python
monitor.run()
```

**The loop runs continuously until you press 'q' or Ctrl+C:**

```
┌─────────────────────────────────────────────────┐
│           MAIN LOOP (Runs forever)              │
│                                                 │
│  For each frame (15 times per second):         │
│                                                 │
│  1. Capture Frame                               │
│  2. Process Frame                               │
│  3. Calculate Metrics                           │
│  4. Draw HUD                                    │
│  5. Log Data                                    │
│  6. Display Window                              │
│  7. Check Keyboard Input                        │
│  8. Control Frame Rate                          │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## DETAILED FRAME PROCESSING (What happens each frame)

### Frame 1: Capture Images
```python
ret_driver, driver_frame = self.driver_cam.read()
ret_road, road_frame = self.road_cam.read()
```
- Reads one frame from each camera
- Resizes to 320x240 if needed
- If road camera fails → `road_frame = None`

### Frame 2: Process Driver Frame
```python
driver_result = self.face_detector.process_frame(driver_frame)
```

**Inside `process_frame()`:**

#### A. Convert to Grayscale
- Converts BGR image to grayscale (faster processing)

#### B. Detect Face
- **If dlib available:**
  - Uses dlib face detector
  - Finds 68 facial landmarks (eyes, nose, mouth, etc.)
  
- **If OpenCV only:**
  - Uses Haar cascade to find face rectangle
  - Uses eye cascade to count visible eyes
  - Creates simplified landmarks

#### C. Calculate Eye Aspect Ratio (EAR)
- Extracts 6 points per eye from landmarks
- Calculates: `EAR = (vertical1 + vertical2) / (2 × horizontal)`
- **EAR < 0.25** = Eyes closed
- **EAR ≥ 0.25** = Eyes open

#### D. Track Eye State
- Adds eye state (open/closed) to history buffer
- History buffer size: 45 seconds × 15 FPS = 675 frames

#### E. Detect Blinks
- If eyes closed for 3+ consecutive frames → Count as blink
- Adds blink to blink history (60 second window)

#### F. Calculate PERCLOS
```python
perclos = closed_frames / total_frames_in_window
```
- Counts how many frames eyes were closed
- Divides by total frames in 45-second window
- Result: 0.0 (all open) to 1.0 (all closed)

#### G. Calculate Blinks Per Minute
```python
blinks_per_min = (blink_count / window_seconds) × 60
```
- Counts blinks in 60-second window
- Converts to per-minute rate

#### H. Detect Yawn (if dlib available)
- Calculates Mouth Aspect Ratio (MAR)
- If MAR > 0.6 for 10+ frames → Yawn detected

#### I. Detect Head Nod (if dlib available)
- Calculates head pitch angle from landmarks
- If pitch > 15° down for 5+ frames → Nod detected

**Returns:** Dictionary with all detection results

### Frame 3: Process Road Frame (if available)
```python
car_close = self.road_detector.detect_car_close(road_frame)
```

**Car Detection:**
- Uses background subtraction to find moving objects
- Looks for large objects in lower center region
- If found → `car_close = True`

### Frame 4: Detect Texting
```python
texting_detected = self.road_detector.detect_texting(driver_frame)
```

**Texting Detection:**
- Analyzes upper portion of driver frame
- Detects skin-colored pixels (hands/phone)
- If skin ratio > threshold → Texting detected

### Frame 5: Calculate Risk Score
```python
risk_score = self.scorer.calculate_score(...)
```

**Score Calculation:**

1. **PERCLOS Score** (60% weight)
   ```
   perclos_score = (PERCLOS / 0.15) × 50
   final_perclos = perclos_score × 0.6
   ```
   - Example: PERCLOS = 0.20 → 40 points × 0.6 = **24 points**

2. **Blinks Score** (40% weight)
   ```
   blinks_score = (blinks_per_min / 20) × 50
   final_blinks = blinks_score × 0.4
   ```
   - Example: 10 blinks/min → 25 points × 0.4 = **10 points**

3. **Add Penalties**
   - Yawn detected: +10 points
   - Nod detected: +5 points
   - Texting detected: +25 points
   - Car close: +30 points

4. **Final Score**
   ```
   risk_score = perclos_points + blinks_points + penalties
   ```
   - Clamped to 0-100 range

### Frame 6: Check Alert Condition
```python
should_alert = self.scorer.should_alert(risk_score)
```

**Alert Logic:**
- If `risk_score >= 50` (threshold) → `should_alert = True`
- Otherwise → `should_alert = False`

### Frame 7: Trigger Alert (if needed)
```python
if should_alert:
    alert_triggered = self.alert_manager.trigger_alert()
```

**Alert Process:**

1. **Check Mute Status**
   - If muted → Skip alert
   - If mute expired → Re-enable alerts

2. **Check Cooldown**
   - If last alert < 8 seconds ago → Skip (prevent spam)

3. **Trigger Buzzer**
   - Sets GPIO 18 to PWM mode
   - Generates 2000Hz tone for 500ms
   - Stops buzzer
   - Updates last alert time

### Frame 8: Compile Metrics Dictionary
```python
metrics = {
    'perclos': 0.000,
    'blinks_per_min': 0.0,
    'risk_score': 12.0,
    'alert_triggered': False,
    'yawn_detected': False,
    'nod_detected': False,
    'texting_detected': False,
    'car_close': False,
    'left_ear': 0.0,
    'right_ear': 0.0,
    'mar': 0.0,
    'head_pitch': 0.0,
    'face_detected': False
}
```

### Frame 9: Draw HUD Overlay
```python
driver_frame = self.draw_hud(driver_frame, metrics)
```

**HUD Elements:**
- Semi-transparent black background (top-left)
- Title: "DROWSINESS MONITOR"
- Risk Score (color-coded: green/yellow/red)
- PERCLOS value
- Blinks per minute
- Face detection status
- Warning messages (Yawn, Nod, Texting, Car Close)
- Alert indicator
- Mute countdown (if muted)

### Frame 10: Log to CSV
```python
self.logger.log(metrics)
```

**Logging:**
- Checks if 1 second has passed since last log
- Writes one row to `drowsiness_log.csv`:
  - Timestamp
  - All metrics values
  - All detection flags

### Frame 11: Display Window
```python
cv2.imshow('Driver Drowsiness Monitor', driver_frame)
```

- Shows camera feed with HUD overlay
- Updates display (if display available)

### Frame 12: Handle Keyboard Input
```python
key = cv2.waitKey(1) & 0xFF
```

- **'q' key** → Sets `self.running = False` → Exits loop
- **'m' key** → Calls `alert_manager.mute()` → Mutes for 30 seconds

### Frame 13: Frame Rate Control
```python
elapsed = time.time() - frame_start
sleep_time = max(0, self.frame_time - elapsed)
if sleep_time > 0:
    time.sleep(sleep_time)
```

- Calculates how long frame processing took
- If faster than target FPS → Sleep to maintain 15 FPS
- Ensures consistent frame rate

### Frame 14: Status Print (Every 5 seconds)
```python
if self.frame_count % (15 * 5) == 0:  # Every 75 frames
    print(f"Frame: {self.frame_count} | Score: {score} | ...")
```

- Prints status to console every 5 seconds
- Shows current metrics

---

## PHASE 4: CLEANUP (When loop exits)

### Step 5: Cleanup
```python
monitor.cleanup()
```

**Cleanup Process:**

1. **Stop Running**
   - Sets `self.running = False`

2. **Release Cameras**
   - `driver_cam.release()`
   - `road_cam.release()`

3. **Close Windows**
   - `cv2.destroyAllWindows()`

4. **Cleanup GPIO**
   - `GPIO.cleanup()` - Releases GPIO pins

5. **Print Message**
   - "Cleanup complete."

---

## DATA FLOW DIAGRAM

```
┌─────────────┐
│   Camera    │
│  /dev/video0│
└──────┬──────┘
       │ Frame (BGR image)
       ▼
┌─────────────────┐
│  FaceDetector   │
│  - Detect face  │
│  - Find eyes    │
│  - Calculate EAR│
│  - Track blinks │
└──────┬──────────┘
       │ Detection Results
       ▼
┌─────────────────┐      ┌──────────────┐
│  FaceDetector   │      │ RoadDetector │
│  - PERCLOS      │      │ - Car close  │
│  - Blinks/min   │      │ - Texting    │
└──────┬──────────┘      └──────┬───────┘
       │                        │
       └────────┬───────────────┘
                │ All Indicators
                ▼
┌─────────────────┐
│   RiskScorer    │
│  - Calculate    │
│    score (0-100)│
└──────┬──────────┘
       │ Risk Score
       ▼
┌─────────────────┐
│  AlertManager   │
│  - Check score  │
│  - Trigger GPIO │
│  - Handle mute  │
└──────┬──────────┘
       │ Alert Status
       ▼
┌─────────────────┐
│  DataLogger     │
│  - Write CSV    │
└─────────────────┘
```

---

## KEY CONCEPTS

### PERCLOS (Percentage of Eyelid Closure)
- **Window**: Last 45 seconds
- **Calculation**: `closed_frames / total_frames`
- **Meaning**: 
  - 0.0 = Eyes always open (alert)
  - 0.15 = 15% closed (threshold)
  - 0.30 = 30% closed (drowsy)
  - 1.0 = Eyes always closed (very drowsy)

### Eye Aspect Ratio (EAR)
- **Formula**: `(vertical1 + vertical2) / (2 × horizontal)`
- **Threshold**: 0.25
- **< 0.25**: Eyes closed
- **≥ 0.25**: Eyes open

### Risk Score Calculation
```
Base Score = (PERCLOS_score × 0.6) + (Blinks_score × 0.4)
Final Score = Base Score + Penalties
```

**Penalties:**
- Yawn: +10
- Nod: +5
- Texting: +25
- Car Close: +30

### Alert Trigger
- **Condition**: `risk_score >= 50`
- **Cooldown**: 8 seconds between alerts
- **Mute**: Can be muted for 30 seconds

---

## EXAMPLE WORKFLOW (One Frame)

**Input:**
- Driver frame: Face visible, eyes open
- Road frame: No car close

**Processing:**
1. Face detected ✅
2. EAR = 0.30 (eyes open)
3. PERCLOS = 0.05 (5% closed in last 45s)
4. Blinks/min = 18
5. No yawn, nod, texting, or car close

**Scoring:**
- PERCLOS score: (0.05/0.15) × 50 × 0.6 = **10 points**
- Blinks score: (18/20) × 50 × 0.4 = **18 points**
- Penalties: 0
- **Total: 28 points**

**Result:**
- Risk Score: 28
- Alert: No (28 < 50)
- Logged to CSV
- HUD shows: "Risk Score: 28.0"

---

## TIMING

- **Frame Rate**: 15 FPS (1 frame every 0.067 seconds)
- **PERCLOS Window**: 45 seconds (675 frames)
- **Blink Window**: 60 seconds (900 frames)
- **Log Interval**: 1 second (15 frames)
- **Status Print**: Every 5 seconds (75 frames)
- **Alert Cooldown**: 8 seconds (120 frames)
- **Mute Duration**: 30 seconds (450 frames)

---

## SUMMARY

**Every Second (15 frames):**
1. Capture 15 frames from cameras
2. Detect face in each frame
3. Track eye state (open/closed)
4. Calculate PERCLOS from 45s history
5. Calculate blinks from 60s history
6. Calculate risk score
7. Check if alert needed
8. Trigger buzzer if score ≥ 50
9. Log to CSV (once per second)
10. Display HUD overlay
11. Handle keyboard input

**The system continuously monitors and updates metrics in real-time!**

---

This is the complete workflow. Every frame goes through this process, creating a continuous monitoring system that detects drowsiness and triggers alerts when needed.

