# Debug Logging Guide - Enhanced Scoring System

## Overview
The enhanced scoring system now includes comprehensive debug logging to help troubleshoot and monitor the drowsiness detection algorithm in real-time.

## Log Files Generated

### 1. `scorer_debug.log`
- **Purpose:** Detailed scoring algorithm debug information
- **Content:** PERCLOS calculations, blink scoring, penalty tracking, alert decisions
- **Format:** Timestamped entries with component breakdowns

### 2. `drowsiness_log.csv` (existing)
- **Purpose:** Structured data logging for analysis
- **Content:** Metrics, scores, and detection flags
- **Format:** CSV with timestamps

## Debug Log Content

### Score Calculation Logs
```
=== SCORE CALCULATION START ===
Inputs: PERCLOS=0.2000, BPM=9.0, yawn=True, nod=False, texting=False, car_close=False
PERCLOS: input=0.2000, clamped=0.2000, raw=40.00, final=24.00 (weight=0.6)
BLINKS (triangular): bpm=9.0, target=16, deviation=7.0, tol=16, raw=28.12
BLINKS final: 11.25 (weight=0.4)
PENALTY awarded: yawn=10pts (last=1000.0s ago)
PENALTY window: 1 active, total=10.0, capped=10.0, new=[('yawn', 10)]
Score components: PERCLOS=24.00 + BLINKS=11.25 + PENALTIES=10.00 = 45.25
Final score: 45.25 (clamped from 45.25)
=== SCORE CALCULATION END ===
```

### Alert Decision Logs
```
--- ALERT DECISION START ---
Calculating rolling averages...
Rolling avg (1.0s): 15 scores, avg=72.3, scores=[('0.1s', '72.1'), ('0.2s', '72.5'), ('0.3s', '72.0')]
Rolling avg (2.0s): 30 scores, avg=68.4, scores=[('0.1s', '72.1'), ('0.2s', '68.9'), ('0.3s', '67.2')]
Alert state: currently=False, avg_on=72.30 (need>=70.0), avg_off=68.40 (need<=55.0), cooldown=0.0s
🚨 ALERT TRIGGERED: avg_on=72.30 >= 70.0, cooldown elapsed
ALERT STATE CHANGED: False -> True
--- ALERT DECISION END: True ---
```

### Penalty Tracking Logs
```
⚠️ PENALTY awarded: yawn=10pts (last=12.3s ago)
PENALTY blocked: nod in refractory (3.2s < 5.0s)
PENALTY cleanup: removed 2 old penalties
PENALTY window: 3 active, total=25.0, capped=25.0, new=[('texting', 25)]
```

## Viewing Debug Logs

### 1. Basic Log Viewing
```bash
# View last 50 lines
python3 view_debug_logs.py

# View last 100 lines
python3 view_debug_logs.py --lines 100
```

### 2. Real-time Monitoring
```bash
# Follow logs in real-time (like tail -f)
python3 view_debug_logs.py --follow
```

### 3. Filtered Viewing
```bash
# Show only score calculations
python3 view_debug_logs.py --filter score

# Show only alert decisions
python3 view_debug_logs.py --filter alert

# Show only penalty events
python3 view_debug_logs.py --filter penalty

# Show only PERCLOS calculations
python3 view_debug_logs.py --filter perclos

# Show only blink scoring
python3 view_debug_logs.py --filter blinks
```

### 4. Combined Options
```bash
# Follow alert logs in real-time
python3 view_debug_logs.py --follow --filter alert

# View last 200 penalty events
python3 view_debug_logs.py --lines 200 --filter penalty
```

## Debug Log Levels

### DEBUG Level (Most Verbose)
- Individual component calculations
- Rolling average details
- Penalty refractory checks
- Score history management

### INFO Level (Important Events)
- Alert state changes
- Alert triggers and clears
- Major scoring events

## Troubleshooting with Debug Logs

### 1. Score Too Low/High
**Check:** PERCLOS and blink component contributions
```bash
python3 view_debug_logs.py --filter score | grep "Score components"
```

### 2. Alerts Not Triggering
**Check:** Rolling averages and thresholds
```bash
python3 view_debug_logs.py --filter alert | grep "Alert state"
```

### 3. Penalties Not Working
**Check:** Refractory periods and windowing
```bash
python3 view_debug_logs.py --filter penalty
```

### 4. Alert Chatter
**Check:** Hysteresis behavior and hold times
```bash
python3 view_debug_logs.py --filter alert | grep "ALERT STATE CHANGED"
```

## Performance Impact

- **File I/O:** Minimal impact, logs written asynchronously
- **CPU:** ~1-2% overhead for debug logging
- **Storage:** ~1MB per hour of operation
- **Memory:** Negligible (uses efficient logging handlers)

## Production Deployment

### Disable Debug Logging
```json
{
  "logging": {
    "debug_enabled": false,
    "debug_level": "INFO"
  }
}
```

### Log Rotation
```bash
# Setup logrotate for debug logs
sudo nano /etc/logrotate.d/drowsiness-debug

# Add configuration:
/path/to/scorer_debug.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

## Testing Debug Logs

### Run Test with Debug Output
```bash
# Terminal 1: Run test
python3 test_enhanced_scoring.py

# Terminal 2: Monitor logs
python3 view_debug_logs.py --follow --filter score
```

### Verify Log Content
```bash
# Check log file exists and has content
ls -la scorer_debug.log

# Count log entries
wc -l scorer_debug.log

# Search for specific events
grep "ALERT TRIGGERED" scorer_debug.log
```

## Integration with Main System

The debug logging is automatically enabled when you run:
```bash
python3 main.py
```

Monitor in real-time with:
```bash
python3 view_debug_logs.py --follow
```

This provides complete visibility into the scoring algorithm's decision-making process for development, testing, and troubleshooting.