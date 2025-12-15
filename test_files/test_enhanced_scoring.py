#!/usr/bin/env python3
"""
Test script for the enhanced scoring system.
Demonstrates the corrected PERCLOS scaling, triangular blink scoring, 
windowed penalties, and hysteresis alerting with debug logging.
"""

import time
import logging
from score import RiskScorer

# Enable debug logging to console for this test
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

debug_logger = logging.getLogger('scorer_debug')
debug_logger.addHandler(console_handler)
debug_logger.setLevel(logging.DEBUG)

def test_enhanced_scoring():
    """Test the enhanced scoring system with worked examples."""
    
    # Configuration for testing
    config = {
        'scoring': {
            'perclos_weight': 0.60,
            'blinks_per_min_weight': 0.40,
            'perclos_max': 0.25,  # Cap PERCLOS at 25%
            'blink_mode': 'triangular',
            'blink_target': 16.0,
            'blink_tolerance': 16.0,
            'yawn_penalty': 10,
            'nod_penalty': 5,
            'texting_penalty': 25,
            'car_close_penalty': 30,
            'penalty_window_sec': 10.0,
            'penalty_window_cap': 40.0,
        },
        'alerts': {
            'alert_on_threshold': 70.0,
            'alert_off_threshold': 55.0,
            'hold_on_sec': 1.0,
            'hold_off_sec': 2.0,
            'cooldown_seconds': 8.0,
        }
    }
    
    scorer = RiskScorer(config)
    
    print("=" * 60)
    print("ENHANCED SCORING SYSTEM TEST")
    print("=" * 60)
    
    # Test Case 1: Moderate drowsy
    print("\n--- Test Case 1: Moderate Drowsy ---")
    print("PERCLOS: 0.20, Blinks: 9 bpm, Yawn: True, Nod: False")
    
    score1 = scorer.calculate_score(
        perclos=0.20,
        blinks_per_min=9.0,
        yawn_detected=True,
        nod_detected=False
    )
    
    # Manual calculation for verification
    perclos_points = (0.20 / 0.25) * 50 * 0.60  # = 24.0
    blink_points = 50 * (1 - abs(9 - 16) / 16) * 0.40  # = 8.75
    penalty_points = 10  # yawn
    expected = perclos_points + blink_points + penalty_points
    
    print(f"Calculated Score: {score1:.1f}")
    print(f"Expected Score: {expected:.1f}")
    print(f"PERCLOS contribution: {perclos_points:.1f}")
    print(f"Blinks contribution: {blink_points:.1f}")
    print(f"Penalty contribution: {penalty_points}")
    
    should_alert1 = scorer.should_alert(score=score1)
    print(f"Should Alert: {should_alert1}")
    
    # Test Case 2: High risk
    print("\n--- Test Case 2: High Risk ---")
    print("PERCLOS: 0.25, Blinks: 6 bpm, Yawn: True, Nod: True")
    
    # Wait a bit to avoid refractory period
    time.sleep(0.1)
    
    score2 = scorer.calculate_score(
        perclos=0.25,
        blinks_per_min=6.0,
        yawn_detected=True,
        nod_detected=True
    )
    
    # Manual calculation
    perclos_points2 = (0.25 / 0.25) * 50 * 0.60  # = 30.0
    blink_points2 = 50 * (1 - abs(6 - 16) / 16) * 0.40  # = 7.5
    penalty_points2 = 15  # yawn + nod (within window)
    expected2 = perclos_points2 + blink_points2 + penalty_points2
    
    print(f"Calculated Score: {score2:.1f}")
    print(f"Expected Score: {expected2:.1f}")
    print(f"PERCLOS contribution: {perclos_points2:.1f}")
    print(f"Blinks contribution: {blink_points2:.1f}")
    print(f"Penalty contribution: {penalty_points2}")
    
    should_alert2 = scorer.should_alert(score=score2)
    print(f"Should Alert: {should_alert2}")
    
    # Test Case 3: Normal driving
    print("\n--- Test Case 3: Normal Driving ---")
    print("PERCLOS: 0.05, Blinks: 16 bpm, No events")
    
    time.sleep(0.1)
    
    score3 = scorer.calculate_score(
        perclos=0.05,
        blinks_per_min=16.0,
        yawn_detected=False,
        nod_detected=False
    )
    
    # Manual calculation
    perclos_points3 = (0.05 / 0.25) * 50 * 0.60  # = 6.0
    blink_points3 = 50 * (1 - abs(16 - 16) / 16) * 0.40  # = 20.0 (optimal)
    penalty_points3 = 0  # no events
    expected3 = perclos_points3 + blink_points3 + penalty_points3
    
    print(f"Calculated Score: {score3:.1f}")
    print(f"Expected Score: {expected3:.1f}")
    print(f"PERCLOS contribution: {perclos_points3:.1f}")
    print(f"Blinks contribution: {blink_points3:.1f}")
    print(f"Penalty contribution: {penalty_points3}")
    
    should_alert3 = scorer.should_alert(score=score3)
    print(f"Should Alert: {should_alert3}")
    
    # Test Case 4: Test hysteresis
    print("\n--- Test Case 4: Hysteresis Test ---")
    print("Simulating sustained high score for alert ON")
    
    # Simulate sustained high scores to trigger alert
    for i in range(20):  # 20 frames at 15 FPS ≈ 1.3 seconds
        high_score = scorer.calculate_score(
            perclos=0.25,
            blinks_per_min=5.0,
            yawn_detected=True,
            nod_detected=False
        )
        time.sleep(0.05)  # 50ms between frames
        
        if i % 5 == 0:  # Print every 5th frame
            should_alert = scorer.should_alert(score=high_score)
            print(f"Frame {i+1}: Score={high_score:.1f}, Alert={should_alert}")
    
    print("\nSimulating drop to low score for alert OFF")
    
    # Simulate drop to low scores
    for i in range(30):  # 30 frames ≈ 2 seconds
        low_score = scorer.calculate_score(
            perclos=0.05,
            blinks_per_min=16.0,
            yawn_detected=False,
            nod_detected=False
        )
        time.sleep(0.05)
        
        if i % 10 == 0:  # Print every 10th frame
            should_alert = scorer.should_alert(score=low_score)
            print(f"Frame {i+21}: Score={low_score:.1f}, Alert={should_alert}")
    
    print("\n" + "=" * 60)
    print("SCORING FORMULAS VERIFICATION")
    print("=" * 60)
    
    print("\n1. PERCLOS Formula:")
    print("   perclos_clamped = clamp(PERCLOS, 0, 0.25)")
    print("   perclos_score = (perclos_clamped / 0.25) * 50")
    print("   final_perclos = perclos_score * 0.60")
    print("   Example: PERCLOS=0.20 → (0.20/0.25)*50*0.60 = 24.0 points")
    
    print("\n2. Blinks Formula (Triangular):")
    print("   raw = 50 * (1 - abs(bpm - 16) / 16)")
    print("   final_blinks = clamp(raw, 0, 50) * 0.40")
    print("   Example: 9 bpm → 50*(1-7/16)*0.40 = 8.75 points")
    print("   Example: 16 bpm → 50*(1-0/16)*0.40 = 20.0 points (optimal)")
    
    print("\n3. Penalties (Windowed with Refractory):")
    print("   Yawn: +10 points (refractory: 10s)")
    print("   Nod: +5 points (refractory: 5s)")
    print("   Texting: +25 points (refractory: 10s)")
    print("   Car Close: +30 points (refractory: 5s)")
    print("   Max penalty in 10s window: 40 points")
    
    print("\n4. Alert Logic (Hysteresis):")
    print("   Alert ON: avg(1s) >= 70.0 AND cooldown elapsed")
    print("   Alert OFF: avg(2s) <= 55.0")
    print("   Cooldown: 8 seconds between alert activations")
    
    print("\n" + "=" * 60)
    print("DEBUG LOGGING")
    print("=" * 60)
    print("\nDebug logs are written to 'scorer_debug.log'")
    print("Use the following commands to view logs:")
    print("\n  # View all recent logs:")
    print("  python3 view_debug_logs.py")
    print("\n  # Follow logs in real-time:")
    print("  python3 view_debug_logs.py --follow")
    print("\n  # Filter by type:")
    print("  python3 view_debug_logs.py --filter score")
    print("  python3 view_debug_logs.py --filter alert")
    print("  python3 view_debug_logs.py --filter penalty")
    print("\n  # View specific number of lines:")
    print("  python3 view_debug_logs.py --lines 100")

if __name__ == "__main__":
    print("Starting enhanced scoring test with debug logging...")
    print("Debug logs will be written to 'scorer_debug.log'")
    print("Run 'python3 view_debug_logs.py --follow' in another terminal to see real-time logs\n")
    
    test_enhanced_scoring()