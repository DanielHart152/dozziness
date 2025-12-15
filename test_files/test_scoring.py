#!/usr/bin/env python3
"""
Test script to verify the scoring formulas match the specifications.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from score import RiskScorer

def test_perclos_formula():
    """Test PERCLOS scaling formula."""
    print("=== PERCLOS Formula Test ===")
    
    # Mock config for testing
    config = {
        'scoring': {
            'perclos_weight': 0.60,
            'blinks_per_min_weight': 0.40,
            'perclos_max': 0.25,
            'blink_mode': 'triangular',
            'blink_target': 16.0,
            'blink_tolerance': 16.0,
        },
        'alerts': {},
        'logging': {'debug_interval_seconds': 999}  # Disable debug logging
    }
    
    scorer = RiskScorer(config)
    
    # Test case from specification: PERCLOS = 0.20 → should give 24 points
    perclos = 0.20
    expected = 24.0  # (0.20/0.25)*50*0.60 = 24
    actual = scorer._perclos_points(perclos)
    
    print(f"PERCLOS = {perclos}")
    print(f"Formula: ({perclos}/0.25) * 50 * 0.60")
    print(f"Expected: {expected}")
    print(f"Actual: {actual:.2f}")
    print(f"Match: {'PASS' if abs(actual - expected) < 0.01 else 'FAIL'}")
    print()

def test_blink_formula():
    """Test blink rate triangular formula."""
    print("=== Blink Rate Formula Test ===")
    
    config = {
        'scoring': {
            'perclos_weight': 0.60,
            'blinks_per_min_weight': 0.40,
            'perclos_max': 0.25,
            'blink_mode': 'triangular',
            'blink_target': 16.0,
            'blink_tolerance': 16.0,
        },
        'alerts': {},
        'logging': {'debug_interval_seconds': 999}
    }
    
    scorer = RiskScorer(config)
    
    # Test cases
    test_cases = [
        (16.0, 20.0),  # Perfect target → max 20 points
        (9.0, 11.25),  # |9-16|/16 = 0.4375 → (1-0.4375)*50*0.40 = 11.25
        (6.0, 7.5),    # |6-16|/16 = 0.625 → (1-0.625)*50*0.40 = 7.5
        (0.0, 0.0),    # |0-16|/16 = 1.0 → (1-1.0)*50*0.40 = 0
        (32.0, 0.0),   # |32-16|/16 = 1.0 → (1-1.0)*50*0.40 = 0
    ]
    
    for bpm, expected in test_cases:
        actual = scorer._blink_points(bpm)
        deviation = abs(bpm - 16.0)
        raw = 50.0 * (1.0 - deviation / 16.0)
        raw = max(0.0, min(raw, 50.0))
        
        print(f"BPM = {bpm}")
        print(f"Formula: 50 * (1 - |{bpm}-16|/16) * 0.40")
        print(f"Raw score: {raw:.2f}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual:.2f}")
        print(f"Match: {'PASS' if abs(actual - expected) < 0.01 else 'FAIL'}")
        print()

def test_complete_examples():
    """Test complete scoring examples from specification."""
    print("=== Complete Scoring Examples ===")
    
    config = {
        'scoring': {
            'perclos_weight': 0.60,
            'blinks_per_min_weight': 0.40,
            'perclos_max': 0.25,
            'blink_mode': 'triangular',
            'blink_target': 16.0,
            'blink_tolerance': 16.0,
            'yawn_penalty': 10,
            'nod_penalty': 5,
            'texting_penalty': 25,
            'car_close_penalty': 30,
            'yawn_refractory_sec': 10.0,
            'nod_refractory_sec': 5.0,
            'texting_refractory_sec': 10.0,
            'car_close_refractory_sec': 5.0,
            'penalty_window_sec': 10.0,
            'penalty_window_cap': 40.0,
        },
        'alerts': {},
        'logging': {'debug_interval_seconds': 999}
    }
    
    scorer = RiskScorer(config)
    
    # Case 1: PERCLOS 0.20, blinks 9 bpm, yawn true, nod false
    print("Case 1 (moderate drowsy):")
    print("PERCLOS 0.20, blinks 9 bpm, yawn true, nod false")
    
    score1 = scorer.calculate_score(
        perclos=0.20, blinks_per_min=9.0,
        yawn_detected=True, nod_detected=False,
        texting_detected=False, car_close=False
    )
    
    # Expected breakdown:
    # PERCLOS: (0.20/0.25)*50*0.60 = 24
    # Blinks: 50*(1-|9-16|/16)*0.40 = 50*0.5625*0.40 = 11.25
    # Penalties: +10 (yawn)
    # Total ≈ 45.25
    expected1 = 45.25
    
    print(f"Expected: ~{expected1}")
    print(f"Actual: {score1:.2f}")
    print(f"Match: {'PASS' if abs(score1 - expected1) < 2.0 else 'FAIL'}")
    print()
    
    # Case 2: PERCLOS 0.25, blinks 6 bpm, yawn true, nod true
    print("Case 2 (high risk):")
    print("PERCLOS 0.25, blinks 6 bpm, yawn true, nod true")
    
    score2 = scorer.calculate_score(
        perclos=0.25, blinks_per_min=6.0,
        yawn_detected=True, nod_detected=True,
        texting_detected=False, car_close=False
    )
    
    # Expected breakdown:
    # PERCLOS: (0.25/0.25)*50*0.60 = 30
    # Blinks: 50*(1-|6-16|/16)*0.40 = 50*0.375*0.40 = 7.5
    # Penalties: +10 +5 = +15
    # Total ≈ 52.5
    expected2 = 52.5
    
    print(f"Expected: ~{expected2}")
    print(f"Actual: {score2:.2f}")
    print(f"Match: {'PASS' if abs(score2 - expected2) < 2.0 else 'FAIL'}")
    print()

def test_alert_logic():
    """Test alert hysteresis logic."""
    print("=== Alert Logic Test ===")
    
    config = {
        'scoring': {
            'perclos_weight': 0.60,
            'blinks_per_min_weight': 0.40,
            'perclos_max': 0.25,
            'blink_mode': 'triangular',
            'blink_target': 16.0,
            'blink_tolerance': 16.0,
        },
        'alerts': {
            'hold_on_sec': 1.0,
            'hold_off_sec': 2.0,
            'cooldown_seconds': 8.0,
        },
        'logging': {'debug_interval_seconds': 999}
    }
    
    scorer = RiskScorer(config)
    
    # Test thresholds
    print(f"Alert ON threshold: {scorer.alert_on} (should be 70.0)")
    print(f"Alert OFF threshold: {scorer.alert_off} (should be 55.0)")
    print(f"Hold ON time: {scorer.hold_on_sec}s")
    print(f"Hold OFF time: {scorer.hold_off_sec}s")
    print(f"Cooldown: {scorer.alert_cooldown_sec}s")
    
    # Test alert triggering
    should_alert_low = scorer.should_alert(score=60.0)  # Below threshold
    should_alert_high = scorer.should_alert(score=75.0)  # Above threshold
    
    print(f"Score 60.0 -> Alert: {should_alert_low} (should be False)")
    print(f"Score 75.0 -> Alert: {should_alert_high} (should be True)")
    print()

if __name__ == "__main__":
    print("Testing Driver Drowsiness Scoring System")
    print("=" * 50)
    print()
    
    test_perclos_formula()
    test_blink_formula()
    test_complete_examples()
    test_alert_logic()
    
    print("=" * 50)
    print("Testing complete!")