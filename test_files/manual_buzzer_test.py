#!/usr/bin/env python3
"""
Manual Buzzer Test - Quick test for GPIO 18 buzzer
Run this on your Raspberry Pi to test the buzzer
"""
import time
import sys

def test_buzzer():
    try:
        import RPi.GPIO as GPIO
        print("🔧 Testing GPIO 18 buzzer...")
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)
        
        print("Method 1: Simple ON/OFF (for active buzzers)")
        GPIO.output(18, GPIO.HIGH)
        print("Buzzer ON for 2 seconds...")
        time.sleep(2)
        GPIO.output(18, GPIO.LOW)
        print("Buzzer OFF")
        
        time.sleep(1)
        
        print("Method 2: PWM (for passive buzzers)")
        buzzer = GPIO.PWM(18, 2000)  # 2000Hz
        buzzer.start(50)  # 50% duty cycle
        print("PWM buzzer ON for 2 seconds...")
        time.sleep(2)
        buzzer.stop()
        print("PWM buzzer OFF")
        
        GPIO.cleanup()
        print("✅ Test completed - did you hear the buzzer?")
        
    except ImportError:
        print("❌ RPi.GPIO not available")
        print("Install with: pip3 install RPi.GPIO")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running with: sudo python3 manual_buzzer_test.py")
        try:
            GPIO.cleanup()
        except:
            pass

if __name__ == "__main__":
    test_buzzer()