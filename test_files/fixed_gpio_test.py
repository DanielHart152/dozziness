#!/usr/bin/env python3
"""
Fixed GPIO Buzzer Test - Resolves PWM object conflict
"""
import time
import sys

try:
    import RPi.GPIO as GPIO
    print("✅ RPi.GPIO imported successfully")
except ImportError:
    print("❌ RPi.GPIO not available")
    sys.exit(1)

def test_pwm_buzzer():
    """Test PWM buzzer with proper cleanup"""
    print("\n🔊 Testing PWM buzzer...")
    try:
        GPIO.cleanup()  # Clean up any existing PWM objects
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)
        
        frequencies = [1000, 2000, 3000]
        
        for freq in frequencies:
            print(f"Testing {freq}Hz for 1 second...")
            buzzer = GPIO.PWM(18, freq)
            buzzer.start(50)
            time.sleep(1)
            buzzer.stop()
            del buzzer  # Explicitly delete PWM object
            time.sleep(0.5)
        
        print("✅ PWM buzzer test completed")
        return True
    except Exception as e:
        print(f"❌ PWM buzzer test failed: {e}")
        return False

def main():
    print("🚨 Fixed GPIO 18 Buzzer Test")
    print("=" * 30)
    
    try:
        if test_pwm_buzzer():
            print("\n✅ PWM test passed!")
        else:
            print("\n❌ PWM test failed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    finally:
        GPIO.cleanup()
        print("🧹 GPIO cleanup completed")

if __name__ == "__main__":
    main()