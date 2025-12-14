#!/usr/bin/env python3
"""
GPIO Buzzer Test Script
Tests GPIO 18 buzzer functionality with different methods
"""
import time
import sys

try:
    import RPi.GPIO as GPIO
    print("✅ RPi.GPIO imported successfully")
except ImportError:
    print("❌ RPi.GPIO not available - install with: pip3 install RPi.GPIO")
    sys.exit(1)

def test_basic_gpio():
    """Test basic GPIO on/off"""
    print("\n🔧 Testing basic GPIO on/off...")
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)
        
        print("Setting GPIO 18 HIGH for 2 seconds...")
        GPIO.output(18, GPIO.HIGH)
        time.sleep(2)
        
        print("Setting GPIO 18 LOW")
        GPIO.output(18, GPIO.LOW)
        
        print("✅ Basic GPIO test completed")
        return True
    except Exception as e:
        print(f"❌ Basic GPIO test failed: {e}")
        return False

def test_pwm_buzzer():
    """Test PWM buzzer functionality"""
    print("\n🔊 Testing PWM buzzer...")
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)
        
        # Test different frequencies
        frequencies = [1000, 2000, 3000]
        
        for freq in frequencies:
            print(f"Testing {freq}Hz for 1 second...")
            buzzer = GPIO.PWM(18, freq)
            buzzer.start(50)  # 50% duty cycle
            time.sleep(1)
            buzzer.stop()
            time.sleep(0.5)
        
        print("✅ PWM buzzer test completed")
        return True
    except Exception as e:
        print(f"❌ PWM buzzer test failed: {e}")
        return False

def test_permissions():
    """Test GPIO permissions"""
    print("\n🔐 Testing GPIO permissions...")
    try:
        # Try to access GPIO memory
        with open('/dev/gpiomem', 'r') as f:
            print("✅ Can access /dev/gpiomem")
    except PermissionError:
        print("❌ Permission denied for /dev/gpiomem")
        print("💡 Try: sudo usermod -a -G gpio $USER")
        print("💡 Then logout and login again")
        return False
    except FileNotFoundError:
        print("❌ /dev/gpiomem not found")
        return False
    except Exception as e:
        print(f"❌ GPIO permission test failed: {e}")
        return False
    
    return True

def main():
    print("🚨 GPIO 18 Buzzer Diagnostic Test")
    print("=" * 40)
    
    # Check if running on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'Raspberry Pi' not in f.read():
                print("⚠️  Warning: Not running on Raspberry Pi")
    except:
        print("⚠️  Warning: Cannot detect if running on Raspberry Pi")
    
    try:
        # Test permissions first
        if not test_permissions():
            print("\n❌ Permission issues detected")
            return
        
        # Test basic GPIO
        if not test_basic_gpio():
            print("\n❌ Basic GPIO failed")
            return
        
        # Test PWM buzzer
        if not test_pwm_buzzer():
            print("\n❌ PWM buzzer failed")
            return
        
        print("\n✅ All GPIO tests passed!")
        print("\n🔧 Hardware Check:")
        print("1. Verify buzzer is connected to GPIO 18 (physical pin 12)")
        print("2. Check buzzer polarity (+ to GPIO 18, - to GND)")
        print("3. If using active buzzer, try passive buzzer or vice versa")
        print("4. Test with multimeter: should see voltage changes")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        try:
            GPIO.cleanup()
            print("🧹 GPIO cleanup completed")
        except:
            pass

if __name__ == "__main__":
    main()