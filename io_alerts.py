"""
I/O and alert management module for Raspberry Pi.
Handles GPIO buzzer control, cooldown, and mute functionality.
"""
import time
import threading
from typing import Dict
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available. GPIO functions will be simulated.")


class AlertManager:
    """Manage alerts, buzzer, and mute functionality."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gpio_available = GPIO_AVAILABLE
        alerts = config['alerts']
        self.buzzer_pin = alerts['buzzer_gpio_pin']
        self.mute_button_pin = alerts['mute_button_gpio_pin']
        self.cooldown_seconds = alerts['cooldown_seconds']
        self.mute_duration_seconds = alerts['mute_duration_seconds']
        self.buzzer_frequency = alerts['buzzer_frequency_hz']
        self.buzzer_duration_ms = alerts['buzzer_duration_ms']
        
        # Variable duration settings
        self.variable_duration = alerts.get('variable_duration', False)
        self.low_risk_duration_ms = alerts.get('low_risk_duration_ms', 1000)
        self.medium_risk_duration_ms = alerts.get('medium_risk_duration_ms', 2500)
        self.high_risk_duration_ms = alerts.get('high_risk_duration_ms', 4000)
        self.low_risk_threshold = alerts.get('low_risk_threshold', 70)
        self.high_risk_threshold = alerts.get('high_risk_threshold', 85)
        
        self.last_alert_time = 0.0
        self.mute_until = 0.0
        self.is_muted = False
        
        if self.gpio_available:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.buzzer_pin, GPIO.OUT)
            GPIO.setup(self.mute_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self.mute_button_thread = threading.Thread(target=self._monitor_mute_button, daemon=True)
            self.mute_button_thread.start()
            print(f"GPIO initialized - Buzzer: GPIO {self.buzzer_pin}, Mute Button: GPIO {self.mute_button_pin}")
        else:
            print("GPIO simulation mode - buzzer and mute button will print to console")
    
    def _monitor_mute_button(self):
        """Monitor mute button in background thread."""
        if not self.gpio_available:
            return
        last_state = GPIO.HIGH
        while True:
            try:
                current_state = GPIO.input(self.mute_button_pin)
                if last_state == GPIO.HIGH and current_state == GPIO.LOW:
                    self.mute()
                last_state = current_state
                time.sleep(0.1)
            except:
                break
    
    def mute(self):
        """Mute alerts for configured duration."""
        self.mute_until = time.time() + self.mute_duration_seconds
        self.is_muted = True
        print(f"Alerts muted for {self.mute_duration_seconds} seconds")
    
    def check_mute_status(self):
        """Check if mute period has expired."""
        if self.is_muted and time.time() >= self.mute_until:
            self.is_muted = False
            print("Mute period expired - alerts re-enabled")
    
    def get_alert_duration(self, risk_score: float) -> int:
        """Get buzzer duration based on risk score level."""
        if not self.variable_duration:
            return self.buzzer_duration_ms
        
        if risk_score < self.low_risk_threshold:
            return self.low_risk_duration_ms
        elif risk_score < self.high_risk_threshold:
            return self.medium_risk_duration_ms
        else:
            return self.high_risk_duration_ms
    
    def trigger_alert(self, force: bool = False, risk_score: float = 70.0) -> bool:
        """Trigger buzzer alert with variable duration based on risk score."""
        current_time = time.time()
        self.check_mute_status()
        
        if not force:
            if self.is_muted:
                return False
            time_since_last = current_time - self.last_alert_time
            if time_since_last < self.cooldown_seconds:
                return False
        
        # Get duration based on risk score
        duration_ms = self.get_alert_duration(risk_score)
        
        if self.gpio_available:
            try:
                # Method 1: Try PWM buzzer (for passive buzzers)
                risk_level = "LOW" if risk_score < self.low_risk_threshold else "MEDIUM" if risk_score < self.high_risk_threshold else "HIGH"
                print(f"🔊 {risk_level} RISK Alert! Score: {risk_score:.1f}, Duration: {duration_ms}ms")
                buzzer = GPIO.PWM(self.buzzer_pin, self.buzzer_frequency)
                buzzer.start(50)  # 50% duty cycle
                time.sleep(duration_ms / 1000.0)
                buzzer.stop()
                print("✅ PWM buzzer completed")
            except Exception as e:
                print(f"❌ PWM buzzer failed: {e}")
                try:
                    # Method 2: Try simple on/off (for active buzzers)
                    print(f"🔊 Trying simple on/off on GPIO {self.buzzer_pin}...")
                    GPIO.output(self.buzzer_pin, GPIO.HIGH)
                    time.sleep(duration_ms / 1000.0)
                    GPIO.output(self.buzzer_pin, GPIO.LOW)
                    print("✅ Simple buzzer completed")
                except Exception as e2:
                    print(f"❌ Simple buzzer also failed: {e2}")
                    print(f"💡 Check: GPIO {self.buzzer_pin} wiring, buzzer type, permissions")
        else:
            risk_level = "LOW" if risk_score < self.low_risk_threshold else "MEDIUM" if risk_score < self.high_risk_threshold else "HIGH"
            print(f"[BUZZER] {risk_level} RISK Alert! Score: {risk_score:.1f}, Duration: {duration_ms}ms")
        
        self.last_alert_time = current_time
        return True
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self.gpio_available:
            try:
                GPIO.cleanup()
            except:
                pass

