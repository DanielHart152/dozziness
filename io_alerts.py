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
    
    def trigger_alert(self, force: bool = False) -> bool:
        """Trigger buzzer alert if conditions are met."""
        current_time = time.time()
        self.check_mute_status()
        
        if not force:
            if self.is_muted:
                return False
            time_since_last = current_time - self.last_alert_time
            if time_since_last < self.cooldown_seconds:
                return False
        
        if self.gpio_available:
            try:
                buzzer = GPIO.PWM(self.buzzer_pin, self.buzzer_frequency)
                buzzer.start(50)
                time.sleep(self.buzzer_duration_ms / 1000.0)
                buzzer.stop()
            except Exception as e:
                print(f"Error triggering buzzer: {e}")
        else:
            print(f"[BUZZER] Alert triggered! (Frequency: {self.buzzer_frequency}Hz, Duration: {self.buzzer_duration_ms}ms)")
        
        self.last_alert_time = current_time
        return True
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self.gpio_available:
            try:
                GPIO.cleanup()
            except:
                pass

