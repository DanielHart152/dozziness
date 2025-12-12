"""
CSV logging module for drowsiness metrics.
"""
import csv
import os
from datetime import datetime
from typing import Dict


class DataLogger:
    """Log drowsiness metrics to CSV file."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.csv_file = config['logging']['csv_file']
        self.log_interval = config['logging']['log_interval_seconds']
        self.last_log_time = 0.0
        
        if not os.path.exists(self.csv_file):
            self._write_header()
    
    def _write_header(self):
        """Write CSV header row."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'perclos', 'blinks_per_min', 'risk_score', 'alert_triggered',
                'yawn_detected', 'nod_detected', 'texting_detected', 'car_close',
                'left_ear', 'right_ear', 'mar', 'head_pitch'
            ])
    
    def log(self, metrics: Dict, force: bool = False):
        """Log metrics to CSV file."""
        import time
        current_time = time.time()
        
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        row = [
            timestamp, metrics.get('perclos', 0.0), metrics.get('blinks_per_min', 0.0),
            metrics.get('risk_score', 0.0), metrics.get('alert_triggered', False),
            metrics.get('yawn_detected', False), metrics.get('nod_detected', False),
            metrics.get('texting_detected', False), metrics.get('car_close', False),
            metrics.get('left_ear', 0.0), metrics.get('right_ear', 0.0),
            metrics.get('mar', 0.0), metrics.get('head_pitch', 0.0)
        ]
        
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            self.last_log_time = current_time
        except Exception as e:
            print(f"Error writing to log file: {e}")

