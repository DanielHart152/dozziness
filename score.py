"""
Risk scoring + alert logic for Driver Drowsiness Detection
- PERCLOS (heavier weight)
- Blink score (triangular by default; inverted linear option)
- Windowed penalties (cap per 10s; refractory per event)
- Hysteresis + hold times + cooldown for alerts
"""

from collections import deque
import time
from typing import Dict, Optional


def clamp(x, lo, hi):
    """Clamp value between lo and hi."""
    return lo if x < lo else hi if x > hi else x


class RiskScorer:
    """Enhanced risk scorer with windowed penalties and hysteresis alerting."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Enhanced configuration with defaults
        scoring = config.get('scoring', {})
        alerts = config.get('alerts', {})
        
        # Feature weights and scaling
        self.perclos_max = scoring.get('perclos_max', 0.25)  # cap PERCLOS at 25%
        self.perclos_weight = scoring.get('perclos_weight', 0.60)  # 60% weight
        self.blink_weight = scoring.get('blinks_per_min_weight', 0.40)  # 40% weight
        self.blink_mode = scoring.get('blink_mode', 'triangular')  # or 'inverted_linear'
        self.blink_target = scoring.get('blink_target', 16.0)  # optimal bpm
        self.blink_tolerance = scoring.get('blink_tolerance', 16.0)  # triangle half-width
        
        # Penalty configuration
        self.penalty_points = {
            'yawn': scoring.get('yawn_penalty', 10),
            'nod': scoring.get('nod_penalty', 5),
            'texting': scoring.get('texting_penalty', 25),
            'car_close': scoring.get('car_close_penalty', 30),
        }
        
        # Refractory periods (minimum time between same event)
        self.penalty_refractory_sec = {
            'yawn': scoring.get('yawn_refractory_sec', 10.0),
            'nod': scoring.get('nod_refractory_sec', 5.0),
            'texting': scoring.get('texting_refractory_sec', 10.0),
            'car_close': scoring.get('car_close_refractory_sec', 5.0),
        }
        
        self.penalty_window_sec = scoring.get('penalty_window_sec', 10.0)  # sliding window
        self.penalty_window_cap = scoring.get('penalty_window_cap', 40.0)  # max penalty in window
        
        # Alert logic with hysteresis
        self.alert_on = alerts.get('alert_on_threshold', 70.0)  # turn on at this score
        self.alert_off = alerts.get('alert_off_threshold', 55.0)  # turn off at this score
        self.hold_on_sec = alerts.get('hold_on_sec', 1.0)  # averaging window for ON
        self.hold_off_sec = alerts.get('hold_off_sec', 2.0)  # averaging window for OFF
        self.alert_cooldown_sec = alerts.get('cooldown_seconds', 8.0)  # min gap between alerts
        
        # State tracking
        self._score_hist = deque()  # (timestamp, score) for rolling averages
        self._penalty_hist = deque()  # (timestamp, points) for windowed penalties
        self._last_event_time = {name: -1e9 for name in self.penalty_points.keys()}
        self._is_alerting = False
        self._last_alert_on = -1e9
    
    def _perclos_points(self, perclos: float) -> float:
        """Calculate PERCLOS contribution to score."""
        # Cap PERCLOS, scale to 0..50, then apply weight
        perclos_c = clamp(perclos, 0.0, self.perclos_max)
        raw = (perclos_c / self.perclos_max) * 50.0
        return raw * self.perclos_weight  # max 30 points (50 * 0.6)
    
    def _blink_points(self, bpm: float) -> float:
        """Calculate blink rate contribution to score."""
        if self.blink_mode == 'triangular':
            # Triangular: optimal at target bpm, decreases linearly to 0 at target±tolerance
            target = self.blink_target
            tol = max(1e-6, self.blink_tolerance)
            raw = 50.0 * (1.0 - abs(bpm - target) / tol)
            raw = clamp(raw, 0.0, 50.0)
        else:
            # Inverted linear: 0 bpm → 50 points, 20 bpm → 0 points
            x = clamp(bpm / 20.0, 0.0, 1.0)
            raw = (1.0 - x) * 50.0
        
        return raw * self.blink_weight  # max 20 points (50 * 0.4)
    
    def _apply_penalties(self, events: Dict[str, bool], now: float) -> float:
        """Apply windowed penalties with refractory periods."""
        # Award new points if event fired and refractory period elapsed
        for name, fired in (events or {}).items():
            if not fired or name not in self.penalty_points:
                continue
                
            last_t = self._last_event_time.get(name, -1e9)
            refractory = self.penalty_refractory_sec.get(name, 0.0)
            
            if (now - last_t) >= refractory:
                pts = float(self.penalty_points[name])
                self._penalty_hist.append((now, pts))
                self._last_event_time[name] = now
        
        # Remove old penalty points outside window
        while self._penalty_hist and (now - self._penalty_hist[0][0]) > self.penalty_window_sec:
            self._penalty_hist.popleft()
        
        # Cap total penalty contribution within window
        total = sum(p for (_, p) in self._penalty_hist)
        return min(total, self.penalty_window_cap)
    
    def calculate_score(self, perclos: float, blinks_per_min: float, 
                       yawn_detected: bool = False, nod_detected: bool = False,
                       texting_detected: bool = False, car_close: bool = False,
                       now: Optional[float] = None) -> float:
        """Calculate risk score (0-100) with enhanced logic."""
        if now is None:
            now = time.time()
        
        # Calculate component scores
        p_pts = self._perclos_points(perclos)
        b_pts = self._blink_points(blinks_per_min)
        
        # Apply penalties with windowing and refractory
        events = {
            'yawn': yawn_detected,
            'nod': nod_detected,
            'texting': texting_detected,
            'car_close': car_close
        }
        pen_pts = self._apply_penalties(events, now)
        
        # Combine and clamp
        score = p_pts + b_pts + pen_pts
        score = clamp(score, 0.0, 100.0)
        
        # Record for rolling average windows
        self._score_hist.append((now, score))
        # Keep ~3.5 seconds of history (enough for ON 1s & OFF 2s windows)
        while self._score_hist and (now - self._score_hist[0][0]) > 3.5:
            self._score_hist.popleft()
        
        return score
    
    def _rolling_avg(self, horizon_sec: float, now: float) -> float:
        """Calculate rolling average over specified time horizon."""
        total, count = 0.0, 0
        for t, s in reversed(self._score_hist):
            if now - t > horizon_sec:
                break
            total += s
            count += 1
        return (total / count) if count else 0.0
    
    def should_alert(self, score: Optional[float] = None, now: Optional[float] = None) -> bool:
        """Determine if alert should be triggered using hysteresis and hold times."""
        if now is None:
            now = time.time()
        
        # Ensure history includes latest score if provided ad-hoc
        if score is not None:
            self._score_hist.append((now, clamp(score, 0.0, 100.0)))
            while self._score_hist and (now - self._score_hist[0][0]) > 3.5:
                self._score_hist.popleft()
        
        # Calculate rolling averages
        avg_on = self._rolling_avg(self.hold_on_sec, now)
        avg_off = self._rolling_avg(self.hold_off_sec, now)
        
        # Hysteresis logic
        if not self._is_alerting:
            # Turn ON: score high enough AND cooldown elapsed
            if avg_on >= self.alert_on and (now - self._last_alert_on) >= self.alert_cooldown_sec:
                self._is_alerting = True
                self._last_alert_on = now
        else:
            # Turn OFF: score low enough for hold period
            if avg_off <= self.alert_off:
                self._is_alerting = False
        
        return self._is_alerting