"""
Risk scoring + alert logic for Driver Drowsiness Detection
- PERCLOS (heavier weight)
- Blink score (triangular by default; inverted linear option)
- Windowed penalties (cap per 10s; refractory per event)
- Hysteresis + hold times + cooldown for alerts
"""

from collections import deque
import time
import logging
from typing import Dict, Optional

# Setup debug logger
logging.basicConfig(level=logging.INFO)
debug_logger = logging.getLogger('scorer_debug')
debug_logger.setLevel(logging.DEBUG)

# Create file handler for debug logs
if not debug_logger.handlers:
    handler = logging.FileHandler('scorer_debug.log')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    debug_logger.addHandler(handler)


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
        
        # Debug logging control
        logging_config = config.get('logging', {})
        self.debug_interval = logging_config.get('debug_interval_seconds', 5.0)
        self._last_debug_log = 0
        
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
        final = raw * self.perclos_weight
        
        if (time.time() - self._last_debug_log) < self.debug_interval:
            debug_logger.debug(f"PERCLOS: input={perclos:.4f}, clamped={perclos_c:.4f}, "
                              f"raw={raw:.2f}, final={final:.2f} (weight={self.perclos_weight})")
        return final
    
    def _blink_points(self, bpm: float) -> float:
        """Calculate blink rate contribution to score."""
        if self.blink_mode == 'triangular':
            # Triangular: optimal at target bpm, decreases linearly to 0 at target±tolerance
            target = self.blink_target
            tol = max(1e-6, self.blink_tolerance)
            deviation = abs(bpm - target)
            raw = 50.0 * (1.0 - deviation / tol)
            raw = clamp(raw, 0.0, 50.0)
            if (time.time() - self._last_debug_log) < self.debug_interval:
                debug_logger.debug(f"BLINKS (triangular): bpm={bpm:.1f}, target={target}, "
                                  f"deviation={deviation:.1f}, tol={tol}, raw={raw:.2f}")
        else:
            # Inverted linear: 0 bpm → 50 points, 20 bpm → 0 points
            x = clamp(bpm / 20.0, 0.0, 1.0)
            raw = (1.0 - x) * 50.0
            if (time.time() - self._last_debug_log) < self.debug_interval:
                debug_logger.debug(f"BLINKS (inverted): bpm={bpm:.1f}, x={x:.3f}, raw={raw:.2f}")
        
        final = raw * self.blink_weight
        if (time.time() - self._last_debug_log) < self.debug_interval:
            debug_logger.debug(f"BLINKS final: {final:.2f} (weight={self.blink_weight})")
        return final
    
    def _apply_penalties(self, events: Dict[str, bool], now: float) -> float:
        """Apply windowed penalties with refractory periods."""
        new_penalties = []
        
        # Award new points if event fired and refractory period elapsed
        for name, fired in (events or {}).items():
            if not fired or name not in self.penalty_points:
                continue
                
            last_t = self._last_event_time.get(name, -1e9)
            refractory = self.penalty_refractory_sec.get(name, 0.0)
            time_since = now - last_t
            
            if time_since >= refractory:
                pts = float(self.penalty_points[name])
                self._penalty_hist.append((now, pts))
                self._last_event_time[name] = now
                new_penalties.append((name, pts))
                debug_logger.debug(f"PENALTY awarded: {name}={pts}pts (last={time_since:.1f}s ago)")
            else:
                debug_logger.debug(f"PENALTY blocked: {name} in refractory ({time_since:.1f}s < {refractory}s)")
        
        # Remove old penalty points outside window
        removed_count = 0
        while self._penalty_hist and (now - self._penalty_hist[0][0]) > self.penalty_window_sec:
            old_penalty = self._penalty_hist.popleft()
            removed_count += 1
        
        if removed_count > 0:
            debug_logger.debug(f"PENALTY cleanup: removed {removed_count} old penalties")
        
        # Cap total penalty contribution within window
        total = sum(p for (_, p) in self._penalty_hist)
        capped = min(total, self.penalty_window_cap)
        
        active_penalties = [(now - t, p) for t, p in self._penalty_hist]
        debug_logger.debug(f"PENALTY window: {len(active_penalties)} active, total={total:.1f}, "
                          f"capped={capped:.1f}, new={new_penalties}")
        
        return capped
    
    def calculate_score(self, perclos: float, blinks_per_min: float, 
                       yawn_detected: bool = False, nod_detected: bool = False,
                       texting_detected: bool = False, car_close: bool = False,
                       now: Optional[float] = None) -> float:
        """Calculate risk score (0-100) with enhanced logic."""
        if now is None:
            now = time.time()
        
        # Only log detailed debug info at intervals
        should_debug = (now - self._last_debug_log) >= self.debug_interval
        if should_debug:
            self._last_debug_log = now
            debug_logger.debug(f"\n=== SCORE CALCULATION START ===")
            debug_logger.debug(f"Inputs: PERCLOS={perclos:.4f}, BPM={blinks_per_min:.1f}, "
                              f"yawn={yawn_detected}, nod={nod_detected}, texting={texting_detected}, car_close={car_close}")
        
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
        raw_score = p_pts + b_pts + pen_pts
        score = clamp(raw_score, 0.0, 100.0)
        
        if should_debug:
            debug_logger.debug(f"Score components: PERCLOS={p_pts:.2f} + BLINKS={b_pts:.2f} + PENALTIES={pen_pts:.2f} = {raw_score:.2f}")
            debug_logger.debug(f"Final score: {score:.2f} (clamped from {raw_score:.2f})")
        
        # Record for rolling average windows
        self._score_hist.append((now, score))
        # Keep ~3.5 seconds of history (enough for ON 1s & OFF 2s windows)
        hist_removed = 0
        while self._score_hist and (now - self._score_hist[0][0]) > 3.5:
            self._score_hist.popleft()
            hist_removed += 1
        
        if should_debug:
            if hist_removed > 0:
                debug_logger.debug(f"Score history: removed {hist_removed} old entries, {len(self._score_hist)} remaining")
            debug_logger.debug(f"=== SCORE CALCULATION END ===\n")
        return score
    
    def _rolling_avg(self, horizon_sec: float, now: float) -> float:
        """Calculate rolling average over specified time horizon."""
        total, count = 0.0, 0
        scores_in_window = []
        
        for t, s in reversed(self._score_hist):
            age = now - t
            if age > horizon_sec:
                break
            total += s
            count += 1
            scores_in_window.append((age, s))
        
        avg = (total / count) if count else 0.0
        if (time.time() - self._last_debug_log) < self.debug_interval:
            debug_logger.debug(f"Rolling avg ({horizon_sec}s): {count} scores, avg={avg:.2f}, "
                              f"scores={([(f'{age:.1f}s', f'{s:.1f}') for age, s in scores_in_window[:3]])}")
        return avg
    
    def should_alert(self, score: Optional[float] = None, now: Optional[float] = None) -> bool:
        """Determine if alert should be triggered using hysteresis and hold times."""
        if now is None:
            now = time.time()
        
        # Always log alert decisions (important events)
        debug_logger.debug(f"\n--- ALERT DECISION START ---")
        
        # Ensure history includes latest score if provided ad-hoc
        if score is not None:
            clamped_score = clamp(score, 0.0, 100.0)
            self._score_hist.append((now, clamped_score))
            debug_logger.debug(f"Added ad-hoc score: {clamped_score:.2f}")
            while self._score_hist and (now - self._score_hist[0][0]) > 3.5:
                self._score_hist.popleft()
        
        # Calculate rolling averages
        if (now - self._last_debug_log) < self.debug_interval:
            debug_logger.debug(f"Calculating rolling averages...")
        avg_on = self._rolling_avg(self.hold_on_sec, now)
        avg_off = self._rolling_avg(self.hold_off_sec, now)
        
        # Hysteresis logic
        prev_alerting = self._is_alerting
        cooldown_remaining = max(0, self.alert_cooldown_sec - (now - self._last_alert_on))
        
        if (now - self._last_debug_log) < self.debug_interval:
            debug_logger.debug(f"Alert state: currently={self._is_alerting}, avg_on={avg_on:.2f} (need>={self.alert_on}), "
                              f"avg_off={avg_off:.2f} (need<={self.alert_off}), cooldown={cooldown_remaining:.1f}s")
        
        if not self._is_alerting:
            # Turn ON: score high enough AND cooldown elapsed
            can_turn_on = avg_on >= self.alert_on and cooldown_remaining <= 0
            if can_turn_on:
                self._is_alerting = True
                self._last_alert_on = now
                debug_logger.info(f"ALERT TRIGGERED: avg_on={avg_on:.2f} >= {self.alert_on}, cooldown elapsed")
            else:
                reasons = []
                if avg_on < self.alert_on:
                    reasons.append(f"avg_on={avg_on:.2f} < {self.alert_on}")
                if cooldown_remaining > 0:
                    reasons.append(f"cooldown={cooldown_remaining:.1f}s")
                if (now - self._last_debug_log) < self.debug_interval:
                    debug_logger.debug(f"Alert blocked: {', '.join(reasons)}")
        else:
            # Turn OFF: score low enough for hold period
            if avg_off <= self.alert_off:
                self._is_alerting = False
                debug_logger.info(f"ALERT CLEARED: avg_off={avg_off:.2f} <= {self.alert_off}")
            else:
                if (now - self._last_debug_log) < self.debug_interval:
                    debug_logger.debug(f"Alert continues: avg_off={avg_off:.2f} > {self.alert_off}")
        
        if self._is_alerting != prev_alerting:
            debug_logger.info(f"ALERT STATE CHANGED: {prev_alerting} -> {self._is_alerting}")
        
        if (now - self._last_debug_log) < self.debug_interval:
            debug_logger.debug(f"--- ALERT DECISION END: {self._is_alerting} ---\n")
        return self._is_alerting