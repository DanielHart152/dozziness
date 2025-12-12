"""
Risk score calculation module.
Combines PERCLOS, blinks per minute, yawn, and nod into a 0-100 risk score.
"""
from typing import Dict


class RiskScorer:
    """Calculate drowsiness risk score from various indicators."""
    
    def __init__(self, config: Dict):
        self.config = config
        scoring = config['scoring']
        self.perclos_weight = scoring['perclos_weight']
        self.blinks_per_min_weight = scoring['blinks_per_min_weight']
        self.perclos_threshold = scoring['perclos_threshold']
        self.blinks_per_min_threshold = scoring['blinks_per_min_threshold']
        self.yawn_penalty = scoring['yawn_penalty']
        self.nod_penalty = scoring['nod_penalty']
        self.texting_penalty = scoring['texting_penalty']
        self.car_close_penalty = scoring['car_close_penalty']
    
    def calculate_score(self, perclos: float, blinks_per_min: float, 
                       yawn_detected: bool = False, nod_detected: bool = False,
                       texting_detected: bool = False, car_close: bool = False) -> float:
        """Calculate risk score (0-100) from drowsiness indicators."""
        score = 0.0
        perclos_score = min(100.0, (perclos / self.perclos_threshold) * 50.0)
        score += perclos_score * self.perclos_weight
        if blinks_per_min > 0:
            blinks_score = min(100.0, (blinks_per_min / self.blinks_per_min_threshold) * 50.0)
        else:
            blinks_score = 30.0
        score += blinks_score * self.blinks_per_min_weight
        if yawn_detected:
            score += self.yawn_penalty
        if nod_detected:
            score += self.nod_penalty
        if texting_detected:
            score += self.texting_penalty
        if car_close:
            score += self.car_close_penalty
        score = max(0.0, min(100.0, score))
        return score
    
    def should_alert(self, score: float) -> bool:
        """Determine if alert should be triggered based on risk score."""
        threshold = self.config['alerts']['risk_threshold']
        return score >= threshold

