"""
Vision processing module for driver drowsiness detection.
Handles face/eye detection, PERCLOS calculation, blink counting, yawn, and head nod detection.
"""
import cv2
import numpy as np
import os
from collections import deque
from typing import Tuple, Optional, Dict
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Using OpenCV Haar cascades as fallback.")


class FaceDetector:
    """Face and facial landmark detection using dlib or OpenCV."""
    
    def __init__(self, config: Dict):
        """
        Initialize face detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.use_dlib = DLIB_AVAILABLE
        
        if self.use_dlib:
            self.detector = dlib.get_frontal_face_detector()
            
            # Try to load shape predictor (68-point facial landmarks)
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                predictor_path = os.path.join(os.path.dirname(__file__), predictor_path)
            
            try:
                if os.path.exists(predictor_path):
                    self.predictor = dlib.shape_predictor(predictor_path)
                else:
                    raise FileNotFoundError("Shape predictor not found")
            except:
                print("Warning: shape_predictor_68_face_landmarks.dat not found.")
                print("Falling back to OpenCV Haar cascades.")
                self.use_dlib = False
        
        if not self.use_dlib:
            # Fallback to OpenCV Haar cascades
            # Try different paths for cascade files
            cascade_paths = [
                getattr(cv2, 'data', None) and cv2.data.haarcascades,  # Standard path
                '/usr/share/opencv4/haarcascades/',  # Debian/Ubuntu system path
                '/usr/local/share/opencv4/haarcascades/',  # Local install path
                '/usr/share/opencv/haarcascades/',  # Older OpenCV path
            ]
            
            cascade_path = None
            for path in cascade_paths:
                if path and os.path.exists(path):
                    cascade_path = path
                    break
            
            if cascade_path is None:
                # Try to find cascades in common locations
                import glob
                possible_locations = [
                    '/usr/share/opencv*/haarcascades/',
                    '/usr/local/share/opencv*/haarcascades/',
                    '/opt/opencv*/share/opencv*/haarcascades/',
                ]
                for pattern in possible_locations:
                    matches = glob.glob(pattern)
                    if matches:
                        cascade_path = matches[0]
                        break
            
            if cascade_path is None:
                raise FileNotFoundError(
                    "Could not find OpenCV Haar cascade files. "
                    "Try: sudo apt install opencv-data"
                )
            
            face_cascade_file = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
            eye_cascade_file = os.path.join(cascade_path, 'haarcascade_eye.xml')
            
            if not os.path.exists(face_cascade_file):
                raise FileNotFoundError(f"Face cascade not found: {face_cascade_file}")
            if not os.path.exists(eye_cascade_file):
                raise FileNotFoundError(f"Eye cascade not found: {eye_cascade_file}")
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_file)
            self.predictor = None
            self.detector = None
            print(f"Using OpenCV Haar cascades from: {cascade_path}")
        
        # Eye aspect ratio thresholds
        self.EYE_AR_THRESH = config['detection']['eye_ar_threshold']
        self.EYE_AR_CONSEC_FRAMES = config['detection']['eye_ar_consec_frames']
        
        # Mouth aspect ratio for yawn detection
        self.MOUTH_AR_THRESH = config['detection']['mouth_ar_threshold']
        self.YAWN_CONSEC_FRAMES = config['detection']['yawn_consec_frames']
        
        # Head pitch for nod detection
        self.HEAD_PITCH_THRESH = config['detection']['head_pitch_threshold']
        self.NOD_CONSEC_FRAMES = config['detection']['nod_consec_frames']
        
        # Blink tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.eye_closed_frames = 0
        self.blink_history = deque(maxlen=int(config['detection']['blink_window_seconds'] * config['cameras']['fps']))
        
        # PERCLOS tracking
        self.perclos_window = int(config['detection']['perclos_window_seconds'] * config['cameras']['fps'])
        self.eye_state_history = deque(maxlen=self.perclos_window)
        
        # Yawn tracking
        self.yawn_frames = 0
        self.yawn_detected = False
        
        # Head nod tracking
        self.nod_frames = 0
        self.nod_detected = False
        
        # Face tracking
        self.face_tracker = None
        self.last_face_position = None
        self.face_lost_frames = 0
        self.tracking_active = False
        self.face_movement_threshold = config['detection'].get('face_movement_threshold', 50)
        self.max_face_lost_frames = config['detection'].get('max_face_lost_frames', 10)
        
        # Face landmarks indices (68-point model)
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.MOUTH_POINTS = list(range(48, 68))
        self.NOSE_TIP = 30
        self.CHIN = 8
        self.FOREHEAD = 27
    
    def eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """Calculate eye aspect ratio (EAR)."""
        if len(eye_points) < 6:
            return 0.3
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        if horizontal == 0:
            return 0.3
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def mouth_aspect_ratio(self, mouth_points: np.ndarray) -> float:
        """Calculate mouth aspect ratio (MAR) for yawn detection."""
        if len(mouth_points) < 12:
            return 0.3
        vertical_1 = np.linalg.norm(mouth_points[2] - mouth_points[10])
        vertical_2 = np.linalg.norm(mouth_points[4] - mouth_points[8])
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
        if horizontal == 0:
            return 0.3
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar
    
    def head_pitch_angle(self, landmarks: np.ndarray) -> float:
        """Estimate head pitch angle from facial landmarks."""
        if len(landmarks) < 31:
            return 0.0
        nose_tip = landmarks[self.NOSE_TIP]
        chin = landmarks[self.CHIN]
        forehead = landmarks[self.FOREHEAD]
        nose_to_chin = chin - nose_tip
        nose_to_forehead = forehead - nose_tip
        dot_product = np.dot(nose_to_chin, nose_to_forehead)
        norm_product = np.linalg.norm(nose_to_chin) * np.linalg.norm(nose_to_forehead)
        if norm_product == 0:
            return 0.0
        cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        if nose_tip[1] > chin[1]:
            return angle
        else:
            return -angle
    
    def init_face_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initialize face tracker with detected face bounding box."""
        try:
            self.face_tracker = cv2.TrackerCSRT_create()
            success = self.face_tracker.init(frame, bbox)
            if success:
                self.tracking_active = True
                self.last_face_position = bbox
                self.face_lost_frames = 0
                print(f"Face tracking initialized: {bbox}")
            else:
                print("Face tracker initialization failed")
            return success
        except Exception as e:
            print(f"Error initializing face tracker: {e}")
            return False
    
    def update_face_tracker(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Update face tracker and return tracked bounding box."""
        if not self.tracking_active or self.face_tracker is None:
            return None
        
        success, bbox = self.face_tracker.update(frame)
        if success:
            bbox = tuple(map(int, bbox))
            self.last_face_position = bbox
            self.face_lost_frames = 0
            return bbox
        else:
            self.face_lost_frames += 1
            if self.face_lost_frames > self.max_face_lost_frames:
                self.tracking_active = False
                self.face_tracker = None
            return self.last_face_position
    
    def detect_face_movement(self, current_pos: Tuple[int, int, int, int]) -> bool:
        """Detect significant face movement."""
        if self.last_face_position is None:
            return False
        
        last_x, last_y, last_w, last_h = self.last_face_position
        curr_x, curr_y, curr_w, curr_h = current_pos
        
        # Calculate center points
        last_center = (last_x + last_w//2, last_y + last_h//2)
        curr_center = (curr_x + curr_w//2, curr_y + curr_h//2)
        
        # Calculate movement distance
        movement = np.sqrt((curr_center[0] - last_center[0])**2 + 
                          (curr_center[1] - last_center[1])**2)
        
        return movement > self.face_movement_threshold
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple]:
        """Detect face and landmarks in frame with tracking support."""
        # Try tracking first if active
        if self.tracking_active:
            tracked_bbox = self.update_face_tracker(frame)
            if tracked_bbox is not None:
                # Use tracked position for landmark detection
                if self.use_dlib and self.predictor is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    x, y, w, h = tracked_bbox
                    face_rect = dlib.rectangle(x, y, x + w, y + h)
                    landmarks = self.predictor(gray, face_rect)
                    landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
                    return (face_rect, landmarks_array, tracked_bbox)
                else:
                    # OpenCV fallback with tracking
                    x, y, w, h = tracked_bbox
                    mock_landmarks = np.array([
                        [x + w//4, y + h//3], [x + w//2, y + h//3], [x + 3*w//4, y + h//3],
                        [x + w//2, y + h//2], [x + w//2, y + 4*h//5], [x + w//2, y + h//4],
                    ])
                    return ((x, y, w, h), mock_landmarks, 2, tracked_bbox)
        
        # Fall back to detection if tracking failed or not active
        if not self.use_dlib:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]
            
            # Initialize tracker with detected face
            if not self.tracking_active:
                self.init_face_tracker(frame, (x, y, w, h))
            
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            eye_count = len(eyes)
            mock_landmarks = np.array([
                [x + w//4, y + h//3], [x + w//2, y + h//3], [x + 3*w//4, y + h//3],
                [x + w//2, y + h//2], [x + w//2, y + 4*h//5], [x + w//2, y + h//4],
            ])
            return ((x, y, w, h), mock_landmarks, eye_count, (x, y, w, h))
        
        if self.predictor is None or self.detector is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        
        face_rect = faces[0]
        
        # Initialize tracker with detected face
        bbox = (face_rect.left(), face_rect.top(), 
               face_rect.width(), face_rect.height())
        if not self.tracking_active:
            self.init_face_tracker(frame, bbox)
        
        landmarks = self.predictor(gray, face_rect)
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        return (face_rect, landmarks_array, bbox)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame for drowsiness indicators."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        result = {
            'face_detected': False, 'left_ear': 0.0, 'right_ear': 0.0, 'avg_ear': 0.0,
            'mar': 0.0, 'head_pitch': 0.0, 'eyes_closed': False,
            'blink_detected': False, 'yawn_detected': False, 'nod_detected': False,
            'face_tracked': self.tracking_active, 'face_movement_detected': False,
            'face_bbox': None, 'tracking_center': None, 'last_center': None
        }
        
        face_data = self.detect_face(frame)
        if face_data is None:
            self.eye_state_history.append(False)
            return result
        
        result['face_detected'] = True
        result['face_tracked'] = self.tracking_active
        
        # Extract face bounding box for display (always show when face detected)
        face_bbox = None
        if self.use_dlib and len(face_data) >= 2:
            face_rect = face_data[0]
            if hasattr(face_rect, 'left'):
                # dlib rectangle
                face_bbox = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
            else:
                # tuple format
                face_bbox = face_rect
        elif not self.use_dlib and len(face_data) >= 1:
            face_bbox = face_data[0]  # OpenCV returns (x,y,w,h) as first element
        
        # Always set face_bbox when face is detected
        if face_bbox:
            result['face_bbox'] = face_bbox
            x, y, w, h = face_bbox
            result['tracking_center'] = (x + w//2, y + h//2)
            if self.last_face_position:
                lx, ly, lw, lh = self.last_face_position
                result['last_center'] = (lx + lw//2, ly + lh//2)
        
        # Check for face movement if tracking
        tracked_bbox = None
        if len(face_data) >= 3:
            tracked_bbox = face_data[-1]  # Last element should be bbox
        elif len(face_data) >= 4:
            tracked_bbox = face_data[3]   # Fourth element for OpenCV case
        
        if tracked_bbox is not None and isinstance(tracked_bbox, tuple) and len(tracked_bbox) == 4:
            result['face_movement_detected'] = self.detect_face_movement(tracked_bbox)
        
        if self.use_dlib and len(face_data) >= 2:
            face_rect, landmarks = face_data[0], face_data[1]
            left_eye = landmarks[self.LEFT_EYE_POINTS]
            right_eye = landmarks[self.RIGHT_EYE_POINTS]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            result['left_ear'] = left_ear
            result['right_ear'] = right_ear
            result['avg_ear'] = avg_ear
            mouth = landmarks[self.MOUTH_POINTS]
            mar = self.mouth_aspect_ratio(mouth)
            result['mar'] = mar
            pitch = self.head_pitch_angle(landmarks)
            result['head_pitch'] = pitch
        else:
            if len(face_data) >= 3:
                face_rect, landmarks, eye_count = face_data[0], face_data[1], face_data[2]
            else:
                face_rect, landmarks = face_data[0], face_data[1]
                eye_count = 2
            eyes_visible = eye_count >= 2
            avg_ear = 0.3 if eyes_visible else 0.15
            result['left_ear'] = avg_ear
            result['right_ear'] = avg_ear
            result['avg_ear'] = avg_ear
            result['mar'] = 0.3
            result['head_pitch'] = 0.0
        
        eyes_closed = result['avg_ear'] < self.EYE_AR_THRESH
        result['eyes_closed'] = eyes_closed
        self.eye_state_history.append(eyes_closed)
        
        if eyes_closed:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
                result['blink_detected'] = True
                self.blink_history.append(True)
            else:
                self.blink_history.append(False)
            self.eye_closed_frames = 0
        
        if self.use_dlib:
            mar = result['mar']
            if mar > self.MOUTH_AR_THRESH:
                self.yawn_frames += 1
                if self.yawn_frames >= self.YAWN_CONSEC_FRAMES:
                    result['yawn_detected'] = True
                    self.yawn_detected = True
            else:
                self.yawn_frames = 0
                self.yawn_detected = False
            
            pitch = result['head_pitch']
            if pitch > self.HEAD_PITCH_THRESH:
                self.nod_frames += 1
                if self.nod_frames >= self.NOD_CONSEC_FRAMES:
                    result['nod_detected'] = True
                    self.nod_detected = True
            else:
                self.nod_frames = 0
                self.nod_detected = False
        
        return result
    
    def get_perclos(self) -> float:
        """Calculate PERCLOS (Percentage of Eyelid Closure)."""
        if len(self.eye_state_history) == 0:
            return 0.0
        closed_count = sum(self.eye_state_history)
        return closed_count / len(self.eye_state_history)
    
    def get_blinks_per_minute(self) -> float:
        """Calculate blinks per minute."""
        if len(self.blink_history) == 0:
            return 0.0
        fps = self.config['cameras']['fps']
        window_seconds = len(self.blink_history) / fps
        blink_count = sum(self.blink_history)
        if window_seconds == 0:
            return 0.0
        return (blink_count / window_seconds) * 60.0
    
    def reset(self):
        """Reset all tracking counters."""
        self.blink_counter = 0
        self.total_blinks = 0
        self.eye_closed_frames = 0
        self.blink_history.clear()
        self.eye_state_history.clear()
        self.yawn_frames = 0
        self.yawn_detected = False
        self.nod_frames = 0
        self.nod_detected = False
        # Reset face tracking
        self.face_tracker = None
        self.last_face_position = None
        self.face_lost_frames = 0
        self.tracking_active = False


class RoadDetector:
    """Road camera analysis for car proximity and texting detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.car_close_threshold = config['road_detection']['car_close_threshold']
        self.motion_sensitivity = config['road_detection']['motion_sensitivity']
        self.texting_enabled = config['road_detection']['texting_detection_enabled']
        self.texting_threshold = config['road_detection']['texting_hand_threshold']
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.prev_frame = None
    
    def detect_car_close(self, frame: np.ndarray) -> bool:
        """Detect if a car is suddenly close ahead."""
        if frame is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        h, w = frame.shape[:2]
        lower_center_region = (int(w * 0.3), int(h * 0.5), int(w * 0.7), h)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (w * h * self.car_close_threshold):
                x, y, cw, ch = cv2.boundingRect(contour)
                if (lower_center_region[0] < x < lower_center_region[2] and 
                    lower_center_region[1] < y < lower_center_region[3]):
                    return True
        return False
    
    def detect_texting(self, frame: np.ndarray) -> bool:
        """Detect if driver is texting/reading (hand near face area)."""
        if not self.texting_enabled or frame is None:
            return False
        h, w = frame.shape[:2]
        roi = frame[0:int(h * 0.6), int(w * 0.2):int(w * 0.8)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
        return skin_ratio > self.texting_threshold

