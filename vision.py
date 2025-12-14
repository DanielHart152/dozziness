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
            # Check OpenCV version and use appropriate tracker API
            cv_version = cv2.__version__
            print(f"🎯 TRACKING: OpenCV version {cv_version}")
            
            trackers = []
            
            # For OpenCV 4.5.1+, trackers are in cv2.legacy
            if hasattr(cv2, 'legacy'):
                try:
                    # Try the most common working APIs
                    if hasattr(cv2.legacy, 'TrackerKCF_create'):
                        trackers.append(lambda: cv2.legacy.TrackerKCF_create())
                        print(f"🎯 TRACKING: Added legacy.TrackerKCF_create")
                    if hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        trackers.append(lambda: cv2.legacy.TrackerCSRT_create())
                        print(f"🎯 TRACKING: Added legacy.TrackerCSRT_create")
                    if hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                        trackers.append(lambda: cv2.legacy.TrackerMOSSE_create())
                        print(f"🎯 TRACKING: Added legacy.TrackerMOSSE_create")
                except Exception as e:
                    print(f"🎯 TRACKING: Legacy tracker error: {e}")
            
            # For older OpenCV versions
            if hasattr(cv2, 'TrackerKCF_create'):
                trackers.append(lambda: cv2.TrackerKCF_create())
                print(f"🎯 TRACKING: Added TrackerKCF_create")
            if hasattr(cv2, 'TrackerCSRT_create'):
                trackers.append(lambda: cv2.TrackerCSRT_create())
                print(f"🎯 TRACKING: Added TrackerCSRT_create")
            
            print(f"🎯 TRACKING: Found {len(trackers)} tracker types to try")
            
            if len(trackers) == 0:
                print(f"🎯 TRACKING: No trackers available in OpenCV {cv_version}")
                print(f"🎯 TRACKING: Available cv2 attributes: {[attr for attr in dir(cv2) if 'Track' in attr]}")
                if hasattr(cv2, 'legacy'):
                    print(f"🎯 TRACKING: Available cv2.legacy attributes: {[attr for attr in dir(cv2.legacy) if 'Track' in attr]}")
            
            for tracker_func in trackers:
                try:
                    if tracker_func is None:
                        continue
                    self.face_tracker = tracker_func()
                    if self.face_tracker is not None:
                        print(f"🎯 TRACKING: Trying to init tracker with bbox {bbox}")
                        success = self.face_tracker.init(frame, bbox)
                        print(f"🎯 TRACKING: Tracker init returned: {success}")
                        if success:
                            self.tracking_active = True
                            self.last_face_position = bbox
                            self.face_lost_frames = 0
                            print(f"🎯 TRACKING: ✅ Face tracker initialized successfully at {bbox}")
                            return True
                        else:
                            print(f"🎯 TRACKING: ❌ Tracker init failed for this tracker type")
                except Exception as e:
                    print(f"🎯 TRACKING: Tracker failed: {e}")
                    continue
            
            # If no trackers available or all fail, provide helpful info
            print("🎯 TRACKING: ❌ No working OpenCV trackers found")
            print(f"🎯 TRACKING: Your OpenCV version: {cv2.__version__}")
            print("🎯 TRACKING: Try: pip install opencv-contrib-python")
            self.tracking_active = False
            self.face_tracker = None
            return False
        except Exception as e:
            self.tracking_active = False
            self.face_tracker = None
            print(f"🎯 TRACKING: ❌ Face tracker initialization error: {e}")
            return False
    
    def update_face_tracker(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Update face tracker and return tracked bounding box."""
        if not self.tracking_active:
            return None
        
        # Simple tracking fallback - just return current position
        # (position gets updated by detection in detect_face)
        if self.face_tracker == "simple":
            return self.last_face_position
        
        # OpenCV tracker
        if self.face_tracker is not None:
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
                    print(f"🎯 TRACKING: ❌ Tracker deactivated due to too many lost frames")
        
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
        # Always do fresh detection first, then use tracking for consistency
        # This ensures we get accurate face positions
        
        # Fall back to detection if tracking failed or not active
        if not self.use_dlib:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]
            
            # Initialize tracker with detected face
            if not hasattr(self, '_opencv_debug_count'):
                self._opencv_debug_count = 0
            self._opencv_debug_count += 1
            
            if self._opencv_debug_count % 30 == 0:
                print(f"🎯 TRACKING: OpenCV face detected at ({x}, {y}, {w}, {h}), tracking_active={self.tracking_active}")
            
            # Only try to initialize tracking occasionally
            if not self.tracking_active and self._opencv_debug_count % 10 == 0:
                if self._opencv_debug_count % 30 == 0:
                    print(f"🎯 TRACKING: Attempting to initialize OpenCV tracker...")
                success = self.init_face_tracker(frame, (x, y, w, h))
                if success:
                    print(f"🎯 TRACKING: ✅ OpenCV tracker initialized successfully")
                elif self._opencv_debug_count % 30 == 0:
                    print(f"🎯 TRACKING: ❌ OpenCV tracker init failed")
            
            if self.tracking_active:
                # Update tracker position with current detection
                self.last_face_position = (x, y, w, h)
            
            roi_gray = gray[y:y+h, x:x+w]
            # Try multiple detection parameters for better eye detection
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
            if len(eyes) == 0:
                # Try more sensitive detection
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(8, 8))
            eye_count = len(eyes)
            print(f"DEBUG: OpenCV eye detection - found {eye_count} eyes in face region")
            
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
        detection_bbox = (face_rect.left(), face_rect.top(), 
                         face_rect.width(), face_rect.height())
        
        # Only print debug occasionally to reduce spam
        if not hasattr(self, '_dlib_debug_count'):
            self._dlib_debug_count = 0
        self._dlib_debug_count += 1
        
        if self._dlib_debug_count % 30 == 0:
            print(f"🎯 TRACKING: dlib face detected at {detection_bbox}, tracking_active={self.tracking_active}")
        
        # Only try to initialize tracking occasionally to avoid spam
        if not self.tracking_active and self._dlib_debug_count % 10 == 0:
            if self._dlib_debug_count % 30 == 0:
                print(f"🎯 TRACKING: Attempting to initialize tracker...")
            success = self.init_face_tracker(frame, detection_bbox)
            if success:
                print(f"🎯 TRACKING: ✅ Tracker initialized successfully")
            elif self._dlib_debug_count % 30 == 0:
                print(f"🎯 TRACKING: ❌ Tracker init failed")
        
        if self.tracking_active:
            # Update tracker position with current detection
            self.last_face_position = detection_bbox
        
        landmarks = self.predictor(gray, face_rect)
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        return (face_rect, landmarks_array, detection_bbox, detection_bbox)
    
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
            # No face detected - turn off tracking
            if self.tracking_active:
                print("🎯 TRACKING: ❌ Face lost, deactivating tracking")
                self.tracking_active = False
                self.face_tracker = None
            self.eye_state_history.append(False)
            return result
        
        result['face_detected'] = True
        result['face_tracked'] = self.tracking_active
        
        # Debug tracking status every few frames
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1
        
        if self._debug_frame_count % 30 == 0:  # Every 30 frames
            print(f"🎯 TRACKING STATUS: active={self.tracking_active}, tracker_exists={self.face_tracker is not None}")
        
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
        if face_bbox and len(face_bbox) == 4:
            result['face_bbox'] = face_bbox
            x, y, w, h = face_bbox
            result['tracking_center'] = (x + w//2, y + h//2)
            # Update last position for movement detection
            if self.last_face_position and self.last_face_position != face_bbox:
                lx, ly, lw, lh = self.last_face_position
                result['last_center'] = (lx + lw//2, ly + lh//2)
            self.last_face_position = face_bbox
            # print(f"DEBUG: Face bbox set: {face_bbox}")  # Debug output
        else:
            print(f"DEBUG: Face detected but no valid bbox. face_bbox={face_bbox}, face_data length={len(face_data) if face_data else 0}")
            
        # Check for face movement if tracking
        tracked_bbox = None
        if len(face_data) >= 4:
            tracked_bbox = face_data[3]   # Fourth element for bbox
        elif len(face_data) >= 3:
            tracked_bbox = face_data[2]   # Third element fallback
        
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
            # Use actual eye detection results for real blink detection
            if eye_count >= 2:
                avg_ear = 0.3  # Eyes clearly visible = open
            elif eye_count == 1:
                avg_ear = 0.25  # One eye detected = borderline
            else:
                avg_ear = 0.2   # No eyes detected = likely closed/blinking
            result['left_ear'] = avg_ear
            result['right_ear'] = avg_ear
            result['avg_ear'] = avg_ear
            result['mar'] = 0.3
            result['head_pitch'] = 0.0
            print(f"DEBUG: OpenCV mode - eye_count={eye_count}, avg_ear={avg_ear} ({'open' if avg_ear >= self.EYE_AR_THRESH else 'closed'})")
        
        eyes_closed = result['avg_ear'] < self.EYE_AR_THRESH
        result['eyes_closed'] = eyes_closed
        self.eye_state_history.append(eyes_closed)
        
        # Debug eye detection (only when state changes)
        if not hasattr(self, '_last_eye_state') or self._last_eye_state != eyes_closed:
            print(f"DEBUG: EYE STATE CHANGE - EAR={result['avg_ear']:.3f}, threshold={self.EYE_AR_THRESH}, closed={eyes_closed}")
            self._last_eye_state = eyes_closed
        
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
        perclos = closed_count / len(self.eye_state_history)
        print(f"DEBUG: PERCLOS calc: {closed_count}/{len(self.eye_state_history)} = {perclos:.4f}")
        return perclos
    
    def get_blinks_per_minute(self) -> float:
        """Calculate blinks per minute."""
        if len(self.blink_history) == 0:
            return 0.0
        fps = self.config['cameras']['fps']
        window_seconds = len(self.blink_history) / fps
        blink_count = sum(self.blink_history)
        bpm = (blink_count / window_seconds) * 60.0 if window_seconds > 0 else 0.0
        print(f"DEBUG: BPM calc: {blink_count} blinks in {window_seconds:.1f}s = {bpm:.1f} bpm")
        return bpm
    
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

