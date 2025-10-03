"""
Face Recognition System
Production-ready approach
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
import datetime
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from dataclasses import dataclass, field
import concurrent.futures

# Import ML libraries
from ultralytics import YOLO

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

from database import DatabaseManager
from config import Config

HEADLESS = os.environ.get("HEADLESS", "1") == "1"


@dataclass
class Detection:
    """Simple detection object"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    track_id: int
    person_id: Optional[str] = None
    recognition_confidence: float = 0.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    # Simple tracking
    frame_count: int = 0
    positive_recognitions: int = 0

    # Session tracking
    entry_logged: bool = False
    exit_logged: bool = False
    authorized_entry_time: Optional[datetime.datetime] = None

    # Status management
    status: str = "detecting"  # detecting, authorized, unauthorized
    duration: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    decision_made: bool = False
    alert_triggered: bool = False
    evidence_saved: bool = False

    # Location context
    entrance_zone: str = "unknown"


class SimpleFaceTracker:
    """Simple distance-based face tracker"""

    def __init__(self, max_distance: float = 100.0, max_frames_missing: int = 15):
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def update_tracks(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[
        Tuple[Tuple[int, int, int, int], float, int]]:
        """Update tracks with simple matching"""
        current_time = time.time()
        updated_detections = []

        # Mark all tracks as not updated
        for track_id in self.tracks:
            self.tracks[track_id]['updated'] = False

        for bbox, confidence in detections:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Find best matching track
            best_track_id = None
            best_distance = float('inf')

            for track_id, track_info in self.tracks.items():
                if track_info['updated']:
                    continue

                dx = center_x - track_info['center_x']
                dy = center_y - track_info['center_y']
                distance = np.sqrt(dx * dx + dy * dy)

                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id].update({
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_time,
                    'frames_missing': 0,
                    'updated': True
                })
                track_id = best_track_id
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_time,
                    'frames_missing': 0,
                    'updated': True
                }

            updated_detections.append((bbox, confidence, track_id))

        # Clean up old tracks
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if not track_info['updated']:
                track_info['frames_missing'] += 1
                if track_info['frames_missing'] > self.max_frames_missing:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return updated_detections


class FaceRecognitionSystem:
    """
    Simplified Face Recognition System
    Simple, fast, and reliable like commercial systems
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        self.logger = logging.getLogger(__name__)

        # System state
        self.is_running = False
        self.stop_event = threading.Event()

        # Pipeline queues
        self.frame_queue = queue.Queue(maxsize=3)
        self.recognition_queue = queue.Queue(maxsize=10)
        self.display_queue = queue.Queue(maxsize=2)

        # Camera and streaming
        self.capture = None
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Face detection and recognition
        self.face_detector = None
        self.known_faces = {}
        self.database = DatabaseManager()

        # Simple tracking
        self.face_tracker = SimpleFaceTracker(max_distance=100.0, max_frames_missing=15)
        self.active_tracks: Dict[int, Detection] = {}

        # SIMPLE THRESHOLDS (like commercial systems)
        self.recognition_threshold = 0.50  # Much lower, more realistic
        self.high_confidence_threshold = 0.75  # For immediate authorization
        self.min_recognitions_needed = 3  # Need 3 positive recognitions
        self.max_decision_time = 3.0  # Fast decisions (3 seconds max)

        # Security settings
        self._tts_lock = threading.Lock()
        self.alert_delay = 0.5  # Fast alerts
        self.audio_alerts_enabled = self.config.get('alerts.sound_enabled', True)
        self.save_evidence = self.config.get('alerts.save_evidence', True)
        self.last_alert_time = 0
        self.alert_cooldown = 3.0

        # Display settings
        self.window_name = "SIMPLE SURVEILLANCE SYSTEM"
        self.window_width = self.config.get('display.window_width', 1920)
        self.window_height = self.config.get('display.window_height', 1080)

        # Performance settings
        self.min_face_size = 60  # Smaller minimum size
        self.frame_skip_rate = self.config.get('camera.frame_skip', 2)
        self.frame_skip_counter = 0

        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'authorized_detections': 0,
            'unauthorized_detections': 0,
            'alerts_triggered': 0,
            'evidence_saved': 0,
            'detection_fps': 0.0,
            'recognition_fps': 0.0
        }

        # Threading
        self.thread_pool = None
        self.threads = []

        self._initialize_system()

    def _initialize_system(self):
        """Initialize simple surveillance system"""
        try:
            # Device detection
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    self.logger.info("🚀 M3 Max GPU acceleration enabled")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    self.logger.info("🚀 GPU acceleration enabled")
                else:
                    self.device = "cpu"
                    self.logger.info("💻 Using CPU processing")
            except ImportError:
                self.device = "cpu"

            # Load YOLO model
            model_paths = ["models/yolov11n-face.pt", "models/yolov11n.pt", "yolo11n.pt"]

            for model_path in model_paths:
                if Path(model_path).exists():
                    self.face_detector = YOLO(model_path)
                    self.logger.info(f"🎯 Model loaded: {model_path}")
                    break
            else:
                self.face_detector = YOLO("yolo11n.pt")
                self.logger.info("🎯 YOLO model downloaded")

            if self.device != "cpu":
                self.face_detector.to(self.device)

            # Load face database
            self._load_face_database()

            # Create directories
            for dir_path in ["logs/security_evidence", "logs/screenshots"]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

            # Thread pool
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=3, thread_name_prefix="Recognition"
            )

            self._print_configuration()
            self.logger.info("✅ Simple surveillance system initialized")

        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def _load_face_database(self):
        """Load face database"""
        self.known_faces = self.database.get_face_encodings()
        self.logger.info(f"👥 Database loaded: {len(self.known_faces)} authorized personnel")

        for person_id, encodings in self.known_faces.items():
            self.logger.info(f"   - {person_id}: {len(encodings)} encodings")

        if len(self.known_faces) == 0:
            self.logger.warning("⚠️ NO AUTHORIZED PERSONNEL IN DATABASE")
        else:
            print(f"✅ Simple database ready: {list(self.known_faces.keys())}")

    def _print_configuration(self):
        """Print simple system configuration"""
        print(f"\n{'=' * 60}")
        print("🔹 SIMPLE FACE RECOGNITION SYSTEM")
        print(f"{'=' * 60}")
        print(f"🎯 Recognition threshold: {self.recognition_threshold}")
        print(f"🎯 High confidence threshold: {self.high_confidence_threshold}")
        print(f"⏱️ Max decision time: {self.max_decision_time}s")
        print(f"📊 Known personnel: {len(self.known_faces)}")
        print(f"🎮 Device: {self.device}")
        print(f"✅ SIMPLE FEATURES:")
        print(f"   • Fast recognition (3 seconds)")
        print(f"   • Simple threshold-based decisions")
        print(f"   • Realistic confidence levels")
        print(f"   • Voice alerts and visual boxes")
        print(f"{'=' * 60}")

    def start_monitoring(self, rtsp_url: str, duration: Optional[int] = None) -> bool:
        """Start surveillance monitoring"""
        try:
            self.logger.info("🚀 STARTING SIMPLE SURVEILLANCE")

            if not self._connect_camera(rtsp_url):
                return False

            self._setup_display()
            self.is_running = True
            self._start_threads()
            self._announce_system_ready()

            if not HEADLESS:
                self._run_surveillance_interface(duration)
            else:
                start = time.time()
                while self.is_running and (duration is None or time.time() - start < duration):
                    time.sleep(0.1)

            self._stop_monitoring()
            return True

        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
            self._stop_monitoring()
            return False

    def _connect_camera(self, rtsp_url: str) -> bool:
        """Connect to RTSP camera"""
        try:
            self.logger.info(f"📹 Connecting to camera: {rtsp_url}")

            self.capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            # Optimize settings
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FPS, 25)
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

            # Test connection
            for attempt in range(3):
                ret, frame = self.capture.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    self.logger.info(f"✅ Camera connected: {w}x{h}")
                    return True
                time.sleep(1)

            self.logger.error("❌ Camera connection failed")
            return False

        except Exception as e:
            self.logger.error(f"Camera connection error: {e}")
            return False

    def _setup_display(self):
        """Setup display if not headless"""
        if HEADLESS:
            return
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
            self.logger.info(f"🖥️ Display setup: {self.window_width}x{self.window_height}")
        except Exception as e:
            self.logger.warning(f"Display setup error: {e}")

    def _start_threads(self):
        """Start simple pipeline threads"""
        self.logger.info("🧵 Starting simple pipeline...")

        threads = [
            threading.Thread(target=self._stream_capture_thread, name="StreamCapture", daemon=True),
            threading.Thread(target=self._face_detection_thread, name="FaceDetection", daemon=True),
            threading.Thread(target=self._recognition_thread, name="Recognition", daemon=True),
            threading.Thread(target=self._security_monitoring_thread, name="SecurityMonitoring", daemon=True),
        ]

        self.threads = threads
        for thread in threads:
            thread.start()
            self.logger.info(f"✅ Started {thread.name}")

        time.sleep(1)

    def _stream_capture_thread(self):
        """Simple stream capture thread"""
        self.logger.info("📹 Stream capture started")
        frame_count = 0

        while not self.stop_event.is_set():
            try:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                frame_count += 1
                self.stats['frames_captured'] = frame_count

                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Frame skipping for performance
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.frame_skip_rate != 0:
                    continue

                try:
                    self.frame_queue.put(frame.copy(), block=False)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame.copy(), block=False)
                    except queue.Empty:
                        pass

                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Stream capture error: {e}")
                time.sleep(0.1)

    def _face_detection_thread(self):
        """Simple face detection with tracking"""
        self.logger.info("🎯 Face detection started")
        fps_counter = deque(maxlen=30)

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                detections = self._detect_and_track_faces(frame)

                self.stats['frames_processed'] += 1
                self.stats['faces_detected'] += len(detections)

                fps_counter.append(time.time())
                if len(fps_counter) >= 2:
                    self.stats['detection_fps'] = len(fps_counter) / (fps_counter[-1] - fps_counter[0])

                # Send to recognition
                for detection in detections:
                    try:
                        self.recognition_queue.put((frame, detection), block=False)
                    except queue.Full:
                        pass

                # Send to display
                try:
                    self.display_queue.put((frame, detections), block=False)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put((frame, detections), block=False)
                    except queue.Empty:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Face detection error: {e}")

    def _detect_and_track_faces(self, frame: np.ndarray) -> List[Detection]:
        """Simple face detection and tracking"""
        detections = []
        current_time = datetime.datetime.now()

        try:
            # YOLO detection
            results = self.face_detector(
                frame,
                conf=0.5,  # Lower confidence for detection
                iou=0.4,
                device=self.device,
                verbose=False
            )

            if not results or results[0].boxes is None:
                return detections

            # Extract detections
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            # Filter detections
            yolo_detections = []
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                face_width = x2 - x1
                face_height = y2 - y1

                # Simple size filtering
                if (face_width >= self.min_face_size and
                        face_height >= self.min_face_size):
                    yolo_detections.append(((x1, y1, x2, y2), conf))

            # Update tracking
            tracked_detections = self.face_tracker.update_tracks(yolo_detections)

            # Create/update Detection objects
            for bbox, conf, track_id in tracked_detections:
                x1, y1, x2, y2 = bbox

                if track_id in self.active_tracks:
                    # Update existing detection
                    detection = self.active_tracks[track_id]
                    detection.bbox = (x1, y1, x2, y2)
                    detection.confidence = conf
                    detection.timestamp = current_time
                    detection.frame_count += 1
                    detection.last_seen = time.time()
                    detection.duration = time.time() - detection.first_seen
                    detection.entrance_zone = self._get_entrance_zone((x1, y1, x2, y2))
                else:
                    # Create new detection
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        track_id=track_id,
                        timestamp=current_time,
                        frame_count=1,
                        status="detecting",
                        entrance_zone=self._get_entrance_zone((x1, y1, x2, y2)),
                        first_seen=time.time(),
                        last_seen=time.time()
                    )
                    self.active_tracks[track_id] = detection
                    self.logger.info(f"🔵 NEW FACE: Track {track_id} in {detection.entrance_zone}")

                detections.append(detection)

        except Exception as e:
            self.logger.error(f"Detection and tracking error: {e}")

        self._cleanup_old_tracks()
        return detections

    def _recognition_thread(self):
        """Simple recognition thread"""
        self.logger.info("🧠 Recognition started")
        fps_counter = deque(maxlen=30)

        while not self.stop_event.is_set():
            try:
                frame, detection = self.recognition_queue.get(timeout=0.5)

                # Simple face recognition
                person_id, confidence = self._recognize_face_simple(frame, detection.bbox)

                # Update detection
                if person_id and confidence >= self.recognition_threshold:
                    detection.positive_recognitions += 1
                    detection.person_id = person_id
                    detection.recognition_confidence = confidence

                    self.logger.info(f"✅ RECOGNIZED: {person_id} = {confidence:.3f}")

                    # High confidence = immediate authorization
                    if confidence >= self.high_confidence_threshold:
                        detection.status = "authorized"
                        detection.decision_made = True
                        self.stats['authorized_detections'] += 1

                        if not detection.entry_logged:
                            detection.entry_logged = True
                            detection.authorized_entry_time = datetime.datetime.now()
                            self._log_authorized_entry(detection)

                # Simple status update
                self._update_status_simple(detection)

                self.stats['faces_recognized'] += 1
                fps_counter.append(time.time())
                if len(fps_counter) >= 2:
                    self.stats['recognition_fps'] = len(fps_counter) / (fps_counter[-1] - fps_counter[0])

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Recognition error: {e}")

    def _recognize_face_simple(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], float]:
        """Simple face recognition - like commercial systems"""
        try:
            x1, y1, x2, y2 = bbox

            # Simple padding
            padding = 20
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            y2_pad = min(frame.shape[0], y2 + padding)

            face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if face_crop.size == 0:
                return None, 0.0

            # Simple preprocessing
            face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LANCZOS4)

            # Create embedding
            representation = DeepFace.represent(
                img_path=face_resized,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )

            if not representation or len(representation) == 0:
                return None, 0.0

            test_embedding = np.array(representation[0]["embedding"])
            test_embedding = test_embedding / (np.linalg.norm(test_embedding) + 1e-8)

            # Simple similarity calculation
            best_person = None
            best_similarity = 0.0

            for person_id, stored_encodings in self.known_faces.items():
                for stored_encoding in stored_encodings:
                    try:
                        stored_array = np.array(stored_encoding)
                        stored_norm = stored_array / (np.linalg.norm(stored_array) + 1e-8)

                        # Cosine similarity
                        similarity = np.dot(test_embedding, stored_norm)
                        similarity = max(0.0, min(1.0, similarity))

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_person = person_id

                    except Exception:
                        continue

            return best_person, best_similarity

        except Exception as e:
            self.logger.error(f"Recognition error: {e}")
            return None, 0.0

    def _update_status_simple(self, detection: Detection):
        """Simple status update logic"""
        current_time = time.time()
        elapsed = current_time - detection.first_seen

        # Fast decisions
        if detection.status == "detecting":
            # Need enough positive recognitions OR high confidence
            if (detection.positive_recognitions >= self.min_recognitions_needed or
                    detection.recognition_confidence >= self.high_confidence_threshold):
                detection.status = "authorized"
                detection.decision_made = True
                self.stats['authorized_detections'] += 1

                if not detection.entry_logged:
                    detection.entry_logged = True
                    detection.authorized_entry_time = datetime.datetime.now()
                    self._log_authorized_entry(detection)

                self.logger.info(f"🟢 AUTHORIZED: {detection.person_id}")

            # Fast rejection for unknown faces
            elif elapsed > self.max_decision_time and detection.positive_recognitions == 0:
                detection.status = "unauthorized"
                detection.decision_made = True
                self.stats['unauthorized_detections'] += 1
                self.logger.warning(f"🔴 UNAUTHORIZED: Track {detection.track_id}")

    def _get_entrance_zone(self, bbox: Tuple[int, int, int, int]) -> str:
        """Get entrance zone"""
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if self.current_frame is not None:
                frame_width = self.current_frame.shape[1]
                frame_height = self.current_frame.shape[0]

                if center_x < frame_width // 3:
                    horizontal = "LEFT"
                elif center_x > 2 * frame_width // 3:
                    horizontal = "RIGHT"
                else:
                    horizontal = "CENTER"

                vertical = "NEAR" if center_y > frame_height // 2 else "FAR"
                return f"{horizontal}_{vertical}"

            return "UNKNOWN"

        except Exception:
            return "UNKNOWN"

    def _security_monitoring_thread(self):
        """Simple security monitoring"""
        self.logger.info("🚨 Security monitoring started")

        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                for detection in list(self.active_tracks.values()):
                    detection.duration = current_time - detection.first_seen

                    # Trigger alerts for unauthorized faces
                    if (detection.status == "unauthorized" and
                            not detection.alert_triggered and
                            detection.duration >= self.alert_delay):
                        detection.alert_triggered = True
                        self._execute_security_protocol(detection)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")

    def _execute_security_protocol(self, detection: Detection):
        """Execute simple security protocol"""
        try:
            current_time = time.time()

            if current_time - self.last_alert_time < self.alert_cooldown:
                return

            self.last_alert_time = current_time
            self.stats['alerts_triggered'] += 1

            self.logger.critical("🚨 SECURITY ALERT")

            # Save evidence
            if self.save_evidence and not detection.evidence_saved:
                self._save_security_evidence(detection)
                detection.evidence_saved = True

            # Audio alert
            if self.audio_alerts_enabled:
                self._trigger_audio_alert(detection)

            # Log security event
            self._log_security_event(detection)

        except Exception as e:
            self.logger.error(f"Security protocol error: {e}")

    def _save_security_evidence(self, detection: Detection):
        """Save security evidence"""
        try:
            if self.current_frame is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/security_evidence/ALERT_track_{detection.track_id}_{timestamp}.jpg"

                Path(filename).parent.mkdir(parents=True, exist_ok=True)

                # Create evidence frame
                evidence_frame = self.current_frame.copy()
                x1, y1, x2, y2 = detection.bbox

                # Draw alert box
                cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 8)
                cv2.putText(evidence_frame, f"SECURITY ALERT - Track {detection.track_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Add timestamp
                time_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(evidence_frame, time_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imwrite(filename, evidence_frame)
                self.stats['evidence_saved'] += 1
                self.logger.critical(f"📸 EVIDENCE: {filename}")

        except Exception as e:
            self.logger.error(f"Evidence save error: {e}")

    def _trigger_audio_alert(self, detection: Detection):
        """Trigger French audio alert"""
        try:
            message = "Alerte sécurité ! Personne non autorisée détectée."
            self._speak_french_alert(message)
        except Exception as e:
            self.logger.error(f"Audio alert error: {e}")

    def _speak_french_alert(self, message: str):
        """French TTS for alerts"""
        import subprocess, platform

        with self._tts_lock:
            try:
                if platform.system() == 'Darwin':
                    subprocess.run(['say', '-v', 'Amélie', message],
                                   capture_output=True, timeout=5, check=False)
                elif platform.system() == 'Windows':
                    subprocess.run(['powershell', '-Command',
                                    f'Add-Type -AssemblyName System.Speech; '
                                    f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                                    f'$synth.Speak("{message}")'],
                                   capture_output=True, timeout=10, check=False)
                else:
                    subprocess.run(['espeak', '-v', 'fr', '-s', '150', message],
                                   capture_output=True, timeout=5, check=False)
            except Exception as e:
                self.logger.error(f"TTS error: {e}")

    def _log_event(self, event: Dict):
        """Log event to file"""
        try:
            log_file = Path("logs/logging_access.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"Event logging error: {e}")

    def _log_authorized_entry(self, detection: Detection):
        """Log authorized entry"""
        try:
            event = {
                'timestamp': detection.authorized_entry_time.isoformat(),
                'event_type': 'AUTHORIZED_ENTRY',
                'person_id': detection.person_id,
                'person_name_french': detection.person_id.replace('_', ' ').title(),
                'confidence': detection.recognition_confidence,
                'track_id': f"track_{detection.track_id}",
                'entrance_zone': detection.entrance_zone,
                'location': 'entrance_gate',
                'message_french': f"{detection.person_id.replace('_', ' ').title()} est entré dans la caserne",
                'recognition_method': 'simple_threshold'
            }
            self._log_event(event)

            self.logger.info(f"🚪➡️ ENTRÉE: {detection.person_id.replace('_', ' ').title()}")

        except Exception as e:
            self.logger.error(f"Entry logging error: {e}")

    def _log_security_event(self, detection: Detection):
        """Log security event"""
        try:
            event = {
                'timestamp': detection.timestamp.isoformat(),
                'event_type': 'SECURITY_ALERT',
                'person_id': 'UNAUTHORIZED',
                'track_id': f"track_{detection.track_id}",
                'duration': detection.duration,
                'entrance_zone': detection.entrance_zone,
                'location': 'entrance_gate',
                'alert_level': 'CRITICAL',
                'recognition_method': 'simple_threshold'
            }
            self._log_event(event)
        except Exception as e:
            self.logger.error(f"Security logging error: {e}")

    def _cleanup_old_tracks(self):
        """Clean up old tracks"""
        current_time = time.time()
        expired_tracks = []

        for track_id, detection in self.active_tracks.items():
            if current_time - detection.last_seen > 5.0:  # 5 second timeout
                expired_tracks.append(track_id)

        for track_id in expired_tracks:
            detection = self.active_tracks[track_id]

            # Log exit for authorized personnel
            if (detection.status == "authorized" and
                    detection.entry_logged and
                    not detection.exit_logged and
                    detection.person_id):
                self._log_authorized_exit(detection)

            del self.active_tracks[track_id]

    def _log_authorized_exit(self, detection: Detection):
        """Log authorized exit"""
        try:
            detection.exit_logged = True
            exit_time = datetime.datetime.now()

            if detection.authorized_entry_time:
                visit_duration = (exit_time - detection.authorized_entry_time).total_seconds()
            else:
                visit_duration = detection.duration

            event = {
                'timestamp': exit_time.isoformat(),
                'event_type': 'AUTHORIZED_EXIT',
                'person_id': detection.person_id,
                'person_name_french': detection.person_id.replace('_', ' ').title(),
                'track_id': f"track_{detection.track_id}",
                'visit_duration_seconds': visit_duration,
                'location': 'entrance_gate',
                'message_french': f"{detection.person_id.replace('_', ' ').title()} a quitté la caserne"
            }
            self._log_event(event)

            self.logger.info(f"🚪⬅️ SORTIE: {detection.person_id.replace('_', ' ').title()}")

        except Exception as e:
            self.logger.error(f"Exit logging error: {e}")

    def _announce_system_ready(self):
        """Announce system ready"""
        try:
            self._speak_french_alert("Système de surveillance activé.")
            self.logger.info("✅ SIMPLE SYSTEM READY")
        except Exception as e:
            self.logger.error(f"Announcement error: {e}")

    def _run_surveillance_interface(self, duration: Optional[int]):
        """Run simple surveillance interface"""
        if HEADLESS:
            return

        start_time = time.time()
        print(f"\n{'=' * 60}")
        print("🔹 SIMPLE SURVEILLANCE SYSTEM ACTIVE")
        print("Fast • Simple • Reliable")
        print("Controls: [Q]uit [S]creenshot [D]ebug")
        print(f"{'=' * 60}")

        while self.is_running:
            try:
                self._update_surveillance_display()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    self._save_manual_screenshot()
                elif key == ord('d'):
                    self._print_debug_info()

                if duration and (time.time() - start_time) >= duration:
                    break

            except KeyboardInterrupt:
                break

    def _update_surveillance_display(self):
        """Update simple surveillance display"""
        try:
            try:
                frame, detections = self.display_queue.get_nowait()
            except queue.Empty:
                return

            display_frame = frame.copy()
            self._draw_simple_detections(display_frame, detections)
            self._draw_simple_interface(display_frame)

            cv2.imshow(self.window_name, display_frame)

        except Exception as e:
            self.logger.error(f"Display error: {e}")

    def _draw_simple_detections(self, frame: np.ndarray, detections: List[Detection]):
        """Draw simple detection boxes"""
        try:
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox

                # Simple color coding
                if detection.status == "detecting":
                    color = (255, 200, 0)  # Blue
                    status_text = f"🔵 DETECTING #{detection.track_id}"
                    thickness = 4
                elif detection.status == "authorized":
                    color = (0, 255, 0)  # Green
                    status_text = f"🟢 {detection.person_id.replace('_', ' ').upper()}"
                    thickness = 6
                elif detection.status == "unauthorized":
                    color = (0, 0, 255)  # Red
                    status_text = f"🔴 ALERT #{detection.track_id}"
                    thickness = 8
                else:
                    color = (128, 128, 128)
                    status_text = f"? #{detection.track_id}"
                    thickness = 3

                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Draw status label
                label_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 40), (x1 + label_size[0] + 15, y1), color, -1)
                cv2.putText(frame, status_text, (x1 + 7, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Draw confidence if available
                if detection.recognition_confidence > 0:
                    conf_text = f"Conf: {detection.recognition_confidence:.2f}"
                    cv2.putText(frame, conf_text, (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception as e:
            self.logger.error(f"Detection drawing error: {e}")

    def _draw_simple_interface(self, frame: np.ndarray):
        """Draw simple interface overlay"""
        try:
            h, w = frame.shape[:2]

            # Count statuses
            status_counts = {}
            for detection in self.active_tracks.values():
                status_counts[detection.status] = status_counts.get(detection.status, 0) + 1

            # Header
            cv2.rectangle(frame, (0, 0), (w, 100), (30, 30, 30), -1)
            cv2.putText(frame, "🔹 SIMPLE SURVEILLANCE ACTIVE", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # Status line
            status_line = f"Detecting: {status_counts.get('detecting', 0)} | " \
                          f"Authorized: {status_counts.get('authorized', 0)} | " \
                          f"Alerts: {status_counts.get('unauthorized', 0)}"
            cv2.putText(frame, status_line, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Statistics panel
            stats_text = [
                f"Faces: {self.stats['faces_detected']}",
                f"Authorized: {self.stats['authorized_detections']}",
                f"Alerts: {self.stats['alerts_triggered']}",
                f"FPS: {self.stats['detection_fps']:.1f}",
                f"Threshold: {self.recognition_threshold}",
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (w - 250, 120 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        except Exception as e:
            self.logger.error(f"Interface drawing error: {e}")

    def _save_manual_screenshot(self):
        """Save manual screenshot"""
        try:
            if self.current_frame is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/screenshots/screenshot_{timestamp}.jpg"
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filename, self.current_frame)
                print(f"📸 Screenshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Screenshot error: {e}")

    def _print_debug_info(self):
        """Print simple debug information"""
        print(f"\n{'=' * 50}")
        print("🔹 SIMPLE DEBUG INFO")
        print(f"{'=' * 50}")
        print(f"Active tracks: {len(self.active_tracks)}")
        print(f"Known faces: {len(self.known_faces)}")
        print(f"Recognition threshold: {self.recognition_threshold}")

        if self.active_tracks:
            print("\nActive detections:")
            for track_id, detection in self.active_tracks.items():
                status_emoji = {"detecting": "🔵", "authorized": "🟢", "unauthorized": "🔴"}.get(detection.status, "⚪")
                print(f"  {status_emoji} Track {track_id}: {detection.status}")
                print(f"    Duration: {detection.duration:.1f}s")
                print(f"    Positive recognitions: {detection.positive_recognitions}")
                if detection.person_id:
                    print(f"    Person: {detection.person_id} ({detection.recognition_confidence:.3f})")

        print(f"{'=' * 50}")

    def _stop_monitoring(self):
        """Stop simple monitoring system"""
        try:
            self.logger.info("🛑 Stopping simple surveillance...")
            self.is_running = False
            self.stop_event.set()

            if self.capture:
                self.capture.release()

            if not HEADLESS:
                cv2.destroyAllWindows()

            if self.thread_pool:
                self.thread_pool.shutdown(wait=True, timeout=2)

            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=1)

            # Simple report
            print(f"\n{'=' * 50}")
            print("📊 SIMPLE SURVEILLANCE REPORT")
            print(f"{'=' * 50}")
            print(f"✅ Authorized: {self.stats['authorized_detections']}")
            print(f"🚨 Alerts: {self.stats['alerts_triggered']}")
            print(f"👤 Faces detected: {self.stats['faces_detected']}")
            print(f"🧠 Recognitions: {self.stats['faces_recognized']}")
            print(f"📸 Evidence saved: {self.stats['evidence_saved']}")
            print(f"🎯 Simple & Fast Recognition")
            print(f"{'=' * 50}")

            self.logger.info("✅ Simple surveillance stopped")

        except Exception as e:
            self.logger.error(f"Stop error: {e}")


# Compatibility aliases
CompleteFaceRecognitionSystem = FaceRecognitionSystem
EnhancedSurveillanceSystem = FaceRecognitionSystem