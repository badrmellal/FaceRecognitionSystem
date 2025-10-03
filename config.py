"""
Enhanced Configuration Management
Optimized for accurate face recognition
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict


class Config:
    """Enhanced Configuration Manager with better recognition settings"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)

        # ENHANCED DEFAULT CONFIGURATION FOR BETTER ACCURACY
        self.default_config = {
            "camera": {
                "frame_skip": 1,  # Reduced for better tracking
                "buffer_size": 1,  # Minimal buffer
                "target_fps": 25,  # Good FPS for 5MP
                "timeout_ms": 5000,
                "force_substream": True
            },

            "face_detection": {
                "confidence_threshold": 0.6,  # Balanced threshold
                "min_face_size": 80,  # Minimum size for good recognition
                "nms_threshold": 0.4
            },

            "face_recognition": {
                "model": "ArcFace",
                "similarity_threshold": 0.72,  # Optimized threshold
                "high_confidence_threshold": 0.85,  # High confidence threshold
                "min_confidence_gap": 0.12,  # Minimum gap between different people
                "target_size": [112, 112],  # ArcFace standard
                "quality_boost_threshold": 0.8,  # Quality boost for excellent faces
                "excellent_quality_boost": 0.05  # Boost for excellent quality faces
            },

            "quality_thresholds": {
                "min_overall_score": 0.3,  # Minimum quality to process
                "good_quality_threshold": 0.5,  # Good quality threshold
                "excellent_overall": 0.8,  # Excellent quality threshold
                "min_sharpness": 0.2,  # Minimum sharpness
                "min_brightness": 0.3  # Minimum brightness
            },

            "strict_security": {
                "require_identity_consistency": True,  # Require consistent identity
                "require_consecutive_matches": 3,  # Consecutive matches needed
                "min_positive_frames": 4,  # Positive frames needed for authorization
                "min_excellent_quality_frames": 2,  # Excellent quality frames needed
                "max_negative_frames": 3,  # Max negative frames before rejection
                "min_analyzing_frames": 8,  # Minimum frames in analyzing phase
                "max_analysis_time": 10.0  # Maximum analysis time
            },

            "tracking": {
                "max_distance": 80.0,  # Maximum tracking distance
                "max_frames_missing": 20,  # Maximum frames missing before track deletion
                "min_positive_frames": 4,  # Minimum positive frames
                "max_negative_frames": 3,  # Maximum negative frames
                "min_quality_frames": 2,  # Minimum quality frames
                "quality_threshold": 0.3  # Quality threshold
            },

            "alerts": {
                "delay_seconds": 1.5,  # Alert delay
                "sound_enabled": True,
                "french_voice": "Amélie",
                "save_evidence": True,
                "rate_limit": 5.0,  # Seconds between alerts
                "max_evidence_files": 3  # Max evidence files per alert
            },

            "display": {
                "window_width": 2688,  # Large display
                "window_height": 1520,
                "show_confidence": True,
                "show_fps": True,
                "show_quality_scores": True,
                "fullscreen": False,
                "font_scale": 0.7
            },

            "debug": {
                "verbose_recognition": True,  # Verbose recognition logging
                "save_failed_recognitions": True,  # Save failed recognitions
                "log_similarity_scores": True,  # Log all similarity scores
                "save_quality_analysis": True,  # Save quality analysis
                "detailed_tracking": True  # Detailed tracking info
            },

            "logging": {
                "level": "INFO",
                "file": "enhanced_face_recognition.log",
                "max_size_mb": 20,
                "backup_count": 5
            },

            "performance": {
                "processing_threads": 4,  # Processing threads
                "max_tracked_faces": 8,  # Maximum tracked faces
                "cleanup_interval": 5.0,  # Cleanup interval
                "cache_timeout": 2.0,  # Recognition cache timeout
                "max_cache_size": 100  # Maximum cache size
            },

            "database": {
                "backup_on_changes": True,  # Backup on changes
                "max_backups": 10,  # Maximum backups
                "verify_encodings": True,  # Verify encodings on load
                "encoding_quality_check": True  # Check encoding quality
            }
        }

        self.config_data = self._load_or_create_config()
        self._setup_logging()

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create with enhanced defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # Merge with defaults to ensure new features are included
                merged_config = self._merge_configs(self.default_config, loaded_config)

                # Save merged config
                self._save_config(merged_config)
                return merged_config
            else:
                # Create new enhanced config
                self._save_config(self.default_config)
                self.logger.info(f"Created enhanced config: {self.config_path}")
                return self.default_config.copy()

        except Exception as e:
            self.logger.error(f"Config load error: {e}, using defaults")
            return self.default_config.copy()

    def _merge_configs(self, default: Dict, current: Dict) -> Dict:
        """Recursively merge configurations"""
        result = default.copy()

        for key, value in current.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Config save error: {e}")

    def _setup_logging(self):
        """Setup enhanced logging"""
        try:
            log_level = getattr(logging, self.get('logging.level', 'INFO').upper())
            log_file = self.get('logging.file', 'enhanced_face_recognition.log')

            # Create logs directory
            Path('logs').mkdir(exist_ok=True)

            # Setup logging with rotation
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(f'logs/{log_file}', encoding='utf-8')
                ]
            )

        except Exception as e:
            print(f"Logging setup error: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            value = self.config_data

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default

            return value

        except Exception as e:
            self.logger.error(f"Config get error for '{key_path}': {e}")
            return default

    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config_ref = self.config_data

            # Navigate to parent
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]

            # Set value
            config_ref[keys[-1]] = value

            # Save to file
            self._save_config(self.config_data)

            self.logger.info(f"Config updated: {key_path} = {value}")
            return True

        except Exception as e:
            self.logger.error(f"Config set error for '{key_path}': {e}")
            return False

    def get_enhanced_recognition_config(self) -> Dict[str, Any]:
        """Get enhanced recognition configuration"""
        base_config = self.get('face_recognition', {})

        enhanced_config = {
            "similarity_threshold": 0.72,
            "high_confidence_threshold": 0.85,
            "min_confidence_gap": 0.12,
            "target_size": [112, 112],
            "quality_boost_threshold": 0.8,
            "excellent_quality_boost": 0.05,
            "model": "ArcFace",
            **base_config
        }

        return enhanced_config

    def get_optimized_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration optimized for 5MP Imou"""
        base_config = self.get('camera', {})

        optimized = {
            "frame_skip": 1,  # Better tracking
            "buffer_size": 1,  # Minimal buffer
            "target_fps": 25,  # Good for 5MP
            "force_substream": True,
            "timeout_ms": 5000,
            **base_config
        }

        return optimized

    def get_strict_security_config(self) -> Dict[str, Any]:
        """Get strict security configuration"""
        return {
            "require_identity_consistency": True,
            "require_consecutive_matches": 3,
            "min_positive_frames": 4,
            "min_excellent_quality_frames": 2,
            "max_negative_frames": 3,
            "min_analyzing_frames": 8,
            "max_analysis_time": 10.0,
            **self.get('strict_security', {})
        }

    def get_quality_thresholds(self) -> Dict[str, Any]:
        """Get quality assessment thresholds"""
        return {
            "min_overall_score": 0.3,
            "good_quality_threshold": 0.5,
            "excellent_overall": 0.8,
            "min_sharpness": 0.2,
            "min_brightness": 0.3,
            **self.get('quality_thresholds', {})
        }

    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return {
            "window_width": 1920,
            "window_height": 1080,
            "show_confidence": True,
            "show_fps": True,
            "show_quality_scores": True,
            "fullscreen": False,
            "font_scale": 0.7,
            **self.get('display', {})
        }

    def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration"""
        return {
            "delay_seconds": 1.5,
            "sound_enabled": True,
            "french_voice": "Amélie",
            "save_evidence": True,
            "rate_limit": 5.0,
            "max_evidence_files": 3,
            **self.get('alerts', {})
        }

    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration"""
        return {
            "verbose_recognition": True,
            "save_failed_recognitions": True,
            "log_similarity_scores": True,
            "save_quality_analysis": True,
            "detailed_tracking": True,
            **self.get('debug', {})
        }

    def optimize_for_accuracy(self):
        """Optimize configuration for maximum accuracy"""
        try:
            # Recognition accuracy optimizations
            self.set('face_recognition.similarity_threshold', 0.75)
            self.set('face_recognition.min_confidence_gap', 0.15)
            self.set('face_recognition.high_confidence_threshold', 0.88)

            # Quality thresholds
            self.set('quality_thresholds.min_overall_score', 0.4)
            self.set('quality_thresholds.excellent_overall', 0.85)

            # Strict security
            self.set('strict_security.min_positive_frames', 5)
            self.set('strict_security.min_excellent_quality_frames', 3)
            self.set('strict_security.max_negative_frames', 2)

            # Debug for troubleshooting
            self.set('debug.verbose_recognition', True)
            self.set('debug.log_similarity_scores', True)

            self.logger.info("✅ Configuration optimized for maximum accuracy")
            return True

        except Exception as e:
            self.logger.error(f"Accuracy optimization error: {e}")
            return False

    def optimize_for_performance(self):
        """Optimize configuration for better performance"""
        try:
            # Performance optimizations
            self.set('camera.frame_skip', 3)
            self.set('face_detection.min_face_size', 60)
            self.set('quality_thresholds.min_overall_score', 0.25)

            # Less strict requirements
            self.set('strict_security.min_positive_frames', 3)
            self.set('strict_security.min_analyzing_frames', 6)

            # Reduced debugging
            self.set('debug.verbose_recognition', False)
            self.set('debug.save_failed_recognitions', False)

            self.logger.info("✅ Configuration optimized for performance")
            return True

        except Exception as e:
            self.logger.error(f"Performance optimization error: {e}")
            return False

    def print_current_settings(self):
        """Print current important settings"""
        print(f"\n{'=' * 80}")
        print("🔧 CURRENT ENHANCED CONFIGURATION")
        print(f"{'=' * 80}")

        # Recognition settings
        print("🧠 RECOGNITION SETTINGS:")
        print(f"   Similarity threshold: {self.get('face_recognition.similarity_threshold')}")
        print(f"   Min confidence gap: {self.get('face_recognition.min_confidence_gap')}")
        print(f"   High confidence threshold: {self.get('face_recognition.high_confidence_threshold')}")

        # Quality settings
        print("\n✨ QUALITY SETTINGS:")
        print(f"   Min overall score: {self.get('quality_thresholds.min_overall_score')}")
        print(f"   Excellent threshold: {self.get('quality_thresholds.excellent_overall')}")
        print(f"   Min sharpness: {self.get('quality_thresholds.min_sharpness')}")

        # Security settings
        print("\n🔒 SECURITY SETTINGS:")
        print(f"   Min positive frames: {self.get('strict_security.min_positive_frames')}")
        print(f"   Max negative frames: {self.get('strict_security.max_negative_frames')}")
        print(f"   Identity consistency: {self.get('strict_security.require_identity_consistency')}")

        # Camera settings
        print("\n📹 CAMERA SETTINGS:")
        print(f"   Frame skip: {self.get('camera.frame_skip')}")
        print(f"   Target FPS: {self.get('camera.target_fps')}")
        print(f"   Min face size: {self.get('face_detection.min_face_size')}")

        # Debug settings
        print("\n🔍 DEBUG SETTINGS:")
        print(f"   Verbose recognition: {self.get('debug.verbose_recognition')}")
        print(f"   Log similarity scores: {self.get('debug.log_similarity_scores')}")
        print(f"   Detailed tracking: {self.get('debug.detailed_tracking')}")

        print(f"{'=' * 80}")

    def validate_config(self) -> bool:
        """Validate configuration for potential issues"""
        try:
            issues = []

            # Check similarity threshold
            sim_thresh = self.get('face_recognition.similarity_threshold')
            if sim_thresh < 0.5 or sim_thresh > 0.9:
                issues.append(f"Similarity threshold {sim_thresh} may be too extreme")

            # Check confidence gap
            conf_gap = self.get('face_recognition.min_confidence_gap')
            if conf_gap < 0.05:
                issues.append(f"Confidence gap {conf_gap} may be too small for multi-person recognition")

            # Check frame requirements
            min_pos = self.get('strict_security.min_positive_frames')
            max_neg = self.get('strict_security.max_negative_frames')
            if min_pos < 2:
                issues.append(f"Min positive frames {min_pos} may be too low")
            if max_neg > 5:
                issues.append(f"Max negative frames {max_neg} may be too high")

            if issues:
                self.logger.warning("⚠️ Configuration validation issues:")
                for issue in issues:
                    self.logger.warning(f"   - {issue}")
                return False
            else:
                self.logger.info("✅ Configuration validation passed")
                return True

        except Exception as e:
            self.logger.error(f"Config validation error: {e}")
            return False

    def reset_to_defaults(self):
        """Reset configuration to optimized defaults"""
        try:
            self.config_data = self.default_config.copy()
            self._save_config(self.config_data)
            self.logger.info("✅ Configuration reset to enhanced defaults")
            return True
        except Exception as e:
            self.logger.error(f"Config reset error: {e}")
            return False