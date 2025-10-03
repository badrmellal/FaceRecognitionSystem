"""
Enhanced Database Management for Face Recognition
Improved encoding storage and validation to prevent misidentification
"""

import os
import json
import pickle
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import datetime
import hashlib

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class EnhancedDatabaseManager:
    """Enhanced Database Manager with improved encoding validation and storage"""

    def __init__(self, db_path: str = "known_faces"):
        self.db_path = Path(db_path)
        self.encodings_file = self.db_path / "enhanced_faces.pkl"
        self.metadata_file = self.db_path / "enhanced_metadata.json"
        self.validation_file = self.db_path / "encoding_validation.json"

        self.logger = logging.getLogger(__name__)

        # Initialize directory
        self.db_path.mkdir(exist_ok=True)

        # Enhanced storage
        self.known_faces = {}
        self.metadata = {}
        self.encoding_validation = {}

        # Quality settings
        self.min_encoding_quality = 0.5
        self.max_encodings_per_person = 15
        self.target_encodings_per_person = 8

        self._load_database()
        self.logger.info(f"Enhanced database initialized: {len(self.known_faces)} persons")

    def _load_database(self):
        """Load enhanced database with validation"""
        try:
            # Load encodings
            if self.encodings_file.exists():
                with open(self.encodings_file, 'rb') as f:
                    self.known_faces = pickle.load(f)

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)

            # Load validation data
            if self.validation_file.exists():
                with open(self.validation_file, 'r', encoding='utf-8') as f:
                    self.encoding_validation = json.load(f)

            # Validate loaded data
            self._validate_database_integrity()

            self.logger.info(f"Loaded {len(self.known_faces)} known faces")

        except Exception as e:
            self.logger.error(f"Database load error: {e}")
            self.known_faces = {}
            self.metadata = {}
            self.encoding_validation = {}

    def _validate_database_integrity(self):
        """Validate database integrity and fix issues"""
        try:
            issues_found = 0

            # Check for encoding consistency
            for person_id, encodings in list(self.known_faces.items()):
                if not encodings:
                    self.logger.warning(f"Person {person_id} has no encodings, removing")
                    del self.known_faces[person_id]
                    issues_found += 1
                    continue

                # Validate encoding shapes and quality
                valid_encodings = []
                for i, encoding in enumerate(encodings):
                    try:
                        enc_array = np.array(encoding)
                        if enc_array.shape == (512,):  # ArcFace encoding size
                            # Check if encoding is normalized
                            norm = np.linalg.norm(enc_array)
                            if 0.8 <= norm <= 1.2:  # Reasonable norm range
                                # Normalize if needed
                                normalized_enc = enc_array / (norm + 1e-8)
                                valid_encodings.append(normalized_enc.tolist())
                            else:
                                self.logger.warning(f"Invalid norm for {person_id} encoding {i}: {norm}")
                                issues_found += 1
                        else:
                            self.logger.warning(f"Invalid shape for {person_id} encoding {i}: {enc_array.shape}")
                            issues_found += 1
                    except Exception as e:
                        self.logger.error(f"Error validating {person_id} encoding {i}: {e}")
                        issues_found += 1

                if len(valid_encodings) != len(encodings):
                    self.logger.info(f"Fixed {person_id}: {len(encodings)} -> {len(valid_encodings)} encodings")
                    self.known_faces[person_id] = valid_encodings

                if len(valid_encodings) == 0:
                    self.logger.warning(f"No valid encodings for {person_id}, removing")
                    del self.known_faces[person_id]
                    issues_found += 1

            if issues_found > 0:
                self.logger.warning(f"Fixed {issues_found} database integrity issues")
                self.save_database()

        except Exception as e:
            self.logger.error(f"Database validation error: {e}")

    def save_database(self) -> bool:
        """Save enhanced database with backup"""
        try:
            # Create backup
            self._create_backup()

            # Save encodings
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.known_faces, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)

            # Save validation data
            with open(self.validation_file, 'w', encoding='utf-8') as f:
                json.dump(self.encoding_validation, f, indent=2, ensure_ascii=False)

            self.logger.info("Enhanced database saved successfully")
            return True

        except Exception as e:
            self.logger.error(f"Database save error: {e}")
            return False

    def add_person(self, person_id: str, image_paths: List[str],
                   metadata: Optional[Dict] = None) -> bool:
        """Add person with enhanced encoding validation"""
        try:
            if not DEEPFACE_AVAILABLE:
                self.logger.error("DeepFace not available")
                return False

            if person_id in self.known_faces:
                self.logger.warning(f"Person {person_id} already exists, updating...")

            # Process images and create high-quality encodings
            valid_encodings = []
            encoding_metadata = []
            successful_images = 0

            self.logger.info(f"Processing {len(image_paths)} images for {person_id}...")

            for i, image_path in enumerate(image_paths):
                if not Path(image_path).exists():
                    self.logger.warning(f"Image not found: {image_path}")
                    continue

                try:
                    # Load and validate image
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.warning(f"Cannot load image: {image_path}")
                        continue

                    # Check image quality
                    quality_score = self._assess_image_quality(image)
                    if quality_score < self.min_encoding_quality:
                        self.logger.warning(f"Low quality image skipped: {image_path} (score: {quality_score:.3f})")
                        continue

                    # Create encoding with enhanced preprocessing
                    encoding = self._create_enhanced_encoding(image, image_path)
                    if encoding is not None:
                        valid_encodings.append(encoding)
                        encoding_metadata.append({
                            "source_image": str(image_path),
                            "quality_score": quality_score,
                            "created_at": datetime.datetime.now().isoformat(),
                            "encoding_method": "ArcFace_enhanced",
                            "preprocessing_version": "2.0"
                        })
                        successful_images += 1
                        self.logger.info(f"✅ Encoded {Path(image_path).name} (quality: {quality_score:.3f})")
                    else:
                        self.logger.warning(f"❌ Failed to encode {image_path}")

                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    continue

            if successful_images == 0:
                self.logger.error(f"No valid encodings created for {person_id}")
                return False

            # Quality control - ensure we have good encodings
            if successful_images < 3:
                self.logger.warning(f"Only {successful_images} encodings for {person_id} (recommend 5+)")

            # Remove outlier encodings if too many
            if len(valid_encodings) > self.max_encodings_per_person:
                valid_encodings = self._select_best_encodings(valid_encodings, encoding_metadata)

            # Store encodings
            self.known_faces[person_id] = valid_encodings

            # Store enhanced metadata
            if metadata is None:
                metadata = {}

            enhanced_metadata = {
                "person_id": person_id,
                "num_encodings": len(valid_encodings),
                "num_source_images": len(image_paths),
                "successful_images": successful_images,
                "added_date": datetime.datetime.now().isoformat(),
                "model": "ArcFace_enhanced",
                "preprocessing_version": "2.0",
                "encoding_quality": {
                    "min_quality": min(meta["quality_score"] for meta in encoding_metadata),
                    "max_quality": max(meta["quality_score"] for meta in encoding_metadata),
                    "avg_quality": np.mean([meta["quality_score"] for meta in encoding_metadata])
                },
                "validation_hash": self._compute_validation_hash(valid_encodings),
                **metadata
            }

            self.metadata[person_id] = enhanced_metadata
            self.encoding_validation[person_id] = encoding_metadata

            # Save database
            if self.save_database():
                self.logger.info(f"✅ Successfully added {person_id}: {successful_images} encodings")
                self._log_person_summary(person_id)
                return True
            else:
                # Rollback on save failure
                if person_id in self.known_faces:
                    del self.known_faces[person_id]
                if person_id in self.metadata:
                    del self.metadata[person_id]
                if person_id in self.encoding_validation:
                    del self.encoding_validation[person_id]
                return False

        except Exception as e:
            self.logger.error(f"Error adding person {person_id}: {e}")
            return False

    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality for encoding"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)

            # Brightness analysis
            mean_brightness = np.mean(gray)
            brightness_score = max(0.0, 1.0 - abs(mean_brightness - 128) / 128)

            # Contrast analysis
            contrast_score = np.std(gray) / 128.0
            contrast_score = min(1.0, contrast_score)

            # Overall quality score
            quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)

            return quality_score

        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            return 0.0

    def _create_enhanced_encoding(self, image: np.ndarray, image_path: str) -> Optional[np.ndarray]:
        """Create enhanced encoding with better preprocessing"""
        try:
            # Enhanced preprocessing pipeline
            processed_image = self._enhanced_preprocessing(image)
            if processed_image is None:
                return None

            # Create ArcFace embedding
            representation = DeepFace.represent(
                img_path=processed_image,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )

            if representation and len(representation) > 0:
                encoding = np.array(representation[0]["embedding"])

                # Validate encoding
                if encoding.shape == (512,):
                    # L2 normalization
                    encoding = encoding / (np.linalg.norm(encoding) + 1e-8)

                    # Additional validation
                    if np.all(np.isfinite(encoding)) and not np.all(encoding == 0):
                        return encoding.tolist()
                    else:
                        self.logger.warning(f"Invalid encoding values for {image_path}")
                else:
                    self.logger.warning(f"Unexpected encoding shape: {encoding.shape} for {image_path}")

            return None

        except Exception as e:
            self.logger.error(f"Enhanced encoding creation error for {image_path}: {e}")
            return None

    def _enhanced_preprocessing(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced image preprocessing for consistent encodings"""
        try:
            # Resize if too large (memory efficiency)
            h, w = image.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Noise reduction
            image = cv2.bilateralFilter(image, 9, 75, 75)

            # Histogram equalization for consistent lighting
            if len(image.shape) == 3:
                # Convert to LAB and equalize L channel
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                image = cv2.merge([l, a, b])
                image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

            # Convert to RGB for DeepFace
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            self.logger.error(f"Enhanced preprocessing error: {e}")
            return None

    def _select_best_encodings(self, encodings: List[np.ndarray],
                               metadata: List[Dict]) -> List[np.ndarray]:
        """Select best encodings based on quality and diversity"""
        try:
            if len(encodings) <= self.target_encodings_per_person:
                return encodings

            # Sort by quality score
            quality_sorted = sorted(
                zip(encodings, metadata),
                key=lambda x: x[1]["quality_score"],
                reverse=True
            )

            # Select top quality encodings
            selected = []
            selected_metadata = []

            for encoding, meta in quality_sorted:
                if len(selected) >= self.target_encodings_per_person:
                    break

                # Check diversity (avoid too similar encodings)
                if self._is_diverse_encoding(encoding, selected):
                    selected.append(encoding)
                    selected_metadata.append(meta)

            self.logger.info(f"Selected {len(selected)} best encodings from {len(encodings)}")
            return selected

        except Exception as e:
            self.logger.error(f"Best encoding selection error: {e}")
            return encodings[:self.target_encodings_per_person]

    def _is_diverse_encoding(self, encoding: np.ndarray, existing: List[np.ndarray]) -> bool:
        """Check if encoding is diverse enough from existing ones"""
        try:
            if not existing:
                return True

            enc_array = np.array(encoding)
            min_diversity = 0.95  # Minimum cosine similarity for diversity

            for existing_enc in existing:
                existing_array = np.array(existing_enc)
                similarity = np.dot(enc_array, existing_array)
                if similarity > min_diversity:
                    return False  # Too similar to existing encoding

            return True

        except Exception as e:
            self.logger.error(f"Diversity check error: {e}")
            return True

    def _compute_validation_hash(self, encodings: List[np.ndarray]) -> str:
        """Compute validation hash for encodings"""
        try:
            # Create a hash based on encoding statistics
            all_encodings = np.array(encodings)
            stats = {
                "mean": np.mean(all_encodings, axis=0).tolist(),
                "std": np.std(all_encodings, axis=0).tolist(),
                "count": len(encodings)
            }

            stats_str = json.dumps(stats, sort_keys=True)
            return hashlib.md5(stats_str.encode()).hexdigest()

        except Exception as e:
            self.logger.error(f"Validation hash error: {e}")
            return "unknown"

    def _log_person_summary(self, person_id: str):
        """Log person summary for debugging"""
        try:
            if person_id not in self.known_faces:
                return

            encodings = self.known_faces[person_id]
            metadata = self.metadata.get(person_id, {})

            print(f"\n{'=' * 60}")
            print(f"👤 PERSON SUMMARY: {person_id.upper()}")
            print(f"{'=' * 60}")
            print(f"📊 Encodings: {len(encodings)}")
            print(f"📷 Source images: {metadata.get('num_source_images', 'unknown')}")
            print(f"✅ Successful: {metadata.get('successful_images', 'unknown')}")

            quality_info = metadata.get('encoding_quality', {})
            print(f"⭐ Quality - Min: {quality_info.get('min_quality', 0):.3f}, "
                  f"Max: {quality_info.get('max_quality', 0):.3f}, "
                  f"Avg: {quality_info.get('avg_quality', 0):.3f}")

            # Encoding statistics
            if encodings:
                enc_array = np.array(encodings)
                print(f"🧮 Encoding stats - Shape: {enc_array.shape}, "
                      f"Mean norm: {np.mean([np.linalg.norm(enc) for enc in encodings]):.3f}")

                # Similarity analysis between own encodings
                similarities = []
                for i in range(len(encodings)):
                    for j in range(i + 1, len(encodings)):
                        sim = np.dot(encodings[i], encodings[j])
                        similarities.append(sim)

                if similarities:
                    print(f"🔗 Internal similarity - Min: {min(similarities):.3f}, "
                          f"Max: {max(similarities):.3f}, "
                          f"Avg: {np.mean(similarities):.3f}")

            print(f"🔧 Model: {metadata.get('model', 'unknown')}")
            print(f"📅 Added: {metadata.get('added_date', 'unknown')}")
            print(f"{'=' * 60}")

        except Exception as e:
            self.logger.error(f"Person summary error: {e}")

    def remove_person(self, person_id: str) -> bool:
        """Remove person from enhanced database"""
        try:
            if person_id not in self.known_faces:
                self.logger.warning(f"Person {person_id} not found")
                return False

            # Remove from all storage
            del self.known_faces[person_id]
            if person_id in self.metadata:
                del self.metadata[person_id]
            if person_id in self.encoding_validation:
                del self.encoding_validation[person_id]

            # Save database
            if self.save_database():
                self.logger.info(f"✅ Removed {person_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error removing {person_id}: {e}")
            return False

    def get_face_encodings(self) -> Dict[str, List[np.ndarray]]:
        """Get all face encodings with validation"""
        try:
            # Validate before returning
            validated_faces = {}

            for person_id, encodings in self.known_faces.items():
                valid_encodings = []

                for encoding in encodings:
                    try:
                        enc_array = np.array(encoding)
                        if (enc_array.shape == (512,) and
                                np.all(np.isfinite(enc_array)) and
                                not np.all(enc_array == 0)):
                            valid_encodings.append(enc_array)
                    except Exception:
                        continue

                if valid_encodings:
                    validated_faces[person_id] = valid_encodings
                else:
                    self.logger.warning(f"No valid encodings for {person_id}")

            return validated_faces

        except Exception as e:
            self.logger.error(f"Get encodings error: {e}")
            return {}

    def get_known_persons(self) -> List[str]:
        """Get list of known person IDs"""
        return list(self.known_faces.keys())

    def get_person_info(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced information about a person"""
        if person_id not in self.known_faces:
            return None

        base_info = {
            "person_id": person_id,
            "num_encodings": len(self.known_faces[person_id]),
            "metadata": self.metadata.get(person_id, {})
        }

        # Add validation info
        if person_id in self.encoding_validation:
            base_info["encoding_validation"] = self.encoding_validation[person_id]

        return base_info

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced database statistics"""
        try:
            total_encodings = sum(len(encodings) for encodings in self.known_faces.values())

            # Quality analysis
            all_qualities = []
            for person_id in self.known_faces:
                metadata = self.metadata.get(person_id, {})
                quality_info = metadata.get('encoding_quality', {})
                if 'avg_quality' in quality_info:
                    all_qualities.append(quality_info['avg_quality'])

            # Person encoding distribution
            encoding_counts = [len(encodings) for encodings in self.known_faces.values()]

            stats = {
                "total_persons": len(self.known_faces),
                "total_encodings": total_encodings,
                "average_encodings": total_encodings / len(self.known_faces) if self.known_faces else 0,
                "database_path": str(self.db_path),
                "deepface_available": DEEPFACE_AVAILABLE,
                "quality_analysis": {
                    "avg_quality": np.mean(all_qualities) if all_qualities else 0,
                    "min_quality": min(all_qualities) if all_qualities else 0,
                    "max_quality": max(all_qualities) if all_qualities else 0
                },
                "encoding_distribution": {
                    "min_encodings": min(encoding_counts) if encoding_counts else 0,
                    "max_encodings": max(encoding_counts) if encoding_counts else 0,
                    "avg_encodings": np.mean(encoding_counts) if encoding_counts else 0
                },
                "validation_status": "enhanced"
            }

            return stats

        except Exception as e:
            self.logger.error(f"Stats error: {e}")
            return {"error": str(e)}

    def validate_person_encodings(self, person_id: str) -> Dict[str, Any]:
        """Validate specific person's encodings"""
        try:
            if person_id not in self.known_faces:
                return {"error": "Person not found"}

            encodings = self.known_faces[person_id]
            enc_array = np.array(encodings)

            # Compute internal similarities
            similarities = []
            for i in range(len(encodings)):
                for j in range(i + 1, len(encodings)):
                    sim = np.dot(encodings[i], encodings[j])
                    similarities.append(sim)

            # Compute norms
            norms = [np.linalg.norm(enc) for enc in encodings]

            validation_result = {
                "person_id": person_id,
                "num_encodings": len(encodings),
                "encoding_shape": enc_array.shape,
                "internal_similarities": {
                    "min": min(similarities) if similarities else 0,
                    "max": max(similarities) if similarities else 0,
                    "mean": np.mean(similarities) if similarities else 0,
                    "std": np.std(similarities) if similarities else 0
                },
                "norms": {
                    "min": min(norms),
                    "max": max(norms),
                    "mean": np.mean(norms),
                    "std": np.std(norms)
                },
                "validation_passed": all(0.8 <= norm <= 1.2 for norm in norms),
                "quality_consistent": len(set(similarities)) > 1 if similarities else False
            }

            return validation_result

        except Exception as e:
            self.logger.error(f"Validation error for {person_id}: {e}")
            return {"error": str(e)}

    def _create_backup(self):
        """Create enhanced backup of database files"""
        try:
            backup_dir = self.db_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Backup all files
            files_to_backup = [
                (self.encodings_file, f"enhanced_faces_backup_{timestamp}.pkl"),
                (self.metadata_file, f"enhanced_metadata_backup_{timestamp}.json"),
                (self.validation_file, f"encoding_validation_backup_{timestamp}.json")
            ]

            for source_file, backup_name in files_to_backup:
                if source_file.exists():
                    backup_path = backup_dir / backup_name
                    import shutil
                    shutil.copy2(source_file, backup_path)

            # Keep only last 10 backups
            backups = sorted(backup_dir.glob("enhanced_faces_backup_*.pkl"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
                    # Remove corresponding metadata and validation backups
                    timestamp_part = old_backup.stem.split('_')[-2:]
                    timestamp_str = '_'.join(timestamp_part)

                    meta_backup = backup_dir / f"enhanced_metadata_backup_{timestamp_str}.json"
                    if meta_backup.exists():
                        meta_backup.unlink()

                    val_backup = backup_dir / f"encoding_validation_backup_{timestamp_str}.json"
                    if val_backup.exists():
                        val_backup.unlink()

        except Exception as e:
            self.logger.warning(f"Backup creation failed: {e}")

    def is_connected(self) -> bool:
        """Check if database is accessible"""
        return self.db_path.exists() and os.access(self.db_path, os.R_OK | os.W_OK)

    def export_person_analysis(self, person_id: str, output_path: str) -> bool:
        """Export detailed person analysis for debugging"""
        try:
            if person_id not in self.known_faces:
                return False

            analysis = {
                "person_id": person_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "basic_info": self.get_person_info(person_id),
                "validation_results": self.validate_person_encodings(person_id),
                "encoding_details": []
            }

            # Add detailed encoding information
            for i, encoding in enumerate(self.known_faces[person_id]):
                enc_array = np.array(encoding)
                enc_details = {
                    "encoding_index": i,
                    "shape": enc_array.shape,
                    "norm": float(np.linalg.norm(enc_array)),
                    "mean": float(np.mean(enc_array)),
                    "std": float(np.std(enc_array)),
                    "min": float(np.min(enc_array)),
                    "max": float(np.max(enc_array))
                }
                analysis["encoding_details"].append(enc_details)

            # Save analysis
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Person analysis exported: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Export analysis error: {e}")
            return False


# Compatibility alias
DatabaseManager = EnhancedDatabaseManager


# Utility functions
def add_person_from_folder(person_id: str, folder_path: str) -> bool:
    """Enhanced utility to add a person from a folder of images"""
    try:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return False

        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(folder.glob(f'*{ext}')))
            image_files.extend(list(folder.glob(f'*{ext.upper()}')))

        if not image_files:
            print(f"❌ No images found in {folder_path}")
            return False

        # Add to enhanced database
        db = EnhancedDatabaseManager()

        metadata = {
            "source_folder": str(folder_path),
            "department": "Unknown",
            "access_level": "Standard",
            "import_method": "folder_import"
        }

        success = db.add_person(person_id, [str(f) for f in image_files], metadata)

        if success:
            print(f"✅ Added {person_id}: {len(image_files)} images processed")
            return True
        else:
            print(f"❌ Failed to add {person_id}")
            return False

    except Exception as e:
        print(f"❌ Error adding {person_id}: {e}")
        return False


def test_enhanced_database():
    """Test enhanced database functionality"""
    print("🧪 Testing Enhanced Database...")

    db = EnhancedDatabaseManager()

    # Print stats
    stats = db.get_enhanced_stats()
    print(f"📊 Enhanced Database Stats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue}")
        else:
            print(f"   {key}: {value}")

    # List known persons with validation
    persons = db.get_known_persons()
    print(f"\n👥 Known Persons ({len(persons)}):")
    for person in persons:
        info = db.get_person_info(person)
        validation = db.validate_person_encodings(person)
        if info and validation:
            print(f"   - {person}: {info['num_encodings']} encodings, "
                  f"validation: {'✅' if validation['validation_passed'] else '❌'}")

    print(" database test completed")


if __name__ == "__main__":
    test_enhanced_database()