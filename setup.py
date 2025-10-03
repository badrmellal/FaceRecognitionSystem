#!/usr/bin/env python3
"""
Enhanced Setup for Face Recognition System
Improved setup with better validation and debugging tools
"""

import sys
import os
import requests
import logging
import json
from pathlib import Path
import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from database import EnhancedDatabaseManager, add_person_from_folder
from config import Config


def setup_logging():
    """Setup enhanced logging"""
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/setup.log', encoding='utf-8')
        ]
    )


def create_directories():
    """Create necessary directories with enhanced structure"""
    directories = [
        "models",
        "known_faces",
        "logs",
        "logs/security_evidence",
        "logs/unauthorized_access",
        "logs/screenshots",
        "logs/audio_alerts",
        "config",
        "exports",
        "backups"
    ]

    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

    print("✅ Enhanced directory structure created")


def download_file(url: str, output_path: Path) -> bool:
    """Download file with enhanced progress tracking"""
    try:
        print(f"📥 Downloading {output_path.name}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\r    Progress: {progress:.1f}% ({downloaded // 1024 // 1024}MB/{total_size // 1024 // 1024}MB)",
                            end='', flush=True)

        print()  # New line
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def setup_enhanced_models():
    """Setup face detection models with validation"""
    print(f"\n🤖 SETTING UP ENHANCED MODELS")
    print("=" * 50)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Try to get specialized face model
    face_model_path = models_dir / "yolov11n-face.pt"
    if not face_model_path.exists():
        face_model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt"
        print("🎯 Downloading specialized face detection model...")

        if download_file(face_model_url, face_model_path):
            print("✅ Specialized face model downloaded")
        else:
            print("⚠️ Failed to download specialized model, will use general YOLO")
    else:
        print("✅ Specialized face model already present")

    # Get general YOLO model as fallback
    general_model_path = models_dir / "yolov11n.pt"
    if not general_model_path.exists():
        print("🎯 Setting up general YOLO model...")
        try:
            model = YOLO("yolo11n.pt")

            # Move downloaded model to models directory
            if Path("yolo11n.pt").exists():
                shutil.move("yolo11n.pt", str(general_model_path))
            else:
                # Try to save the model
                model.save(str(general_model_path))

            print("✅ General YOLO model ready")

        except Exception as e:
            print(f"⚠️ YOLO model setup issue: {e}")
    else:
        print("✅ General YOLO model already present")

    # Test DeepFace for face recognition
    print("\n🧠 Testing face recognition capabilities...")
    try:
        from deepface import DeepFace
        import numpy as np

        # Test ArcFace with a dummy image
        test_image = np.ones((112, 112, 3), dtype=np.uint8) * 128
        representation = DeepFace.represent(
            img_path=test_image,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False
        )

        if representation and len(representation) > 0:
            encoding = np.array(representation[0]["embedding"])
            if encoding.shape == (512,):
                print("✅ ArcFace model operational")
                print(f"   Embedding shape: {encoding.shape}")
                print(f"   Embedding norm: {np.linalg.norm(encoding):.3f}")
            else:
                print(f"⚠️ Unexpected embedding shape: {encoding.shape}")
        else:
            print("⚠️ ArcFace test failed - no representation")

    except ImportError:
        print("❌ DeepFace not installed")
        print("   Install with: pip install deepface")
        return False
    except Exception as e:
        print(f"⚠️ DeepFace error: {e}")
        if "tf-keras" in str(e).lower():
            print("   Solution: pip install tf-keras")
        elif "tensorflow" in str(e).lower():
            print("   Solution: pip install tensorflow")
        return False

    # Validate model files
    print(f"\n🔍 Validating model files...")
    if face_model_path.exists():
        size_mb = face_model_path.stat().st_size / (1024 * 1024)
        print(f"   Face model: {size_mb:.1f} MB")

    if general_model_path.exists():
        size_mb = general_model_path.stat().st_size / (1024 * 1024)
        print(f"   General model: {size_mb:.1f} MB")

    return True


def setup_enhanced_known_faces():
    """Setup enhanced known faces database with validation"""
    print(f"\n👥 SETTING UP ENHANCED FACE DATABASE")
    print("=" * 50)

    db = EnhancedDatabaseManager()

    # Create example person folders with enhanced instructions
    known_faces_dir = Path("known_faces")
    example_persons = [
        "badr_mellal",
        "mounia_amrhar",
        "personnel_securite",
        "directeur_general",
        "invite_autorise"
    ]

    print("📁 Creating example person folders...")
    for person in example_persons:
        person_dir = known_faces_dir / person
        person_dir.mkdir(exist_ok=True)

        # Create enhanced instruction file
        instructions = f"""ENHANCED INSTRUCTIONS FOR {person.upper()}
{'=' * 70}

🎯 QUALITY REQUIREMENTS FOR BEST RECOGNITION:
   - 5-10 high-quality photos minimum
   - Face should occupy 50-70% of the image
   - Clear, well-lit photos (avoid shadows on face)
   - Multiple angles: front, slight left, slight right
   - Different expressions: neutral, smiling
   - Avoid sunglasses, hats, or face coverings
   - Resolution: 640x640 pixels or higher
   - Sharp, non-blurry images

📸 RECOMMENDED PHOTO TYPES:
   1. {person}_front_neutral.jpg (straight-on, neutral expression)
   2. {person}_front_smile.jpg (straight-on, smiling)
   3. {person}_left_angle.jpg (slight left turn)
   4. {person}_right_angle.jpg (slight right turn)
   5. {person}_up_angle.jpg (slight upward look)
   6. {person}_down_angle.jpg (slight downward look)
   7. {person}_different_lighting.jpg (different lighting condition)

⚠️ AVOID THESE ISSUES:
   - Blurry or out-of-focus images
   - Too much shadow or backlighting
   - Face too small in frame
   - Extreme angles or poses
   - Low resolution images
   - Images with other people's faces

🔧 AFTER ADDING PHOTOS:
   1. Run: python setup.py --add-faces
   2. Check validation results in logs
   3. If recognition issues persist, add more high-quality photos
   4. Use different lighting conditions for robustness

💡 TIPS FOR BEST RESULTS:
   - Take photos in the same location where recognition will happen
   - Use similar lighting conditions as the surveillance area
   - Include photos with and without glasses if person wears them
   - Quality over quantity - 5 excellent photos > 10 poor photos

📊 The system uses ArcFace for recognition with enhanced preprocessing
   and quality validation to prevent misidentification issues.
"""

        readme_path = person_dir / "ENHANCED_INSTRUCTIONS.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(instructions)

    print(f"✅ Created {len(example_persons)} enhanced example folders")

    # Check for existing images and validate quality
    existing_persons = []
    for person_dir in known_faces_dir.iterdir():
        if person_dir.is_dir() and not person_dir.name.startswith('.'):
            # Check for image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(person_dir.glob(f'*{ext}')))
                image_files.extend(list(person_dir.glob(f'*{ext.upper()}')))

            # Filter out instruction files
            image_files = [f for f in image_files if not f.name.startswith(('INSTRUCTIONS', 'ENHANCED_INSTRUCTIONS'))]

            if len(image_files) >= 3:  # Minimum 3 images
                # Quick quality check
                quality_scores = []
                for img_path in image_files[:3]:  # Check first 3 images
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            size_score = min(1.0, (w * h) / (640 * 640))
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                            sharpness_score = min(1.0, sharpness / 300.0)
                            quality_scores.append((size_score + sharpness_score) / 2)
                    except:
                        quality_scores.append(0.0)

                avg_quality = np.mean(quality_scores) if quality_scores else 0.0
                existing_persons.append((person_dir.name, len(image_files), avg_quality))

    if existing_persons:
        print(f"\n📸 Found person folders with images:")
        for person, count, quality in existing_persons:
            quality_indicator = "🟢" if quality > 0.5 else "🟡" if quality > 0.3 else "🔴"
            print(f"   {quality_indicator} {person}: {count} images (quality: {quality:.2f})")

        print(f"\n🟢 = Good quality, 🟡 = Medium quality, 🔴 = Low quality")

        response = input("\nAdd these persons to enhanced database? (o/N): ").strip().lower()

        if response == 'o':
            success_count = 0
            for person, count, quality in existing_persons:
                folder_path = known_faces_dir / person
                print(f"\n🔄 Processing {person}...")

                if add_person_from_folder(person, str(folder_path)):
                    success_count += 1
                    print(f"✅ Successfully added {person}")

                    # Validate the added person
                    validation_result = db.validate_person_encodings(person)
                    if validation_result.get('validation_passed', False):
                        print(f"   ✅ Validation passed")
                    else:
                        print(f"   ⚠️ Validation issues detected")
                        print(f"   💡 Consider adding more high-quality photos")
                else:
                    print(f"❌ Failed to add {person}")

            print(f"\n📊 RESULTS: {success_count}/{len(existing_persons)} persons added successfully")
        else:
            print("⏭️ Skipping person addition")

    # Show enhanced database status
    stats = db.get_enhanced_stats()
    print(f"\n📊 Enhanced Database Status:")
    print(f"   Total persons: {stats['total_persons']}")
    print(f"   Total encodings: {stats['total_encodings']}")

    if stats['total_persons'] > 0:
        print(f"   Average encodings per person: {stats['average_encodings']:.1f}")
        quality_info = stats.get('quality_analysis', {})
        if quality_info.get('avg_quality', 0) > 0:
            print(f"   Average quality: {quality_info['avg_quality']:.3f}")

    if stats['total_persons'] == 0:
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Add high-quality photos to known_faces/ folders")
        print(f"   2. Follow the ENHANCED_INSTRUCTIONS.txt in each folder")
        print(f"   3. Run: python setup.py --add-faces")
        print(f"   4. Test with: python main.py --rtsp-url YOUR_CAMERA_URL")

    return True


def add_faces_enhanced():
    """Enhanced interactive face addition with validation"""
    print(f"\n👤 ENHANCED FACE ADDITION")
    print("=" * 50)

    known_faces_dir = Path("known_faces")

    # Find folders with images
    candidate_folders = []
    for person_dir in known_faces_dir.iterdir():
        if person_dir.is_dir() and not person_dir.name.startswith('.'):
            # Check for images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(person_dir.glob(f'*{ext}')))
                image_files.extend(list(person_dir.glob(f'*{ext.upper()}')))

            # Filter out instruction files
            image_files = [f for f in image_files if not f.name.startswith(('INSTRUCTIONS', 'ENHANCED_INSTRUCTIONS'))]

            if len(image_files) >= 2:
                candidate_folders.append((person_dir.name, len(image_files)))

    if not candidate_folders:
        print("❌ No candidate folders with images found")
        print("   Add photos to known_faces/ folders and try again")
        return False

    db = EnhancedDatabaseManager()
    existing_persons = db.get_known_persons()

    print("📁 Candidate folders:")
    new_folders = []
    for person, count in candidate_folders:
        if person in existing_persons:
            print(f"   🔄 {person}: {count} images (UPDATE EXISTING)")
        else:
            print(f"   ➕ {person}: {count} images (NEW)")
            new_folders.append((person, count))

    if not new_folders:
        print("ℹ️ All persons already exist in database")
        response = input("Update existing persons? (o/N): ").strip().lower()
        if response == 'o':
            new_folders = candidate_folders
        else:
            return True

    print(f"\n🎯 Processing {len(new_folders)} person(s)")
    response = input("Continue with enhanced processing? (O/n): ").strip().lower()

    if response != 'n':
        success_count = 0
        total_validations_passed = 0

        for person, count in new_folders:
            folder_path = known_faces_dir / person
            print(f"\n{'=' * 60}")
            print(f"🔄 PROCESSING: {person.upper()}")
            print(f"{'=' * 60}")

            if add_person_from_folder(person, str(folder_path)):
                success_count += 1
                print(f"✅ Successfully processed {person}")

                # Enhanced validation
                validation_result = db.validate_person_encodings(person)
                if validation_result.get('validation_passed', False):
                    total_validations_passed += 1
                    print(f"✅ Validation: PASSED")

                    # Print detailed validation info
                    sim_info = validation_result.get('internal_similarities', {})
                    print(f"   📊 Internal similarity: {sim_info.get('mean', 0):.3f} ± {sim_info.get('std', 0):.3f}")
                    print(f"   📏 Encoding consistency: ✅")
                else:
                    print(f"⚠️ Validation: ISSUES DETECTED")
                    print(f"   💡 Consider adding more high-quality, diverse photos")

                # Export detailed analysis for debugging
                analysis_path = f"logs/{person}_analysis.json"
                if db.export_person_analysis(person, analysis_path):
                    print(f"   📄 Detailed analysis: {analysis_path}")

            else:
                print(f"❌ Failed to process {person}")

        # Final summary
        print(f"\n{'=' * 60}")
        print(f"📊 ENHANCED PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"✅ Successfully processed: {success_count}/{len(new_folders)}")
        print(f"✅ Validations passed: {total_validations_passed}/{success_count}")

        if total_validations_passed < success_count:
            print(f"⚠️ Some persons have validation issues")
            print(f"💡 Check logs/ for detailed analysis")
            print(f"💡 Consider adding more diverse, high-quality photos")

        # Final database stats
        stats = db.get_enhanced_stats()
        print(f"\n📊 Final Database Stats:")
        print(f"   Persons: {stats['total_persons']}")
        print(f"   Encodings: {stats['total_encodings']}")
        print(f"   Avg quality: {stats.get('quality_analysis', {}).get('avg_quality', 0):.3f}")

        return success_count > 0

    return False


def validate_system_dependencies():
    """Validate all system dependencies"""
    print(f"\n🔍 VALIDATING SYSTEM DEPENDENCIES")
    print("=" * 50)

    issues = []

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        issues.append(f"❌ Python {python_version.major}.{python_version.minor} (require 3.8+)")

    # Check required packages
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('ultralytics', 'ultralytics'),
        ('deepface', 'deepface'),
        ('sklearn', 'scikit-learn'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pandas', 'pandas'),
        ('openpyxl', 'openpyxl')
    ]

    for module, package in required_packages:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            issues.append(f"❌ {package} - install with: pip install {package}")

    # Check for GPU support
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ M3 Max GPU support (MPS)")
        elif torch.cuda.is_available():
            print("✅ CUDA GPU support")
        else:
            print("ℹ️ CPU processing (no GPU acceleration)")
    except ImportError:
        print("ℹ️ PyTorch not available")

    if issues:
        print(f"\n⚠️ DEPENDENCY ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print(f"\n✅ All dependencies validated successfully")
        return True


def create_enhanced_config():
    """Create enhanced configuration file"""
    print(f"\n⚙️ Creating enhanced configuration...")

    config = Config()
    config.optimize_for_accuracy()  # Start with accuracy-optimized settings

    print("✅ Enhanced configuration created")
    config.print_current_settings()

    return True


def main():
    """Enhanced main setup function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Setup for Face Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Full enhanced setup
  %(prog)s --add-faces        # Add faces from folders  
  %(prog)s --models-only      # Setup models only
  %(prog)s --validate-only    # Validate dependencies only
  %(prog)s --optimize-accuracy # Optimize for accuracy
        """
    )

    parser.add_argument('--models-only', action='store_true',
                        help='Setup models only')
    parser.add_argument('--faces-only', action='store_true',
                        help='Setup faces only')
    parser.add_argument('--add-faces', action='store_true',
                        help='Add faces from folders with enhanced validation')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate dependencies only')
    parser.add_argument('--optimize-accuracy', action='store_true',
                        help='Create accuracy-optimized configuration')

    args = parser.parse_args()

    setup_logging()

    print("🚀 ENHANCED FACE RECOGNITION SYSTEM SETUP")
    print("=" * 70)
    print("Features: ArcFace recognition • Enhanced quality validation")
    print("         • Improved accuracy • Better preprocessing")
    print("         • Comprehensive validation • Debug tools")
    print("=" * 70)

    try:
        success = True

        # Validate dependencies first
        if args.validate_only:
            return 0 if validate_system_dependencies() else 1

        if not validate_system_dependencies():
            print(f"\n❌ Dependency validation failed")
            print(f"   Install missing packages and run setup again")
            return 1

        # Create directories
        create_directories()

        if args.add_faces:
            # Only add faces with enhanced validation
            success = add_faces_enhanced()
        elif args.faces_only:
            # Only setup faces
            success = setup_enhanced_known_faces()
        elif args.models_only:
            # Only setup models
            success = setup_enhanced_models()
        elif args.optimize_accuracy:
            # Create optimized config
            success = create_enhanced_config()
        else:
            # Full enhanced setup
            print(f"\n🔧 Starting full enhanced setup...")

            # Setup models
            if not setup_enhanced_models():
                print("❌ Model setup failed")
                success = False

            # Setup enhanced configuration
            if success and not create_enhanced_config():
                print("❌ Configuration setup failed")
                success = False

            # Setup known faces
            if success and not setup_enhanced_known_faces():
                print("❌ Face database setup failed")
                success = False

        if success:
            print(f"\n{'=' * 70}")
            print("🎉 ENHANCED SETUP COMPLETED SUCCESSFULLY!")
            print("")
            print("🚀 NEXT STEPS:")
            print("1. Verify your known faces in known_faces/ folders")
            print("2. Run system validation:")
            print("   python main.py --validate-only")
            print("3. Start monitoring:")
            print("   python main.py --rtsp-url rtsp://admin:pass@IP:554/...")
            print("")
            print("✨ ENHANCED FEATURES ENABLED:")
            print("   • ArcFace recognition with L2 normalization")
            print("   • Enhanced preprocessing pipeline")
            print("   • Quality-based encoding validation")
            print("   • Confidence gap analysis for multi-person scenarios")
            print("   • Identity consistency checking")
            print("   • Comprehensive logging and debugging")
            print("")
            print("🔧 OPTIMIZATION:")
            print("   • Optimized for 5MP Imou cameras")
            print("   • Enhanced accuracy settings")
            print("   • Better false positive prevention")
            print(f"{'=' * 70}")
        else:
            print(f"\n❌ ENHANCED SETUP FAILED")
            print("Check error messages above and try again")
            return 1

    except KeyboardInterrupt:
        print(f"\n\n🛑 Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        logging.error(f"Setup error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())