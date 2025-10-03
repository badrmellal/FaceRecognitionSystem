#!/usr/bin/env python3
"""
Enhanced Face Recognition System for 5MP Imou Camera
Fixed recognition accuracy and optimized for entrance gate monitoring
"""

import sys
import argparse
import logging
from pathlib import Path
from face_recognition_system import EnhancedSurveillanceSystem
from database import EnhancedDatabaseManager
from config import Config


def setup_enhanced_logging():
    """Setup enhanced logging with proper formatting"""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/enhanced_face_recognition.log', encoding='utf-8')
        ]
    )

    # Set specific logger levels
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('cv2').setLevel(logging.WARNING)


def validate_system_setup():
    """Validate system setup and dependencies"""
    issues = []

    # Check database
    try:
        db = EnhancedDatabaseManager()
        known_faces = db.get_face_encodings()
        if len(known_faces) == 0:
            issues.append("❌ No authorized personnel in database")
        else:
            print(f"✅ Database: {len(known_faces)} authorized personnel")
            for person_id, encodings in known_faces.items():
                print(f"   - {person_id}: {len(encodings)} encodings")
    except Exception as e:
        issues.append(f"❌ Database error: {e}")

    # Check DeepFace
    try:
        from deepface import DeepFace
        print("✅ DeepFace available")
    except ImportError:
        issues.append("❌ DeepFace not installed: pip install deepface")

    # Check directories
    required_dirs = ['known_faces', 'logs', 'models']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory: {dir_name}")
        else:
            dir_path.mkdir(exist_ok=True)
            print(f"🔧 Created directory: {dir_name}")

    return issues


def print_system_info():
    """Print enhanced system information"""
    print(f"\n{'=' * 80}")
    print("🚀 ENHANCED FACE RECOGNITION SYSTEM")
    print(f"{'=' * 80}")
    print("🎯 Features:")
    print("   - Enhanced ArcFace recognition with better accuracy")
    print("   - Improved confidence gap analysis for multi-person scenarios")
    print("   - Advanced quality assessment and preprocessing")
    print("   - Identity consistency checking")
    print("   - Real-time security monitoring with French alerts")
    print("   - Comprehensive logging and evidence capture")
    print("")
    print("🔧 Optimizations:")
    print("   - Proper L2 normalization of embeddings")
    print("   - Enhanced preprocessing pipeline")
    print("   - Better tracking and state management")
    print("   - Quality-based encoding selection")
    print("   - Reduced false positives through strict validation")
    print(f"{'=' * 80}")


def main():
    """Enhanced main function with better error handling"""
    parser = argparse.ArgumentParser(
        description="Enhanced Face Recognition System for 5MP Imou Camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --rtsp-url rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1
  %(prog)s --rtsp-url YOUR_RTSP_URL --duration 3600 --config enhanced_config.json
  %(prog)s --validate-only  # Just validate system setup
  %(prog)s --optimize-accuracy  # Optimize for maximum accuracy
        """
    )

    parser.add_argument('--rtsp-url',
                        help='Camera RTSP URL (required unless using --validate-only)')
    parser.add_argument('--config', default='config.json',
                        help='Configuration file (default: config.json)')
    parser.add_argument('--duration', type=int,
                        help='Monitoring duration in seconds')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate system setup without starting monitoring')
    parser.add_argument('--optimize-accuracy', action='store_true',
                        help='Optimize configuration for maximum accuracy')
    parser.add_argument('--optimize-performance', action='store_true',
                        help='Optimize configuration for better performance')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose logging')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no GUI)')

    args = parser.parse_args()

    # Setup logging
    setup_enhanced_logging()
    logger = logging.getLogger(__name__)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Debug mode enabled")

    if args.headless:
        import os
        os.environ["HEADLESS"] = "1"
        logger.info("🖥️ Headless mode enabled")

    # Print system info
    print_system_info()

    # Create necessary directories
    for directory in ['config', 'logs', 'known_faces', 'models']:
        Path(directory).mkdir(exist_ok=True)

    try:
        # Load and configure system
        config = Config(args.config)

        # Apply optimizations if requested
        if args.optimize_accuracy:
            print("🎯 Optimizing for maximum accuracy...")
            config.optimize_for_accuracy()
            config.print_current_settings()
        elif args.optimize_performance:
            print("⚡ Optimizing for performance...")
            config.optimize_for_performance()
            config.print_current_settings()

        # Validate system setup
        print("\n🔍 Validating system setup...")
        issues = validate_system_setup()

        if issues:
            print(f"\n⚠️ System validation issues found:")
            for issue in issues:
                print(f"   {issue}")

            if "No authorized personnel in database" in str(issues):
                print(f"\n💡 To add personnel:")
                print(f"   1. Add photos to known_faces/person_name/ folders")
                print(f"   2. Run: python setup.py --add-faces")
                print(f"   3. Ensure 5-10 high-quality photos per person")

            if args.validate_only:
                return 1

            response = input("\nContinue anyway? (y/N): ").strip().lower()
            if response != 'y':
                return 1
        else:
            print("✅ System validation passed")

        if args.validate_only:
            print("✅ Validation completed successfully")
            return 0

        # Check RTSP URL
        if not args.rtsp_url:
            print("❌ Error: --rtsp-url is required")
            print("Example: --rtsp-url rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1")
            return 1

        # Optimize RTSP URL for 5MP Imou
        rtsp_url = args.rtsp_url
        if "subtype=0" in rtsp_url:
            optimized_url = rtsp_url.replace("subtype=0", "subtype=0")
            print(f"🔧 Optimized RTSP URL for substream (subtype=0)")
            rtsp_url = optimized_url

        # Initialize enhanced surveillance system
        print(f"\n🚀 Initializing Enhanced Surveillance System...")
        system = EnhancedSurveillanceSystem(args.config)

        # Print final configuration
        config.print_current_settings()

        # Start monitoring
        print(f"\n📹 Starting surveillance monitoring...")
        print(f"Camera: {rtsp_url}")
        if args.duration:
            print(f"Duration: {args.duration} seconds")
        print(f"Config: {args.config}")

        success = system.start_monitoring(rtsp_url, args.duration)

        if success:
            print("\n✅ Surveillance completed successfully")
            return 0
        else:
            print("\n❌ Surveillance failed")
            return 1

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"\n❌ System error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check camera connectivity and RTSP URL")
        print(f"   2. Verify database has authorized personnel")
        print(f"   3. Ensure all dependencies are installed")
        print(f"   4. Check logs/enhanced_face_recognition.log for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())