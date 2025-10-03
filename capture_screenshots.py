import cv2
import datetime
from pathlib import Path
import argparse


def capture_screenshots_from_camera(rtsp_url, save_dir="known_faces/", person_id=None, num_images=5):
    save_dir = Path(save_dir)
    if person_id:
        save_dir = save_dir / person_id
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to camera: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(" Could not open camera stream.")
        return False

    print("\nInstructions:")
    print(" - Adjust yourself in front of the camera.")
    print(f" - Press 's' to save a screenshot ({num_images} needed).")
    print(" - Press 'q' to quit at any time.\n")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame.")
            break

        preview = frame.copy()
        cv2.putText(preview, f"Screenshot {count + 1}/{num_images} - Press 's' to capture, 'q' to quit.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Face Screenshot", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"{person_id or 'person'}_{timestamp}_{count + 1}.png"
            cv2.imwrite(str(filename), frame)
            print(f" Saved: {filename}")
            count += 1

        elif key == ord('q'):
            print("Exited by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished screenshot capture.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp', type=str, required=True, help='RTSP or camera URL')
    parser.add_argument('--person', type=str, required=True, help='Person ID (subfolder)')
    parser.add_argument('--num', type=int, default=5, help='Number of screenshots')
    parser.add_argument('--dir', type=str, default="known_faces", help='Save directory')
    args = parser.parse_args()

    capture_screenshots_from_camera(
        args.rtsp,
        save_dir=args.dir,
        person_id=args.person,
        num_images=args.num
    )
