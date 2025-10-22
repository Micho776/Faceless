"""Enhanced live facial recognition with GPU support, face tracking, and confidence scoring.

Features:
- GPU/CUDA acceleration with CNN face detection model
- Face tracking to reduce CPU usage
- Confidence threshold for better accuracy
- Recognition logging with timestamps
- Command-line arguments for configuration

Usage:
    python recognize_live.py --model cnn --track
    python recognize_live.py --model hog --fps
    python recognize_live.py --help

Press 'q' to quit.
"""
import sys
import pickle
import argparse
import warnings
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import json

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import face_recognition
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_PATH = BASE_DIR / "data" / "encodings.pickle"
LOGS_DIR = BASE_DIR / "logs"
LEARNING_DIR = BASE_DIR / "learning_samples"


class FaceTracker:
    """Simple face tracker using correlation trackers."""
    
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        
    def register(self, location, name):
        """Register a new face with location and name."""
        # Try different tracker APIs based on OpenCV version
        try:
            tracker = cv2.legacy.TrackerKCF_create()  # type: ignore
        except AttributeError:
            try:
                tracker = cv2.TrackerKCF_create()  # type: ignore
            except AttributeError:
                tracker = cv2.legacy_TrackerKCF.create()  # type: ignore
        
        self.objects[self.next_id] = {
            'location': location,
            'name': name,
            'tracker': tracker
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove a tracked face."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, frame, detections=None):
        """Update tracked faces. If detections provided, use them; otherwise track existing."""
        if detections is None:
            # Just track existing faces
            to_remove = []
            for object_id, data in self.objects.items():
                tracker = data['tracker']
                success, box = tracker.update(frame)
                
                if success:
                    x, y, w, h = [int(v) for v in box]
                    # Convert to (top, right, bottom, left) format
                    data['location'] = (y, x + w, y + h, x)
                    self.disappeared[object_id] = 0
                else:
                    self.disappeared[object_id] += 1
                    
                if self.disappeared[object_id] > self.max_disappeared:
                    to_remove.append(object_id)
                    
            for object_id in to_remove:
                self.deregister(object_id)
                
        else:
            # New detections provided - match or create new tracks
            if len(detections) == 0:
                for object_id in list(self.disappeared.keys()):
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # Clear old tracks and create new ones
                self.objects = OrderedDict()
                self.disappeared = OrderedDict()
                
                for location, name in detections:
                    top, right, bottom, left = location
                    
                    # Initialize tracker with bounding box
                    # Try different tracker APIs based on OpenCV version
                    try:
                        tracker = cv2.legacy.TrackerKCF_create()  # type: ignore
                    except AttributeError:
                        try:
                            tracker = cv2.TrackerKCF_create()  # type: ignore
                        except AttributeError:
                            tracker = cv2.legacy_TrackerKCF.create()  # type: ignore
                    
                    bbox = (left, top, right - left, bottom - top)
                    tracker.init(frame, bbox)
                    
                    self.objects[self.next_id] = {
                        'location': location,
                        'name': name,
                        'tracker': tracker
                    }
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
                    
        return self.objects


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='FACELESS - Enhanced Live Facial Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recognize_live.py --model cnn --track
  python recognize_live.py --model hog --no-age-gender
  python recognize_live.py --fps
        """
    )
    
    parser.add_argument('--model', type=str, default='hog', choices=['hog', 'cnn'],
                        help='Face detection model: "hog" (CPU, faster) or "cnn" (GPU, more accurate)')
    parser.add_argument('--track', action='store_true',
                        help='Enable face tracking between frames (reduces CPU usage)')
    parser.add_argument('--detect-interval', type=int, default=30,
                        help='Frames between full face detection when tracking (default: 30)')
    parser.add_argument('--fps', action='store_true',
                        help='Show FPS counter on screen')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale factor for processing (0.25-1.0, lower=faster, default: 0.5)')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Process every Nth frame (0=all frames, 1=every other, default: 0)')
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Confidence threshold for recognition (0.0-1.0, default: 0.6)')
    parser.add_argument('--log', action='store_true',
                        help='Enable recognition logging to logs/ directory')
    parser.add_argument('--learn', action='store_true',
                        help='Enable continuous learning - save high-confidence samples')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available cameras and exit')
    
    return parser.parse_args()


def list_available_cameras(max_test=10):
    """List all available camera indices."""
    print("\n[INFO] Scanning for available cameras...")
    available_cameras = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_cameras.append({
                    'index': i,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                print(f"   Camera {i}: {width}x{height} @ {fps}fps")
            cap.release()
    
    if not available_cameras:
        print("   No cameras found!")
    else:
        print(f"\n[INFO] Found {len(available_cameras)} camera(s)")
        print("[INFO] Use --camera INDEX to select a specific camera")
    
    return available_cameras


def main():
    args = parse_args()
    
    # Handle camera listing
    if args.list_cameras:
        list_available_cameras()
        sys.exit(0)
    
    # Setup logging if enabled
    session_log = []
    log_file = None
    if args.log:
        LOGS_DIR.mkdir(exist_ok=True)
        log_filename = f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file = LOGS_DIR / log_filename
        print(f"[INFO] Logging enabled: {log_file}")
    
    # Setup learning if enabled
    learning_samples = {}
    if args.learn:
        LEARNING_DIR.mkdir(exist_ok=True)
        print(f"[INFO] Continuous learning enabled")
        print("[INFO] High-confidence samples will be saved to: learning_samples/")
        print("[INFO] Press 's' to save current frame for training")
    
    # Load saved encodings and names
    print("[INFO] Loading known face encodings...")
    if not ENCODINGS_PATH.exists():
        print(f"[ERROR] Encodings file not found: {ENCODINGS_PATH}")
        print("[INFO] Run encode_faces.py first.")
        sys.exit(1)
    
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    
    known_encodings = data.get("encodings", [])
    known_names = data.get("names", [])
    
    if not known_encodings:
        print("[ERROR] No known encodings loaded. Exiting.")
        sys.exit(1)
    
    unique_people = set(known_names)
    print(f"[INFO] Loaded {len(known_encodings)} encoding(s) for {len(unique_people)} person(s)")
    for person in unique_people:
        count = known_names.count(person)
        print(f"      - {person}: {count} encoding(s)")
    print(f"[INFO] Confidence threshold: {args.confidence:.1%}")
    
    # Initialize face tracker if enabled
    tracker = FaceTracker() if args.track else None
    if args.track:
        print(f"[INFO] Face tracking enabled (detect every {args.detect_interval} frames)")
    
    # Check detection model
    if args.model == 'cnn':
        print("[INFO] Using CNN face detection model (GPU-accelerated if CUDA available)")
        try:
            # Test if CUDA is available
            import torch  # type: ignore
            if torch.cuda.is_available():
                print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                print("[WARN] CUDA not available - CNN will run on CPU (slower)")
        except ImportError:
            print("[WARN] PyTorch not installed - cannot verify CUDA availability")
    else:
        print("[INFO] Using HOG face detection model (CPU-optimized)")
    
    # Initialize webcam
    print(f"[INFO] Starting webcam (device {args.camera})...")
    video_capture = cv2.VideoCapture(args.camera)
    
    # Optimize webcam settings for performance
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    video_capture.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("[INFO] Webcam started. Press 'q' to quit.")
    
    frame_count = 0
    process_count = 0
    fps_start_time = cv2.getTickCount()
    fps = 0
    
    # Track last recognized names to avoid duplicate logs
    last_recognized = {}

    # Main loop
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break
        
        frame_count += 1
        
        # Skip frames for performance (if enabled)
        if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
            cv2.imshow('FACELESS - Live Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)
        
        # Calculate FPS
        process_count += 1
        if args.fps and process_count % 30 == 0:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            fps = 30 / time_diff
            fps_start_time = fps_end_time
        
        # Decide whether to do full face detection or just track
        do_detection = True
        if tracker is not None:
            if frame_count % args.detect_interval != 0:
                do_detection = False
        
        if do_detection:
            # Full face detection and recognition on small frame
            face_locations = face_recognition.face_locations(small_frame, model=args.model)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            # Scale back up face locations for display
            face_locations = [(int(top/args.scale), int(right/args.scale), 
                              int(bottom/args.scale), int(left/args.scale))
                             for (top, right, bottom, left) in face_locations]
            
            detections = []
            for location, face_encoding in zip(face_locations, face_encodings):
                # Calculate face distances (lower = better match)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                # Find best match
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                # Only recognize if confidence meets threshold
                if confidence >= args.confidence:
                    name = known_names[best_match_index]
                    
                    # Save learning sample if enabled and high confidence
                    if args.learn and confidence >= 0.85:
                        sample_key = f"{name}_{len(learning_samples.get(name, []))}"
                        if name not in learning_samples:
                            learning_samples[name] = []
                        
                        # Only save if we don't have too many samples yet
                        if len(learning_samples[name]) < 5:
                            learning_samples[name].append({
                                'encoding': face_encoding,
                                'location': location,
                                'confidence': confidence,
                                'frame': frame_count
                            })
                    
                    # Log recognition if enabled and not recently logged
                    if args.log:
                        current_time = datetime.now()
                        person_key = f"{name}_{location}"
                        
                        # Only log if not seen in last 5 seconds
                        if person_key not in last_recognized or \
                           (current_time - last_recognized[person_key]).seconds > 5:
                            log_entry = {
                                "timestamp": current_time.isoformat(),
                                "name": name,
                                "confidence": float(confidence),
                                "location": location
                            }
                            session_log.append(log_entry)
                            last_recognized[person_key] = current_time
                            print(f"[LOG] Recognized: {name} (confidence: {confidence:.1%})")
                else:
                    name = "Unknown"
                    confidence = 0.0
                
                detections.append((location, name, confidence))
            
            # Update tracker with new detections
            if tracker is not None:
                # Extract locations and names for tracker (drop confidence)
                tracker_detections = [(loc, name) for loc, name, conf in detections]
                tracked_objects = tracker.update(frame, tracker_detections)
                # Add confidence back to tracked objects
                for obj_id, det in zip(tracked_objects.keys(), detections):
                    tracked_objects[obj_id]['confidence'] = det[2]
            else:
                # No tracking - use detections directly with confidence
                tracked_objects = {i: {'location': loc, 'name': name, 'confidence': conf} 
                                 for i, (loc, name, conf) in enumerate(detections)}
        else:
            # Just update existing tracks
            if tracker is not None:
                tracked_objects = tracker.update(frame)  # type: ignore
            else:
                tracked_objects = {}
        
        # Draw results
        for object_id, data in tracked_objects.items():
            top, right, bottom, left = data['location']
            name = data['name']
            confidence = data.get('confidence', 0.0)
            
            # Color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw name with confidence
            font = cv2.FONT_HERSHEY_DUPLEX
            if name != "Unknown":
                label = f"{name} ({confidence:.0%})"
            else:
                label = "Unknown"
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       font, 0.5, (255, 255, 255), 1)
        
        # Show FPS if enabled
        if args.fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show info text
        info_text = f"Model: {args.model.upper()}"
        if args.track:
            info_text += " | Tracking: ON"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('FACELESS - Live Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q'
        if key == ord('q'):
            break
        
        # Save current frame for training on 's'
        elif key == ord('s') and args.learn:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            save_path = LEARNING_DIR / f"manual_capture_{timestamp}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"[SAVED] Frame saved to: {save_path}")
            print("        Add to known_faces/ and re-run encoder to improve recognition")
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Save learning samples if collected
    if args.learn and learning_samples:
        print(f"\n[INFO] Saving {sum(len(v) for v in learning_samples.values())} learning samples...")
        for person_name, samples in learning_samples.items():
            person_dir = LEARNING_DIR / person_name
            person_dir.mkdir(exist_ok=True)
            
            for idx, sample in enumerate(samples):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{person_name}_{timestamp}_{idx}.pkl"
                sample_path = person_dir / filename
                
                with open(str(sample_path), 'wb') as f:
                    pickle.dump({
                        'name': person_name,
                        'encoding': sample['encoding'],
                        'confidence': sample['confidence'],
                        'frame': sample['frame']
                    }, f)
            
            print(f"   {person_name}: {len(samples)} sample(s) saved")
        
        print(f"[INFO] To improve recognition, run: python improve_model.py")
    
    # Save log if enabled
    if args.log and session_log and log_file:
        with open(str(log_file), 'w') as f:
            json.dump({
                "session_start": session_log[0]['timestamp'] if session_log else None,
                "session_end": datetime.now().isoformat(),
                "total_recognitions": len(session_log),
                "recognitions": session_log
            }, f, indent=2)
        print(f"[INFO] Saved {len(session_log)} recognition(s) to {log_file}")
    
    print("[INFO] Webcam feed stopped.")


if __name__ == '__main__':
    main()
