"""Encode known faces from the `known_faces/` directory and save encodings to `data/encodings.pickle`.

Usage:
    python encode_faces.py

This script supports images placed directly in `known_faces/` named like `barack_obama.jpg`
or subfolders for each person `known_faces/Barack_Obama/*.jpg`.
"""
import os
import sys
import pickle
from pathlib import Path
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

import face_recognition
from PIL import Image
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
DATA_DIR = BASE_DIR / "data"
ENCODINGS_PATH = DATA_DIR / "encodings.pickle"


def find_image_files(directory: Path):
    """Find all image files in the directory."""
    exts = {".jpg", ".jpeg", ".png"}
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in exts:
                yield Path(root) / f


def person_name_from_path(path: Path, base: Path) -> str:
    """Extract person name from file path or filename."""
    # If file is inside a subdirectory immediately under known_faces, use that folder name.
    try:
        rel = path.relative_to(base)
    except Exception:
        return path.stem
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    # otherwise, use the filename prefix before an underscore if present
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def main():
    print("=" * 60)
    print("FACELESS - Face Encoder (Multi-Encoding)")
    print("=" * 60)
    
    if not KNOWN_DIR.exists():
        print(f"\n[ERROR] Known faces directory not found: {KNOWN_DIR}")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Group files by person name
    person_images = {}
    files = list(find_image_files(KNOWN_DIR))
    
    if not files:
        print(f"\n[ERROR] No images found in {KNOWN_DIR}. Add images and try again.")
        sys.exit(1)
    
    for img_path in files:
        name = person_name_from_path(img_path, KNOWN_DIR)
        if name not in person_images:
            person_images[name] = []
        person_images[name].append(img_path)
    
    print(f"\n[INFO] Found {len(files)} image(s) for {len(person_images)} person(s)")
    print(f"[INFO] TIP: Add 3-5 photos per person (different angles/lighting) for best accuracy!")

    encodings = []
    names = []
    total_faces = 0

    for person_name, img_paths in person_images.items():
        print(f"\n   Processing: {person_name} ({len(img_paths)} image(s))")
        person_encodings = 0
        
        for img_path in img_paths:
            try:
                img = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(img, model="hog")
                
                if not face_locations:
                    print(f"      ⚠ No faces in {img_path.name}")
                    continue
                    
                face_encs = face_recognition.face_encodings(img, face_locations)
                
                # Store all encodings from this image
                for enc in face_encs:
                    encodings.append(enc)
                    names.append(person_name)
                    person_encodings += 1
                    total_faces += 1
                
                print(f"      ✓ {img_path.name}: {len(face_encs)} face(s)")
            except Exception as e:
                print(f"      ✗ ERROR processing {img_path.name}: {e}")
        
        if person_encodings > 0:
            print(f"      Total: {person_encodings} encoding(s) for {person_name}")

    if not encodings:
        print("\n[ERROR] No face encodings were created. Exiting.")
        sys.exit(1)

    data = {"encodings": encodings, "names": names}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"\n" + "=" * 60)
    print(f"SUCCESS!")
    print(f"   Saved {len(encodings)} total encoding(s)")
    print(f"   People: {', '.join(set(names))}")
    for name in set(names):
        count = names.count(name)
        print(f"      - {name}: {count} encoding(s)")
    print(f"   File: {ENCODINGS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
