"""Improve the facial recognition model by merging learning samples into encodings.

This script takes high-confidence samples collected during recognition sessions
and adds them to the main encodings database for improved accuracy.
"""
import pickle
from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parent
LEARNING_DIR = BASE_DIR / "learning_samples"
ENCODINGS_PATH = BASE_DIR / "data" / "encodings.pickle"
BACKUP_DIR = BASE_DIR / "data" / "backups"


def main():
    parser = argparse.ArgumentParser(description='Improve facial recognition model')
    parser.add_argument('--auto', action='store_true', 
                       help='Auto mode: no prompts, auto-clear samples')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FACELESS - Model Improvement Tool")
    print("=" * 60)
    
    # Check if learning samples exist
    if not LEARNING_DIR.exists() or not list(LEARNING_DIR.glob("*/*.pkl")):
        print("\n[ERROR] No learning samples found!")
        print("[INFO] Run recognition with --learn flag first")
        sys.exit(1)
    
    # Load current encodings
    if not ENCODINGS_PATH.exists():
        print("\n[ERROR] No encodings file found!")
        print("[INFO] Run encode_faces.py first")
        sys.exit(1)
    
    print("\n[1/4] Loading current encodings...")
    with open(ENCODINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    current_encodings = data.get("encodings", [])
    current_names = data.get("names", [])
    
    print(f"   Current: {len(current_encodings)} encoding(s) for {len(set(current_names))} person(s)")
    
    # Load learning samples
    print("\n[2/4] Loading learning samples...")
    new_encodings = []
    new_names = []
    
    for person_dir in LEARNING_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        samples = list(person_dir.glob("*.pkl"))
        
        print(f"   {person_name}: {len(samples)} sample(s)")
        
        for sample_file in samples:
            with open(sample_file, 'rb') as f:
                sample_data = pickle.load(f)
                new_encodings.append(sample_data['encoding'])
                new_names.append(sample_data['name'])
    
    if not new_encodings:
        print("\n[INFO] No new samples to add")
        sys.exit(0)
    
    # Backup current encodings
    print(f"\n[3/4] Creating backup...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    backup_name = f"encodings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle"
    backup_path = BACKUP_DIR / backup_name
    
    with open(backup_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"   Backup saved: {backup_path}")
    
    # Merge encodings
    print(f"\n[4/4] Merging encodings...")
    merged_encodings = current_encodings + new_encodings
    merged_names = current_names + new_names
    
    # Save updated encodings
    updated_data = {
        "encodings": merged_encodings,
        "names": merged_names
    }
    
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(updated_data, f)
    
    print(f"\n" + "=" * 60)
    print("SUCCESS!")
    print(f"   Added {len(new_encodings)} new encoding(s)")
    print(f"   Total: {len(merged_encodings)} encoding(s)")
    print(f"   People: {', '.join(set(merged_names))}")
    for name in set(merged_names):
        count = merged_names.count(name)
        print(f"      - {name}: {count} encoding(s)")
    print(f"\n[INFO] Model improved! Recognition should be more accurate now.")
    print("=" * 60)
    
    # Ask to clear learning samples (auto-clear in auto mode)
    if args.auto:
        import shutil
        shutil.rmtree(LEARNING_DIR)
        LEARNING_DIR.mkdir()
        print("[INFO] Learning samples cleared automatically")
    else:
        print("\nClear learning samples? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(LEARNING_DIR)
            LEARNING_DIR.mkdir()
            print("[INFO] Learning samples cleared")


if __name__ == '__main__':
    main()
