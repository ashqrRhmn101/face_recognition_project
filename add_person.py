"""
add_person.py — নতুন ব্যক্তি যোগ করার স্ক্রিপ্ট
"""

import cv2
import face_recognition
import json
import os
import pickle
import shutil
import numpy as np

KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE   = "database.json"
ENCODINGS_CACHE = "encodings_cache.pkl"


def load_database():
    if not os.path.exists(DATABASE_FILE):
        return {}
    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_database(db):
    with open(DATABASE_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print("[INFO] Database সংরক্ষিত হয়েছে।")


def invalidate_cache():
    if os.path.exists(ENCODINGS_CACHE):
        os.remove(ENCODINGS_CACHE)
        print("[INFO] Encoding cache রিসেট হয়েছে।")


def bgr_to_rgb(frame):
    """
    OpenCV BGR frame → face_recognition-এর জন্য সঠিক uint8 RGB
    numpy ব্যবহার করে নিশ্চিত conversion
    """
    # চ্যানেল উল্টো করে RGB বানানো (BGR → RGB)
    rgb = frame[:, :, ::-1]
    # numpy array হিসেবে uint8 নিশ্চিত করা
    rgb = np.array(rgb, dtype=np.uint8)
    # C-contiguous memory layout নিশ্চিত করা (dlib এটা চায়)
    rgb = np.ascontiguousarray(rgb)
    return rgb


def find_faces(frame):
    """frame থেকে face location বের করে, error হলে খালি list"""
    try:
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = bgr_to_rgb(small)
        locs  = face_recognition.face_locations(rgb, model="hog")
        # scale back to original size
        return [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locs]
    except Exception as e:
        return []


def capture_photos_from_camera(person_id, count=5):
    save_dir = os.path.join(KNOWN_FACES_DIR, person_id)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] ক্যামেরা খোলা যায়নি!")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured   = 0
    frame_tick = 0
    locs       = []

    print(f"\n[INFO] ক্যামেরা চালু। {count}টি ছবি তুলতে হবে।")
    print("       মুখ সরাসরি ক্যামেরার দিকে রাখুন।")
    print("       [SPACE] চাপুন ছবি তুলতে | [Q] বাতিল করতে\n")

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] ক্যামেরা থেকে ফ্রেম পাওয়া যায়নি!")
            break

        display    = frame.copy()
        frame_tick += 1

        # প্রতি ৫ ফ্রেমে face detection
        if frame_tick % 5 == 0:
            locs = find_faces(frame)

        face_found = len(locs) > 0

        # মুখের বক্স আঁকা
        for (top, right, bottom, left) in locs:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 200, 100), 2)

        # স্ট্যাটাস
        status_color = (0, 200, 100) if face_found else (0, 60, 220)
        status_text  = (f"মুখ পাওয়া গেছে! ({captured}/{count})"
                        if face_found else "মুখ খুঁজছি... সামনে আসুন")

        cv2.rectangle(display, (0, 0), (display.shape[1], 45), (20, 20, 20), -1)
        cv2.putText(display, status_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        hint = f"[SPACE] ছবি তুলুন  |  [Q] বাতিল  |  ID: {person_id}  |  {captured}/{count}"
        cv2.putText(display, hint, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(f"ছবি তোলা — {person_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] বাতিল করা হয়েছে।")
            cap.release()
            cv2.destroyAllWindows()
            return False

        elif key == ord(' '):
            filename = os.path.join(save_dir, f"{person_id}_{captured + 1}.jpg")
            cv2.imwrite(filename, frame)
            captured += 1
            print(f"  ✓ ছবি {captured}/{count} সংরক্ষিত → {filename}")

            # ফ্ল্যাশ ইফেক্ট
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 30)
            cv2.imshow(f"ছবি তোলা — {person_id}", flash)
            cv2.waitKey(300)

    cap.release()
    cv2.destroyAllWindows()

    if captured > 0:
        print(f"\n[INFO] {captured}টি ছবি সংরক্ষিত হয়েছে।")
    return captured > 0


def add_from_existing_image(person_id, image_path):
    image_path = image_path.strip().strip('"').strip("'")
    if not os.path.exists(image_path):
        print(f"[ERROR] ছবি পাওয়া যায়নি: {image_path}")
        return False

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[ERROR] ছবি পড়া যায়নি: {image_path}")
        return False

    rgb = bgr_to_rgb(bgr)

    try:
        encodings = face_recognition.face_encodings(rgb)
    except Exception as e:
        print(f"[ERROR] Encoding error: {e}")
        return False

    if not encodings:
        print(f"[ERROR] এই ছবিতে মুখ পাওয়া যায়নি।")
        return False

    save_dir = os.path.join(KNOWN_FACES_DIR, person_id)
    os.makedirs(save_dir, exist_ok=True)
    ext      = os.path.splitext(image_path)[1]
    existing = len([f for f in os.listdir(save_dir) if f.endswith(('.jpg','.jpeg','.png'))])
    dest     = os.path.join(save_dir, f"{person_id}_{existing + 1}{ext}")
    shutil.copy2(image_path, dest)
    print(f"  ✓ ছবি যোগ হয়েছে → {dest}")
    return True


def list_persons():
    db = load_database()
    if not db:
        print("  (কেউ নেই — add_person.py দিয়ে যোগ করুন)")
        return
    print(f"\n  {'ID':<20} {'নাম':<20} {'বয়স':<8} {'পরিচয়'}")
    print("  " + "-" * 60)
    for pid, info in db.items():
        print(f"  {pid:<20} {info.get('name','—'):<20} "
              f"{info.get('age','—'):<8} {info.get('relation','—')}")


def delete_person(person_id):
    db = load_database()
    if person_id not in db:
        print(f"[ERROR] '{person_id}' ডেটাবেসে নেই।")
        return
    confirm = input(f"  সত্যিই '{person_id}' মুছে ফেলবেন? (yes/no): ")
    if confirm.lower() != "yes":
        print("  বাতিল।")
        return
    del db[person_id]
    save_database(db)
    person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
    invalidate_cache()
    print(f"  ✓ '{person_id}' মুছে ফেলা হয়েছে।")


def main_menu():
    print("\n" + "="*50)
    print("   FACE RECOGNITION — ব্যক্তি পরিচালনা")
    print("="*50)

    while True:
        print("\n  [1] নতুন ব্যক্তি যোগ করুন (ক্যামেরা)")
        print("  [2] নতুন ব্যক্তি যোগ করুন (ছবি ফাইল)")
        print("  [3] সব ব্যক্তির তালিকা")
        print("  [4] ব্যক্তি মুছুন")
        print("  [5] বের হন")
        choice = input("\n  বেছে নিন: ").strip()

        if choice == "1":
            print("\n  — নতুন ব্যক্তির তথ্য দিন —")
            person_id  = input("  ID (ইংরেজিতে, যেমন: raju_vai): ").strip()
            name       = input("  পুরো নাম: ").strip()
            age        = input("  বয়স: ").strip()
            relation   = input("  পরিচয় (বন্ধু/ভাই/বাবা): ").strip()
            profession = input("  পেশা (ঐচ্ছিক): ").strip()
            note       = input("  নোট (ঐচ্ছিক): ").strip()
            pc         = input("  কয়টি ছবি তুলবেন? [5]: ").strip()
            photo_count = int(pc) if pc.isdigit() else 5

            if not person_id or not name:
                print("  [ERROR] ID ও নাম আবশ্যক।")
                continue

            success = capture_photos_from_camera(person_id, count=photo_count)
            if not success:
                continue

            db = load_database()
            db[person_id] = {
                "name": name, "age": age,
                "relation": relation,
                "profession": profession,
                "note": note
            }
            save_database(db)
            invalidate_cache()
            print(f"\n  ✓ '{name}' সফলভাবে যোগ হয়েছে!")
            print("  এখন main.py চালালে চেনা যাবে।")

        elif choice == "2":
            print("\n  — ছবি ফাইল থেকে যোগ করুন —")
            person_id  = input("  ID: ").strip()
            name       = input("  পুরো নাম: ").strip()
            age        = input("  বয়স: ").strip()
            relation   = input("  পরিচয়: ").strip()
            profession = input("  পেশা (ঐচ্ছিক): ").strip()
            note       = input("  নোট (ঐচ্ছিক): ").strip()
            imgs_input = input("  ছবির path (কমা দিয়ে একাধিক): ").strip()

            added = sum(
                add_from_existing_image(person_id, p)
                for p in imgs_input.split(",")
            )
            if added > 0:
                db = load_database()
                db[person_id] = {
                    "name": name, "age": age,
                    "relation": relation,
                    "profession": profession,
                    "note": note
                }
                save_database(db)
                invalidate_cache()
                print(f"\n  ✓ '{name}' — {added}টি ছবি সহ যোগ হয়েছে!")

        elif choice == "3":
            list_persons()

        elif choice == "4":
            pid = input("  মুছতে চান কোন ID? ").strip()
            delete_person(pid)

        elif choice == "5":
            print("  বিদায়!")
            break
        else:
            print("  সঠিক অপশন বেছে নিন।")


if __name__ == "__main__":
    main_menu()
