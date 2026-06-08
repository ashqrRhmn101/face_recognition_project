"""
Face Recognition System - main.py
লাইভ ক্যামেরায় মুখ শনাক্ত করে নাম, বয়স ও তথ্য দেখায়
"""

import cv2
import face_recognition
import numpy as np
import json
import os
import pickle
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE   = "database.json"
ENCODINGS_CACHE = "encodings_cache.pkl"
TOLERANCE    = 0.50
FRAME_SKIP   = 3
SCALE_FACTOR = 0.5

COLOR_KNOWN   = (0, 200, 100)
COLOR_UNKNOWN = (0, 60, 220)


def bgr_to_rgb(frame):
    """
    OpenCV BGR → face_recognition-এর জন্য সঠিক uint8 RGB
    numpy contiguous array ব্যবহার করে dlib-এর জন্য
    """
    rgb = frame[:, :, ::-1]                  # BGR → RGB (channel flip)
    rgb = np.array(rgb, dtype=np.uint8)      # uint8 নিশ্চিত
    rgb = np.ascontiguousarray(rgb)          # dlib-এর জন্য C-contiguous
    return rgb


def load_database():
    if not os.path.exists(DATABASE_FILE):
        return {}
    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_known_faces(db):
    # Cache চেক
    if os.path.exists(ENCODINGS_CACHE):
        print("[INFO] Cache থেকে face encodings লোড হচ্ছে...")
        with open(ENCODINGS_CACHE, "rb") as f:
            cache = pickle.load(f)
        names = cache["names"]
        encs  = cache["encodings"]
        if len(names) > 0:
            print(f"[INFO] {len(names)} জনের encoding লোড হয়েছে।")
            return names, encs
        else:
            print("[INFO] Cache খালি, নতুন করে তৈরি হচ্ছে...")
            os.remove(ENCODINGS_CACHE)

    print("[INFO] নতুন করে face encodings তৈরি হচ্ছে...")
    known_encodings = []
    known_names     = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"[WARNING] '{KNOWN_FACES_DIR}' ফোল্ডার তৈরি হয়েছে।")
        print("[WARNING] add_person.py দিয়ে ব্যক্তি যোগ করুন।")
        return known_names, known_encodings

    for person_id in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue

        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, img_file)
            try:
                bgr = cv2.imread(img_path)
                if bgr is None:
                    print(f"  ✗ পড়া যায়নি: {img_path}")
                    continue

                rgb = bgr_to_rgb(bgr)
                encodings = face_recognition.face_encodings(rgb)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_id)
                    print(f"  ✓ {person_id} → {img_file}")
                else:
                    print(f"  ✗ মুখ পাওয়া যায়নি: {img_file}")

            except Exception as e:
                print(f"  ✗ Error [{img_file}]: {e}")

    # Cache সংরক্ষণ
    with open(ENCODINGS_CACHE, "wb") as f:
        pickle.dump({"names": known_names, "encodings": known_encodings}, f)

    print(f"[INFO] মোট {len(known_names)} টি encoding তৈরি হয়েছে।")
    return known_names, known_encodings


def draw_info_box(frame, top, right, bottom, left, name, info, color):
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    length, thickness = 20, 3
    for pt1, pt2 in [
        ((left, top),    (left+length, top)),
        ((left, top),    (left, top+length)),
        ((right, top),   (right-length, top)),
        ((right, top),   (right, top+length)),
        ((left, bottom), (left+length, bottom)),
        ((left, bottom), (left, bottom-length)),
        ((right, bottom),(right-length, bottom)),
        ((right, bottom),(right, bottom-length)),
    ]:
        cv2.line(frame, pt1, pt2, color, thickness)

    lines        = [f"  {name}"] + [f"  {item}" for item in info]
    panel_top    = bottom + 5
    panel_bottom = panel_top + 22 * len(lines) + 10

    # ক্যামেরার বাইরে গেলে উপরে দেখানো
    h = frame.shape[0]
    if panel_bottom > h:
        panel_top    = top - (22 * len(lines) + 15)
        panel_bottom = top - 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, panel_top), (right, panel_bottom), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (left, panel_top), (right, panel_bottom), color, 1)

    for i, line in enumerate(lines):
        y   = panel_top + 18 + i * 22
        sc  = 0.55 if i == 0 else 0.45
        clr = color if i == 0 else (200, 200, 200)
        cv2.putText(frame, line, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, clr, 1, cv2.LINE_AA)


def draw_hud(frame, fps, count):
    h, w = frame.shape[:2]
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 40), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "FACE RECOGNITION SYSTEM", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Detected: {count}  |  {now}",
                (w - 430, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-30), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "  [Q] বন্ধ   [R] Reload   [S] Screenshot",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)


def main():
    print("\n" + "="*50)
    print("   FACE RECOGNITION SYSTEM চালু হচ্ছে...")
    print("="*50 + "\n")

    db = load_database()
    known_names, known_encodings = load_known_faces(db)

    if not known_encodings:
        print("[WARNING] কোনো known face নেই!")
        print("[WARNING] add_person.py চালিয়ে ব্যক্তি যোগ করুন।\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] ক্যামেরা খোলা যায়নি!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[INFO] ক্যামেরা চালু! [Q] চাপুন বন্ধ করতে.\n")

    frame_count    = 0
    face_locations = []
    face_names     = []
    prev_time      = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now_time = datetime.now()
        fps      = 1.0 / max((now_time - prev_time).total_seconds(), 0.001)
        prev_time = now_time

        # face detection — প্রতি FRAME_SKIP ফ্রেমে
        if frame_count % FRAME_SKIP == 0:
            small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb   = bgr_to_rgb(small)

            try:
                face_locations = face_recognition.face_locations(rgb, model="hog")

                if known_encodings and face_locations:
                    face_encodings_list = face_recognition.face_encodings(rgb, face_locations)
                    face_names = []
                    for enc in face_encodings_list:
                        distances = face_recognition.face_distance(known_encodings, enc)
                        best_idx  = int(np.argmin(distances))
                        if distances[best_idx] < TOLERANCE:
                            face_names.append(known_names[best_idx])
                        else:
                            face_names.append("Unknown")
                else:
                    face_names = ["Unknown"] * len(face_locations)

            except Exception as e:
                print(f"[WARNING] Detection error: {e}")
                face_locations = []
                face_names     = []

        # ফলাফল আঁকা
        scale = int(1 / SCALE_FACTOR)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top    *= scale; right  *= scale
            bottom *= scale; left   *= scale

            person = db.get(name, {})
            info_lines = []
            if person:
                if person.get("age"):        info_lines.append(f"Age: {person['age']} yrs")
                if person.get("relation"):   info_lines.append(f"Relation: {person['relation']}")
                if person.get("profession"): info_lines.append(f"Profession: {person['profession']}")
                if person.get("note"):       info_lines.append(f"Note: {person['note']}")
            else:
                info_lines = ["অপরিচিত ব্যক্তি"]

            color        = COLOR_KNOWN if name != "Unknown" else COLOR_UNKNOWN
            display_name = person.get("name", name) if name != "Unknown" else "অপরিচিত"
            draw_info_box(frame, top, right, bottom, left, display_name, info_lines, color)

        draw_hud(frame, fps, len(face_locations))
        cv2.imshow("Face Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n[INFO] Reloading...")
            if os.path.exists(ENCODINGS_CACHE):
                os.remove(ENCODINGS_CACHE)
            db = load_database()
            known_names, known_encodings = load_known_faces(db)
        elif key == ord('s'):
            fn = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[INFO] Screenshot সংরক্ষিত: {fn}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] প্রোগ্রাম বন্ধ হয়েছে।")


if __name__ == "__main__":
    main()
