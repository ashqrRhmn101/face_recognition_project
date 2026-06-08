"""
Face Recognition System - main.py
Features: Face Recognition + Emotion Detection + Attendance Log
"""

import cv2
import face_recognition
import numpy as np
import json
import os
import pickle
import csv
from datetime import datetime

# DeepFace optional import
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False

# ==============================
# কনফিগারেশন
# ==============================
KNOWN_FACES_DIR  = "known_faces"
DATABASE_FILE    = "database.json"
ENCODINGS_CACHE  = "encodings_cache.pkl"
ATTENDANCE_FILE  = "attendance_log.csv"
TOLERANCE        = 0.50
FRAME_SKIP       = 2
SCALE_FACTOR     = 0.4
EMOTION_SKIP     = 15        # প্রতি N ফ্রেমে emotion update
COOLDOWN_SECONDS = 10        # একই ব্যক্তি N সেকেন্ডে একবার log

# ==============================
# রঙ
# ==============================
COLOR_KNOWN   = (0, 200, 100)
COLOR_UNKNOWN = (0, 60, 220)

EMOTION_COLORS = {
    "happy":    (0, 200, 100),
    "sad":      (200, 100, 0),
    "angry":    (0, 0, 220),
    "surprise": (0, 180, 255),
    "fear":     (130, 0, 200),
    "disgust":  (0, 140, 80),
    "neutral":  (160, 160, 160),
}

EMOTION_LABELS = {
    "happy":    "Happy",
    "sad":      "Sad",
    "angry":    "Angry",
    "surprise": "Surprised",
    "fear":     "Fear",
    "disgust":  "Disgust",
    "neutral":  "Neutral",
}


# ==============================
# Attendance Logger
# ==============================
class AttendanceLogger:
    def __init__(self):
        self.last_seen = {}   # {person_id: datetime}
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "ID", "Name", "Emotion", "Session"])

    def log(self, person_id, name, emotion="—"):
        now = datetime.now()
        last = self.last_seen.get(person_id)

        # Cooldown চেক
        if last and (now - last).total_seconds() < COOLDOWN_SECONDS:
            return False

        self.last_seen[person_id] = now
        session = now.strftime("%Y%m%d_%H")

        with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                person_id,
                name,
                emotion,
                session,
            ])

        print(f"  [LOG] {name} ({person_id}) — {now.strftime('%H:%M:%S')} — {emotion}")
        return True

    def get_today_count(self):
        today = datetime.now().strftime("%Y-%m-%d")
        count = 0
        if not os.path.exists(ATTENDANCE_FILE):
            return 0
        with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            seen = set()
            for row in reader:
                if row["Date"] == today and row["ID"] not in seen:
                    seen.add(row["ID"])
                    count += 1
        return count


# ==============================
# Helpers
# ==============================
def bgr_to_rgb(frame):
    rgb = frame[:, :, ::-1]
    rgb = np.array(rgb, dtype=np.uint8)
    rgb = np.ascontiguousarray(rgb)
    return rgb


def load_database():
    if not os.path.exists(DATABASE_FILE):
        return {}
    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_known_faces(db):
    if os.path.exists(ENCODINGS_CACHE):
        print("[INFO] Cache থেকে face encodings লোড হচ্ছে...")
        with open(ENCODINGS_CACHE, "rb") as f:
            cache = pickle.load(f)
        if len(cache["names"]) > 0:
            print(f"[INFO] {len(cache['names'])} জনের encoding লোড হয়েছে।")
            return cache["names"], cache["encodings"]
        os.remove(ENCODINGS_CACHE)

    print("[INFO] নতুন করে face encodings তৈরি হচ্ছে...")
    known_encodings, known_names = [], []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
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
                    continue
                rgb  = bgr_to_rgb(bgr)
                encs = face_recognition.face_encodings(rgb)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(person_id)
                    print(f"  ✓ {person_id} → {img_file}")
                else:
                    print(f"  ✗ মুখ পাওয়া যায়নি: {img_file}")
            except Exception as e:
                print(f"  ✗ Error [{img_file}]: {e}")

    with open(ENCODINGS_CACHE, "wb") as f:
        pickle.dump({"names": known_names, "encodings": known_encodings}, f)
    print(f"[INFO] মোট {len(known_names)} টি encoding তৈরি হয়েছে।")
    return known_names, known_encodings


def detect_emotion(face_bgr):
    """DeepFace দিয়ে emotion detect করে"""
    if not DEEPFACE_OK or face_bgr is None or face_bgr.size == 0:
        return "—", 0
    try:
        h, w = face_bgr.shape[:2]
        if h < 40 or w < 40:
            return "—", 0
        result = DeepFace.analyze(
            face_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        emotion = result.get("dominant_emotion", "neutral")
        score   = result.get("emotion", {}).get(emotion, 0)
        return emotion, round(score)
    except Exception:
        return "—", 0


# ==============================
# Drawing
# ==============================
def draw_box_corners(frame, top, right, bottom, left, color):
    L, T = 22, 3
    pts = [
        ((left,  top),    (left+L, top)),
        ((left,  top),    (left,   top+L)),
        ((right, top),    (right-L,top)),
        ((right, top),    (right,  top+L)),
        ((left,  bottom), (left+L, bottom)),
        ((left,  bottom), (left,   bottom-L)),
        ((right, bottom), (right-L,bottom)),
        ((right, bottom), (right,  bottom-L)),
    ]
    for p1, p2 in pts:
        cv2.line(frame, p1, p2, color, T)


def draw_emotion_bar(frame, top, right, bottom, left, emotion, score, e_color):
    """মুখের উপরে emotion bar"""
    bar_h = 6
    bar_y = top - bar_h - 4
    bar_w = right - left
    if bar_y < 0:
        return

    # background bar
    cv2.rectangle(frame, (left, bar_y), (right, bar_y + bar_h), (50, 50, 50), -1)
    # fill
    fill_w = int(bar_w * min(score, 100) / 100)
    if fill_w > 0:
        cv2.rectangle(frame, (left, bar_y), (left + fill_w, bar_y + bar_h), e_color, -1)

    label = f"{EMOTION_LABELS.get(emotion, emotion)}  {score}%"
    cv2.putText(frame, label, (left, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, e_color, 1, cv2.LINE_AA)


def draw_info_panel(frame, top, right, bottom, left, display_name, info_lines, color):
    lines        = [f"  {display_name}"] + [f"  {item}" for item in info_lines]
    panel_h      = 22 * len(lines) + 10
    panel_top    = bottom + 5
    panel_bottom = panel_top + panel_h

    if panel_bottom > frame.shape[0]:
        panel_top    = top - panel_h - 5
        panel_bottom = top - 5

    ov = frame.copy()
    cv2.rectangle(ov, (left, panel_top), (right, panel_bottom), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (left, panel_top), (right, panel_bottom), color, 1)

    for i, line in enumerate(lines):
        y   = panel_top + 18 + i * 22
        sc  = 0.55 if i == 0 else 0.44
        clr = color if i == 0 else (200, 200, 200)
        cv2.putText(frame, line, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, clr, 1, cv2.LINE_AA)


def draw_attendance_badge(frame, today_count, logged_names):
    """উপরে-ডানে attendance summary"""
    w = frame.shape[1]
    bx, by, bw, bh = w - 210, 48, 200, 50 + 18 * min(len(logged_names), 4)

    ov = frame.copy()
    cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 200, 100), 1)

    cv2.putText(frame, f"Today: {today_count} person(s)",
                (bx+8, by+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "Recent log:",
                (bx+8, by+34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    for i, nm in enumerate(list(logged_names)[-4:]):
        cv2.putText(frame, f"  + {nm}", (bx+8, by+50+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


def draw_hud(frame, fps, count, emotion_on):
    h, w = frame.shape[:2]
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 42), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, "FACE RECOGNITION SYSTEM", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 100), 1, cv2.LINE_AA)

    emo_txt = "Emotion: ON" if emotion_on else "Emotion: OFF [E]"
    emo_clr = (0, 200, 100) if emotion_on else (100, 100, 100)
    cv2.putText(frame, emo_txt, (w//2 - 60, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, emo_clr, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS:{fps:.0f}  Det:{count}  {now}",
                (w - 320, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-28), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "  [Q] Quit   [R] Reload   [S] Screenshot   [E] Emotion toggle   [L] Show log",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)


# ==============================
# Main
# ==============================
def main():
    print("\n" + "="*55)
    print("   FACE RECOGNITION + EMOTION + ATTENDANCE")
    print("="*55 + "\n")

    if not DEEPFACE_OK:
        print("[WARNING] DeepFace নেই — emotion detection বন্ধ।")
        print("          চালু করতে: pip install deepface tf-keras\n")

    db = load_database()
    known_names, known_encodings = load_known_faces(db)

    if not known_encodings:
        print("[WARNING] কোনো known face নেই! add_person.py দিয়ে যোগ করুন.\n")

    logger = AttendanceLogger()
    print(f"[INFO] Attendance log: {ATTENDANCE_FILE}\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] ক্যামেরা খোলা যায়নি!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INFO] ক্যামেরা চালু! [Q] চাপুন বন্ধ করতে.\n")

    frame_count    = 0
    face_locations = []
    face_names     = []
    face_emotions  = {}   # {face_idx: (emotion, score)}
    emotion_on     = DEEPFACE_OK
    logged_names   = []   # recent log list
    prev_time      = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now_time  = datetime.now()
        fps       = 1.0 / max((now_time - prev_time).total_seconds(), 0.001)
        prev_time = now_time

        # ── Face detection ──────────────────────────────
        if frame_count % FRAME_SKIP == 0:
            small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb   = bgr_to_rgb(small)
            try:
                face_locations = face_recognition.face_locations(rgb, model="hog")
                if known_encodings and face_locations:
                    encs_list  = face_recognition.face_encodings(rgb, face_locations)
                    face_names = []
                    for enc in encs_list:
                        dists    = face_recognition.face_distance(known_encodings, enc)
                        best     = int(np.argmin(dists))
                        face_names.append(known_names[best] if dists[best] < TOLERANCE else "Unknown")
                else:
                    face_names = ["Unknown"] * len(face_locations)
            except Exception as e:
                face_locations, face_names = [], []

        # ── Emotion detection ────────────────────────────
        if emotion_on and frame_count % EMOTION_SKIP == 0:
            scale = int(1 / SCALE_FACTOR)
            for i, (top, right, bottom, left) in enumerate(face_locations):
                t, r, b, l = top*scale, right*scale, bottom*scale, left*scale
                pad   = 10
                face_crop = frame[max(0,t-pad):b+pad, max(0,l-pad):r+pad]
                emotion, score = detect_emotion(face_crop)
                face_emotions[i] = (emotion, score)

        # ── Attendance log ───────────────────────────────
        scale = int(1 / SCALE_FACTOR)
        for i, name in enumerate(face_names):
            if name != "Unknown":
                person   = db.get(name, {})
                emo, _   = face_emotions.get(i, ("—", 0))
                disp_name = person.get("name", name)
                logged = logger.log(name, disp_name, emo)
                if logged:
                    logged_names.append(disp_name)

        # ── Draw ─────────────────────────────────────────
        for i, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
            t, r, b, l = top*scale, right*scale, bottom*scale, left*scale

            color   = COLOR_KNOWN if name != "Unknown" else COLOR_UNKNOWN
            person  = db.get(name, {})
            disp    = person.get("name", name) if name != "Unknown" else "Unknown"

            emo, score = face_emotions.get(i, ("neutral", 0))
            e_color    = EMOTION_COLORS.get(emo, (160, 160, 160))

            # মুখের বক্স
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            draw_box_corners(frame, t, r, b, l, color)

            # Emotion bar (উপরে)
            if emotion_on and emo != "—":
                draw_emotion_bar(frame, t, r, b, l, emo, score, e_color)

            # Info panel (নিচে)
            info_lines = []
            if person:
                if person.get("age"):      info_lines.append(f"Age: {person['age']} yrs")
                if person.get("relation"): info_lines.append(f"Relation: {person['relation']}")
                if person.get("profession"): info_lines.append(f"Job: {person['profession']}")
            else:
                info_lines = ["Not in database"]

            if emotion_on and emo != "—":
                info_lines.append(f"Emotion: {EMOTION_LABELS.get(emo, emo)} {score}%")

            draw_info_panel(frame, t, r, b, l, disp, info_lines, color)

        # ── HUD & Badge ──────────────────────────────────
        draw_hud(frame, fps, len(face_locations), emotion_on)
        if logged_names:
            draw_attendance_badge(frame, logger.get_today_count(), logged_names)

        cv2.imshow("Face Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n[INFO] Reloading...")
            if os.path.exists(ENCODINGS_CACHE): os.remove(ENCODINGS_CACHE)
            db = load_database()
            known_names, known_encodings = load_known_faces(db)
        elif key == ord('s'):
            fn = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[INFO] Screenshot: {fn}")
        elif key == ord('e'):
            emotion_on = not emotion_on and DEEPFACE_OK
            print(f"[INFO] Emotion: {'ON' if emotion_on else 'OFF'}")
        elif key == ord('l'):
            print(f"\n[LOG] Attendance file: {os.path.abspath(ATTENDANCE_FILE)}")
            print(f"[LOG] Today count: {logger.get_today_count()}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] প্রোগ্রাম বন্ধ। Log file: {ATTENDANCE_FILE}")


if __name__ == "__main__":
    main()
