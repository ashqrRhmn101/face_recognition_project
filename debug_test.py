"""
debug_test.py - সমস্যা খুঁজে বের করার জন্য
এই ফাইলটা চালাও এবং output দেখাও
"""
import cv2
import numpy as np
import sys

print("="*50)
print("SYSTEM DEBUG TEST")
print("="*50)

# ১. Version চেক
print(f"\n Python: {sys.version}")
print(f" OpenCV: {cv2.__version__}")
print(f" NumPy:  {np.__version__}")

# ২. ক্যামেরা টেস্ট
print("\n[TEST 1] ক্যামেরা চেক...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-এ DSHOW দ্রুত
if not cap.isOpened():
    print("  ✗ ক্যামেরা খোলা যায়নি!")
else:
    ret, frame = cap.read()
    if ret:
        print(f"  ✓ ক্যামেরা OK — frame shape: {frame.shape}, dtype: {frame.dtype}")
    else:
        print("  ✗ ক্যামেরা থেকে ফ্রেম আসেনি!")
    cap.release()

# ৩. Image format টেস্ট
print("\n[TEST 2] Image conversion চেক...")
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print(f"  dummy shape: {dummy.shape}, dtype: {dummy.dtype}")

# পদ্ধতি ১
rgb1 = dummy[:, :, ::-1].copy()
print(f"  method1 (copy): shape={rgb1.shape}, dtype={rgb1.dtype}, contiguous={rgb1.flags['C_CONTIGUOUS']}")

# পদ্ধতি ২
rgb2 = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
print(f"  method2 (cvtColor): shape={rgb2.shape}, dtype={rgb2.dtype}, contiguous={rgb2.flags['C_CONTIGUOUS']}")

# ৪. face_recognition টেস্ট
print("\n[TEST 3] face_recognition চেক...")
try:
    import face_recognition
    print(f"  ✓ face_recognition import OK")
    
    # dummy image দিয়ে টেস্ট
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    locs = face_recognition.face_locations(test_img)
    print(f"  ✓ face_locations কাজ করছে (dummy: {locs})")
except Exception as e:
    print(f"  ✗ Error: {e}")

# ৫. saved ছবি টেস্ট
print("\n[TEST 4] Saved ছবি চেক...")
import os
known_dir = "known_faces"
if os.path.exists(known_dir):
    for person in os.listdir(known_dir):
        pdir = os.path.join(known_dir, person)
        if os.path.isdir(pdir):
            for img in os.listdir(pdir):
                if img.endswith(('.jpg','.jpeg','.png')):
                    path = os.path.join(pdir, img)
                    bgr  = cv2.imread(path)
                    if bgr is None:
                        print(f"  ✗ পড়া যায়নি: {path}")
                        continue
                    print(f"  ✓ {img}: shape={bgr.shape}, dtype={bgr.dtype}")
                    
                    # সব পদ্ধতি টেস্ট
                    try:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        import face_recognition as fr
                        encs = fr.face_encodings(rgb)
                        print(f"     → encodings: {len(encs)} পাওয়া গেছে")
                    except Exception as e:
                        print(f"     → Error: {e}")
else:
    print("  known_faces ফোল্ডার নেই!")

print("\n" + "="*50)
print("DEBUG COMPLETE — এই output টা দেখাও")
print("="*50)
