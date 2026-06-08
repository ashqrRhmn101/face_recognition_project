# Face Recognition System — ব্যবহার গাইড

## ফাইল কাঠামো

```
face_recognition_project/
├── main.py              ← লাইভ ক্যামেরা চালু করে
├── add_person.py        ← নতুন ব্যক্তি যোগ করে
├── database.json        ← সবার তথ্য থাকে
├── known_faces/         ← এখানে ছবি থাকে (auto তৈরি)
│   ├── raju_vai/
│   │   ├── raju_vai_1.jpg
│   │   └── raju_vai_2.jpg
│   └── sumi_apu/
├── encodings_cache.pkl  ← দ্রুত লোডের জন্য cache (auto)
└── requirements.txt
```

---

## ধাপ ১ — লাইব্রেরি ইনস্টল

```bash
pip install -r requirements.txt
```

> **Windows-এ সমস্যা হলে:**
> ```bash
> pip install cmake
> pip install dlib
> pip install face-recognition
> ```

---

## ধাপ ২ — ব্যক্তি যোগ করা

```bash
python add_person.py
```

- অপশন **[1]**: ক্যামেরা দিয়ে ছবি তোলা (সবচেয়ে ভালো)
- অপশন **[2]**: বিদ্যমান ছবি ফাইল যোগ করা
- অপশন **[3]**: সব ব্যক্তির তালিকা দেখা
- অপশন **[4]**: কোনো ব্যক্তি মুছে ফেলা

**টিপস:** প্রতিজনের জন্য ৫–১০টি ছবি তুললে চেনার নির্ভুলতা বাড়ে।
ছবি তোলার সময় বিভিন্ন কোণ থেকে (সামনে, বাঁ, ডান) দেখানো ভালো।

---

## ধাপ ৩ — লাইভ ডিটেকশন চালু

```bash
python main.py
```

**চলার সময় কী-বোর্ড শর্টকাট:**
| কী | কাজ |
|---|---|
| `Q` | প্রোগ্রাম বন্ধ |
| `R` | নতুন ব্যক্তি Reload (add_person চালানোর পর) |
| `S` | Screenshot সংরক্ষণ |

---

## database.json ফরম্যাট

```json
{
  "person_id": {
    "name": "পুরো নাম",
    "age": "বয়স",
    "relation": "পরিচয়",
    "profession": "পেশা",
    "note": "যেকোনো তথ্য"
  }
}
```

`person_id` হবে `known_faces/` ফোল্ডারের ভেতরে তৈরি করা সাব-ফোল্ডারের নাম।

---

## সাধারণ সমস্যা ও সমাধান

| সমস্যা | সমাধান |
|---|---|
| মুখ চেনা যাচ্ছে না | TOLERANCE কমিয়ে ০.৪৫ করুন |
| ভুল মুখ চিনছে | TOLERANCE বাড়িয়ে ০.৫৫ করুন |
| ধীরগতি | FRAME_SKIP বাড়িয়ে ৫ করুন |
| ক্যামেরা না খুললে | VideoCapture(0) → VideoCapture(1) করুন |
| dlib install সমস্যা | Visual Studio Build Tools ইনস্টল করুন |
