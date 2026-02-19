# app.py — VisionAssistant (Flask)
# Camera stream (browser) -> send frames -> YOLO detection -> speech (TTS)
#
# Install:
#   pip install flask flask-cors ultralytics pillow gtts python-multipart
#
# Run:
#   python app.py
#
# Open on phone (same Wi-Fi):
#   http://<PC_IP>:5000

from flask import Flask, request, jsonify
from flask_cors import CORS

import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
from ultralytics import YOLO
from gtts import gTTS


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Model ----------
model = YOLO("yolov8n.pt")  # small + fast

# Classes that we consider potentially dangerous in daily navigation
DANGER = {"car", "bus", "truck", "motorcycle", "bicycle"}

# Human-friendly RU labels (extend anytime)
RU_LABELS = {
    "person": "человек",
    "car": "машина",
    "bus": "автобус",
    "truck": "грузовик",
    "motorcycle": "мотоцикл",
    "bicycle": "велосипед",
    "traffic light": "светофор",
    "stop sign": "знак стоп",
    "chair": "стул",
    "couch": "диван",
    "bed": "кровать",
    "dining table": "стол",
    "laptop": "ноутбук",
    "cell phone": "телефон",
    "tv": "телевизор",
    "bottle": "бутылка",
    "cup": "кружка",
    "book": "книга",
    "backpack": "рюкзак",
    "handbag": "сумка",
    "dog": "собака",
    "cat": "кот",
}

def ru_label(x: str) -> str:
    return RU_LABELS.get(x, x)

# Anti-repeat so it doesn't speak the same thing every frame
LAST = {"phrase": "", "time": 0.0}

def should_speak(phrase: str, cooldown_sec: float = 3.0) -> bool:
    now = time.time()
    if phrase == LAST["phrase"] and (now - LAST["time"]) < cooldown_sec:
        return False
    LAST["phrase"] = phrase
    LAST["time"] = now
    return True


# ---------- Logic: ranking / danger / distance / position ----------
def rank_objects(dets: List[Dict[str, Any]], w: int, h: int) -> List[Dict[str, Any]]:
    if not dets:
        return []
    cx_img, cy_img = w / 2, h / 2
    img_area = max(w * h, 1)

    def score(d: Dict[str, Any]) -> float:
        x1, y1, x2, y2 = d["bbox"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
        center_score = 1.0 / (1.0 + dist)

        priority = 1.35 if d["label"] in DANGER else 1.0
        return (0.6 * area + 0.4 * center_score * img_area) * d["conf"] * priority

    return sorted(dets, key=score, reverse=True)

def pick_main_and_danger(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return [main] or [main, danger] (danger preferred as second)."""
    if not ranked:
        return []
    main = ranked[0]
    danger_obj: Optional[Dict[str, Any]] = None
    for d in ranked[1:]:
        if d["label"] in DANGER and d["label"] != main["label"]:
            danger_obj = d
            break
    if danger_obj:
        return [main, danger_obj]
    # else, maybe second object if confident
    if len(ranked) > 1 and ranked[1]["conf"] >= 0.45:
        return [main, ranked[1]]
    return [main]

def pos_word(cx: float, w: int) -> str:
    x = cx / max(w, 1)
    if x < 0.35:
        return "слева"
    if x > 0.65:
        return "справа"
    return "по центру"

def distance_word(area_ratio: float) -> str:
    # crude approximation, but good enough for demo
    if area_ratio > 0.22:
        return "очень близко"
    if area_ratio > 0.10:
        return "близко"
    if area_ratio > 0.04:
        return "на среднем расстоянии"
    return "далеко"

def danger_word(label: str) -> str:
    return "опасно" if label in DANGER else "не опасно"

def describe(obj: Dict[str, Any], w: int, h: int) -> Dict[str, str]:
    x1, y1, x2, y2 = obj["bbox"]
    cx = (x1 + x2) / 2
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = area / max(w * h, 1)

    return {
        "name": ru_label(obj["label"]),
        "pos": pos_word(cx, w),
        "dist": distance_word(area_ratio),
        "danger": danger_word(obj["label"]),
    }

def make_phrase(objs: List[Dict[str, Any]], w: int, h: int) -> str:
    if not objs:
        return "Я не вижу уверенных объектов. Подойдите ближе или улучшите освещение."
    a = describe(objs[0], w, h)
    phrase = f"Перед вами {a['name']}, {a['pos']}, {a['dist']}. Это {a['danger']}."
    if len(objs) > 1:
        b = describe(objs[1], w, h)
        # если второй — опасный, подчеркнем
        extra = f"Также {b['name']}, {b['pos']}, {b['dist']}. Это {b['danger']}."
        phrase = phrase + " " + extra
    return phrase


# ---------- TTS ----------
def speak_ru(text: str) -> str:
    filename = f"say_{int(time.time() * 1000)}.mp3"
    path = AUDIO_DIR / filename
    gTTS(text=text, lang="ru").save(str(path))
    return f"/static/audio/{filename}"


# ---------- Core inference ----------
def detect_from_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    results = model.predict(img, conf=0.35, verbose=False)
    r = results[0]

    dets: List[Dict[str, Any]] = []
    if r.boxes is not None:
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            label = model.names[cls_id]
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            dets.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})

    ranked = rank_objects(dets, w, h)
    picked = pick_main_and_danger(ranked)
    phrase = make_phrase(picked, w, h)

    payload = {
        "phrase": phrase,
        "detected_count": len(dets),
        "picked": [
            {"label": d["label"], "conf": round(d["conf"], 3)}
            for d in picked
        ],
    }
    return payload


# ---------- UI (single-page demo) ----------
@app.get("/")
def index():
    # HTML embedded for quick demo (you can move it to /public later)
    return f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>VisionAssistant</title>
  <style>
    body {{ margin:0; font-family:system-ui,Arial; background:#0b0b0b; color:#fff; }}
    .wrap {{ max-width: 980px; margin: 14px auto; padding: 12px; }}
    h1 {{ margin: 8px 0 10px; font-size: 26px; }}
    .row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
    button {{
      padding: 12px 14px; border-radius: 12px; border:0; cursor:pointer;
      background:#18f39a; color:#062015; font-weight:700;
    }}
    button.secondary {{ background:#1d1d1d; color:#fff; border:1px solid rgba(255,255,255,.12); }}
    .panel {{ margin-top: 14px; background:#141414; border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:12px; }}
    video {{ width:100%; border-radius:14px; background:#000; }}
    #result {{ margin-top:10px; color: rgba(255,255,255,.85); line-height:1.4; }}
    audio {{ width:100%; margin-top:10px; display:none; }}
    .hint {{ color: rgba(255,255,255,.55); font-size: 13px; margin-top: 8px; }}
    input[type=file]{{ width:100%; padding:10px; background:#111; border-radius:12px; border:1px solid rgba(255,255,255,.12); color:#fff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>VisionAssistant — demo</h1>

    <div class="row">
      <button id="startBtn">▶ Камера</button>
      <button id="stopBtn" class="secondary">■ Стоп</button>
      <button id="voiceBtn" class="secondary">🔊 Voice ON</button>
    </div>

    <div class="panel">
      <video id="video" playsinline autoplay muted></video>
      <div class="hint">Камера: открой с телефона по Wi-Fi. Для iPhone лучше Safari.</div>
      <div id="result">Ожидаю камеру…</div>
      <audio id="audio" controls></audio>
    </div>

    <div class="panel">
      <div style="font-weight:700; margin-bottom:8px;">Загрузка фото</div>
      <input id="file" type="file" accept="image/*" capture="environment">
      <div class="row" style="margin-top:10px;">
        <button id="uploadBtn">📷 Определить по фото</button>
      </div>
      <div class="hint">Фото → распознавание → озвучка.</div>
    </div>
  </div>

<script>
let stream = null;
let timer = null;
let voiceOn = true;

const video = document.getElementById('video');
const result = document.getElementById('result');
const audio = document.getElementById('audio');

document.getElementById('voiceBtn').onclick = () => {{
  voiceOn = !voiceOn;
  document.getElementById('voiceBtn').textContent = voiceOn ? '🔊 Voice ON' : '🔇 Voice OFF';
}};

async function startCamera(){{
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "environment" }}, audio: false }});
    video.srcObject = stream;
    result.textContent = 'Камера включена. Идёт распознавание…';

    if (timer) clearInterval(timer);
    timer = setInterval(captureAndSend, 900);
  }} catch(e) {{
    result.textContent = 'Камера недоступна или доступ запрещён. Проверь разрешения браузера.';
  }}
}}

function stopCamera(){{
  if (timer) clearInterval(timer);
  timer = null;
  if (stream) {{
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }}
  result.textContent = 'Камера остановлена.';
}}

async function captureAndSend(){{
  if (!video.videoWidth) return;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.75));
  const fd = new FormData();
  fd.append('image', blob, 'frame.jpg');
  fd.append('voice', voiceOn ? '1' : '0');

  try {{
    const r = await fetch('/predict/frame', {{ method:'POST', body: fd }});
    const data = await r.json();
    result.textContent = data.phrase || 'Нет результата';

    if (voiceOn && data.audio_url) {{
      audio.src = data.audio_url;
      audio.style.display = 'block';
      audio.play().catch(()=>{{}});
    }}
  }} catch(e) {{
    // ignore transient errors
  }}
}}

document.getElementById('startBtn').onclick = startCamera;
document.getElementById('stopBtn').onclick = stopCamera;

document.getElementById('uploadBtn').onclick = async () => {{
  const input = document.getElementById('file');
  if (!input.files.length) {{
    result.textContent = 'Сначала выберите/сделайте фото.';
    return;
  }}
  const fd = new FormData();
  fd.append('image', input.files[0]);
  fd.append('voice', voiceOn ? '1' : '0');

  result.textContent = 'Распознаю фото…';

  const r = await fetch('/predict/photo', {{ method:'POST', body: fd }});
  const data = await r.json();
  result.textContent = data.phrase || 'Нет результата';
  if (voiceOn && data.audio_url) {{
    audio.src = data.audio_url;
    audio.style.display = 'block';
    audio.play().catch(()=>{{}});
  }}
}};
</script>
</body>
</html>
"""


# ---------- API: frame from camera ----------
@app.post("/predict/frame")
def predict_frame():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    voice = (request.form.get("voice") == "1")
    img_bytes = request.files["image"].read()
    if not img_bytes:
        return jsonify({"error": "empty image"}), 400

    payload = detect_from_image_bytes(img_bytes)

    # speak only if not repeating too often
    audio_url = ""
    if voice and should_speak(payload["phrase"], cooldown_sec=3.0):
        audio_url = speak_ru(payload["phrase"])

    payload["audio_url"] = audio_url
    return jsonify(payload)


# ---------- API: uploaded photo ----------
@app.post("/predict/photo")
def predict_photo():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    voice = (request.form.get("voice") == "1")
    img_bytes = request.files["image"].read()
    if not img_bytes:
        return jsonify({"error": "empty image"}), 400

    payload = detect_from_image_bytes(img_bytes)

    audio_url = ""
    if voice:
        audio_url = speak_ru(payload["phrase"])

    payload["audio_url"] = audio_url
    return jsonify(payload)


# Serve generated audio
@app.get("/static/audio/<path:filename>")
def get_audio(filename):
    # Flask will serve from filesystem automatically if you configure static folder,
    # but in this minimal file we'll rely on default static route:
    # /static/... works because Flask uses 'static' folder by default name.
    # Here kept for clarity if you later change static config.
    return app.send_static_file(f"audio/{filename}")


if __name__ == "__main__":
    # Ensure Flask uses /static from ./static
    app.static_folder = str(STATIC_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)