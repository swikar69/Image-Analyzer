# 🧠 VisionAI — Ensemble Image Analyzer

A full-stack Python web app that analyzes images using **three ML models in ensemble**:

| Model | Framework | Task |
|-------|-----------|------|
| YOLOv8 nano | Ultralytics | Object detection + bounding boxes |
| MobileNetV2 | TensorFlow | Scene/object classification (ImageNet) |
| ResNet-50 | PyTorch | Deep classification (ImageNet) |
| K-Means | scikit-learn | Dominant color extraction |

---

## 🚀 Run locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/image-analyzer.git
cd image-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads YOLOv8 weights (~6 MB) and ImageNet labels automatically.

### 4. Run the app
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 📁 Project structure

```
image-analyzer/
├── app.py              ← Flask backend + ensemble ML pipeline
├── templates/
│   └── index.html      ← Frontend UI
├── requirements.txt
├── Procfile            ← For Railway / Render deploy
├── runtime.txt
├── railway.json
└── .gitignore
```

---

## ☁️ Deploy to Railway (Recommended — Free tier)

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**
3. Select your `image-analyzer` repo
4. Railway auto-detects Python, installs deps, and deploys
5. Click **Generate Domain** → your live URL is ready ✅

---

## ☁️ Deploy to Render (Alternative)

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Click **Deploy** ✅

---

## 🛠️ How the ensemble works

```
Image uploaded
     │
     ├──► YOLOv8        → bounding boxes, object counts, people count
     ├──► MobileNetV2   → top-5 ImageNet classifications
     ├──► ResNet-50     → top-5 ImageNet classifications (cross-validates MobileNet)
     └──► K-Means       → dominant colors (hex, name, %)
          │
          ▼
     Scene classifier (heuristic fusion of all model outputs)
          │
          ▼
     JSON response → rendered in UI
```

---

## 📡 API

### `POST /analyze`
Upload an image file:
```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@photo.jpg"
```

Response:
```json
{
  "summary": "Detected: 2 people, 1 dog, green dominant background. Scene: nature / outdoor.",
  "scene": "nature / outdoor",
  "people_count": 2,
  "animal_counts": { "dog": 1 },
  "object_counts": { "frisbee": 1 },
  "yolo_detections": [...],
  "mobilenet_top5": [...],
  "resnet_top5": [...],
  "dominant_colors": [
    { "hex": "#3a7d44", "rgb": [58,125,68], "name": "green", "percentage": 38.2 }
  ],
  "image_size": [1920, 1080],
  "errors": []
}
```

### `GET /health`
Returns `{"status": "ok"}` — use for uptime monitoring.

---

## 🧩 Tech stack

- **Backend:** Python · Flask · Flask-CORS · Gunicorn
- **ML:** YOLOv8 (Ultralytics) · TensorFlow 2.x · PyTorch 2.x · scikit-learn
- **Frontend:** Vanilla HTML/CSS/JS (no framework needed)
