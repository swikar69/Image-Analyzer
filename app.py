import os
import io
import json
import base64
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# ─── Lazy model holders ───────────────────────────────────────────────────────
_yolo_model = None
_tf_model = None
_tf_labels = None
_torch_model = None
_torch_transform = None
_color_model = None

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
)

# COCO classes for YOLO
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

ANIMAL_CLASSES = {
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"
}


def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")   # nano – downloads once (~6 MB)
    return _yolo_model


def load_tf():
    global _tf_model, _tf_labels
    if _tf_model is None:
        import tensorflow as tf
        import urllib.request
        _tf_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=True
        )
        with urllib.request.urlopen(IMAGENET_LABELS_URL) as r:
            _tf_labels = json.loads(r.read().decode())
    return _tf_model, _tf_labels


def load_torch():
    global _torch_model, _torch_transform
    if _torch_model is None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        _torch_model = models.resnet50(weights="IMAGENET1K_V1")
        _torch_model.eval()
        _torch_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return _torch_model, _torch_transform


# ─── Individual detectors ─────────────────────────────────────────────────────

def run_yolo(pil_img):
    """Object detection with bounding boxes + counts."""
    model = load_yolo()
    results = model(pil_img, verbose=False)[0]
    detections = []
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        })
        counts[label] = counts.get(label, 0) + 1
    return detections, counts


def run_mobilenet(pil_img):
    """Scene / object classification with TF MobileNetV2."""
    model, labels = load_tf()
    import tensorflow as tf
    img = pil_img.convert("RGB").resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype=np.float32)[None]
    )
    preds = model.predict(arr, verbose=0)[0]
    top5_idx = preds.argsort()[-5:][::-1]
    return [
        {"label": labels[i], "confidence": round(float(preds[i]), 3)}
        for i in top5_idx
    ]


def run_resnet(pil_img):
    """Deep classification with PyTorch ResNet-50."""
    import torch
    import urllib.request
    model, transform = load_torch()
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    probs = torch.nn.functional.softmax(out[0], dim=0)
    top5  = torch.topk(probs, 5)

    # Fetch ImageNet labels
    global _tf_labels
    if _tf_labels is None:
        with urllib.request.urlopen(IMAGENET_LABELS_URL) as r:
            _tf_labels = json.loads(r.read().decode())

    return [
        {
            "label": _tf_labels[idx] if idx < len(_tf_labels) else f"class_{idx}",
            "confidence": round(float(prob), 3),
        }
        for prob, idx in zip(top5.values, top5.indices)
    ]


def analyze_colors(pil_img):
    """Dominant color extraction via K-Means clustering."""
    from sklearn.cluster import KMeans

    img = pil_img.convert("RGB").resize((150, 150))
    pixels = np.array(img).reshape(-1, 3).astype(float)
    k = min(6, len(pixels))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(pixels)

    # count pixels per cluster → sort by prevalence
    counts   = np.bincount(km.labels_)
    centers  = km.cluster_centers_[counts.argsort()[::-1]]
    total    = pixels.shape[0]

    colors = []
    for i, (center, cnt) in enumerate(zip(centers, sorted(counts, reverse=True))):
        r, g, b = [int(v) for v in center]
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        colors.append({
            "hex": hex_col,
            "rgb": [r, g, b],
            "name": rgb_to_name(r, g, b),
            "percentage": round(cnt / total * 100, 1),
        })
    return colors


def rgb_to_name(r, g, b):
    """Map RGB to a human-readable color name."""
    names = {
        "red":    (255, 0,   0),
        "green":  (0,   128, 0),
        "blue":   (0,   0,   255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0,   128),
        "pink":   (255, 182, 193),
        "brown":  (139, 69,  19),
        "black":  (0,   0,   0),
        "white":  (255, 255, 255),
        "gray":   (128, 128, 128),
        "cyan":   (0,   255, 255),
        "lime":   (50,  205, 50),
        "navy":   (0,   0,   128),
        "teal":   (0,   128, 128),
        "maroon": (128, 0,   0),
        "beige":  (245, 245, 220),
        "coral":  (255, 127, 80),
        "gold":   (255, 215, 0),
        "silver": (192, 192, 192),
    }
    best, dist = "unknown", float("inf")
    for name, (nr, ng, nb) in names.items():
        d = (r-nr)**2 + (g-ng)**2 + (b-nb)**2
        if d < dist:
            dist, best = d, name
    return best


def classify_scene(yolo_counts, mobilenet_preds, resnet_preds):
    """Heuristic scene classifier combining all model outputs."""
    all_labels = (
        [p["label"].lower() for p in mobilenet_preds] +
        [p["label"].lower() for p in resnet_preds]
    )
    label_str = " ".join(all_labels)
    obj_str   = " ".join(yolo_counts.keys())

    scenes = {
        "street / urban": ["street","road","car","bus","traffic","sidewalk","building"],
        "nature / outdoor": ["mountain","forest","grass","sky","river","ocean","beach","tree"],
        "kitchen": ["kitchen","oven","refrigerator","microwave","sink","bowl","cup"],
        "living room": ["couch","chair","tv","remote","book","vase"],
        "sports": ["ball","bat","frisbee","skateboard","surfboard","tennis","bicycle"],
        "office / workspace": ["laptop","keyboard","mouse","monitor","desk","chair"],
        "food / dining": ["pizza","sandwich","cake","banana","apple","dining table","bottle"],
        "bedroom": ["bed","pillow","lamp","bedroom"],
        "animal habitat": ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra"],
    }
    scores = {}
    for scene, keywords in scenes.items():
        scores[scene] = sum(1 for k in keywords if k in label_str or k in obj_str)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general / indoor"


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble_analyze(pil_img):
    errors = []

    # YOLO
    try:
        yolo_dets, yolo_counts = run_yolo(pil_img)
    except Exception as e:
        yolo_dets, yolo_counts = [], {}
        errors.append(f"YOLO: {e}")

    # MobileNet
    try:
        mobilenet_preds = run_mobilenet(pil_img)
    except Exception as e:
        mobilenet_preds = []
        errors.append(f"MobileNet: {e}")

    # ResNet
    try:
        resnet_preds = run_resnet(pil_img)
    except Exception as e:
        resnet_preds = []
        errors.append(f"ResNet: {e}")

    # Colors
    try:
        colors = analyze_colors(pil_img)
    except Exception as e:
        colors = []
        errors.append(f"Colors: {e}")

    # Counts breakdown
    people_count  = yolo_counts.get("person", 0)
    animal_counts = {k: v for k, v in yolo_counts.items() if k in ANIMAL_CLASSES}
    object_counts = {
        k: v for k, v in yolo_counts.items()
        if k != "person" and k not in ANIMAL_CLASSES
    }

    # Scene
    scene = classify_scene(yolo_counts, mobilenet_preds, resnet_preds)

    # Merge top labels from both classifiers (deduplicated)
    seen = set()
    merged_classifications = []
    for p in mobilenet_preds + resnet_preds:
        lbl = p["label"]
        if lbl not in seen:
            seen.add(lbl)
            merged_classifications.append(p)

    # Build human-readable summary
    parts = []
    if people_count:
        parts.append(f"{people_count} {'person' if people_count==1 else 'people'}")
    for animal, cnt in animal_counts.items():
        parts.append(f"{cnt} {animal}{'s' if cnt>1 else ''}")
    for obj, cnt in object_counts.items():
        parts.append(f"{cnt} {obj}{'s' if cnt>1 else ''}")
    if colors:
        bg_color = colors[0]["name"]
        parts.append(f"{bg_color} dominant background")

    summary = (
        "Detected: " + ", ".join(parts) + f". Scene: {scene}."
        if parts else f"Scene classified as '{scene}'."
    )

    return {
        "summary": summary,
        "scene": scene,
        "people_count": people_count,
        "animal_counts": animal_counts,
        "object_counts": object_counts,
        "yolo_detections": yolo_dets,
        "mobilenet_top5": mobilenet_preds,
        "resnet_top5": resnet_preds,
        "dominant_colors": colors,
        "image_size": list(pil_img.size),
        "errors": errors,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    result = ensemble_analyze(pil_img)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
