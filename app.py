"""
Image Fusion for Improved Situational Awareness in Remote Sensing Operations
Flask Backend - Advanced Image Processing & Fusion Engine
"""

import os
import io
import json
import uuid
import base64
import random
import math
import time
import tempfile
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import cv2

from fusion import run_fusion

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
FUSED_FOLDER  = os.path.join(os.path.dirname(__file__), 'fused')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FUSED_FOLDER,  exist_ok=True)


missions       = {}
fusion_history = []
threat_log     = []
sensor_feeds   = {}

def generate_thermal_overlay(img_array):
    """Simulate thermal imaging overlay"""
    gray = np.mean(img_array[:, :, :3], axis=2)
    thermal = np.zeros_like(img_array[:, :, :3], dtype=np.float64)
    norm = gray / 255.0
    thermal[:, :, 0] = np.clip(norm * 2.0, 0, 1) * 255
    thermal[:, :, 1] = np.clip((norm - 0.3) * 2.0, 0, 1) * 255
    thermal[:, :, 2] = np.clip((norm - 0.6) * 3.0, 0, 1) * 255
    noise = np.random.normal(0, 8, thermal.shape)
    thermal = np.clip(thermal + noise, 0, 255)
    return thermal.astype(np.uint8)

def generate_ir_overlay(img_array):
    """Simulate infrared imaging"""
    gray = np.mean(img_array[:, :, :3], axis=2)
    ir = np.zeros_like(img_array[:, :, :3], dtype=np.float64)
    norm = gray / 255.0
    inverted = 1.0 - norm
    ir[:, :, 0] = inverted * 80
    ir[:, :, 1] = np.clip(inverted * 1.5, 0, 1) * 200
    ir[:, :, 2] = inverted * 120
    noise = np.random.normal(0, 5, ir.shape)
    ir = np.clip(ir + noise, 0, 255)
    return ir.astype(np.uint8)

def generate_sar_overlay(img_array):
    """Simulate Synthetic Aperture Radar"""
    gray = np.mean(img_array[:, :, :3], axis=2)
    sar = np.zeros_like(img_array[:, :, :3], dtype=np.float64)
    edges = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges += np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edges = edges / edges.max() if edges.max() > 0 else edges
    speckle = np.random.exponential(1.0, gray.shape)
    sar_val = (edges * 0.6 + (gray / 255.0) * 0.4) * speckle
    sar_val = np.clip(sar_val / (sar_val.max() if sar_val.max() > 0 else 1), 0, 1)
    sar[:, :, 0] = sar_val * 200
    sar[:, :, 1] = sar_val * 220
    sar[:, :, 2] = sar_val * 180
    return sar.astype(np.uint8)

def generate_multispectral_overlay(img_array):
    """Simulate multispectral imaging (enhanced vegetation/terrain)"""
    ms = img_array[:, :, :3].astype(np.float64)
    ms[:, :, 1] = np.clip(ms[:, :, 1] * 1.4, 0, 255)
    ms[:, :, 0] = np.clip(ms[:, :, 0] * 0.8 + 30, 0, 255)
    ms[:, :, 2] = np.clip(ms[:, :, 2] * 1.2, 0, 255)
    return ms.astype(np.uint8)

def fuse_weighted_average(images, weights=None):
    if weights is None:
        weights = [1.0 / len(images)] * len(images)
    result = np.zeros_like(images[0], dtype=np.float64)
    for img, w in zip(images, weights):
        result += img.astype(np.float64) * w
    return np.clip(result, 0, 255).astype(np.uint8)

def fuse_maximum(images):
    result = images[0].copy().astype(np.float64)
    for img in images[1:]:
        result = np.maximum(result, img.astype(np.float64))
    return result.astype(np.uint8)

def fuse_pca(images):
    stacked = np.stack([img.astype(np.float64).mean(axis=2) for img in images])
    n = stacked.shape[0]
    reshaped = stacked.reshape(n, -1)
    mean = reshaped.mean(axis=1, keepdims=True)
    centered = reshaped - mean
    cov = np.cov(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]
    weights = np.abs(principal) / np.abs(principal).sum()
    return fuse_weighted_average(images, weights.tolist())

def fuse_laplacian(images):
    pil_images = [Image.fromarray(img) for img in images]
    details, bases = [], []
    for pil_img in pil_images:
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=5))
        detail = np.array(pil_img).astype(np.float64) - np.array(blurred).astype(np.float64)
        details.append(detail)
        bases.append(np.array(blurred).astype(np.float64))
    fused_detail = details[0].copy()
    for d in details[1:]:
        mask = np.abs(d) > np.abs(fused_detail)
        fused_detail[mask] = d[mask]
    fused_base = np.mean(bases, axis=0)
    return np.clip(fused_base + fused_detail, 0, 255).astype(np.uint8)

PIL_FUSION_METHODS = {
    'weighted_average': fuse_weighted_average,
    'maximum':          fuse_maximum,
    'pca':              fuse_pca,
    'laplacian':        fuse_laplacian,
}

ALL_METHODS = set(PIL_FUSION_METHODS.keys()) | {'wavelet'}

SENSOR_GENERATORS = {
    'thermal':       generate_thermal_overlay,
    'infrared':      generate_ir_overlay,
    'sar':           generate_sar_overlay,
    'multispectral': generate_multispectral_overlay,
}


def detect_threats(fused_array):
    """Simulate threat detection on fused image"""
    h, w = fused_array.shape[:2]
    threats = []
    num_threats = random.randint(1, 5)
    threat_types = [
        {"type": "Vehicle",             "level": "medium",   "icon": "🚗"},
        {"type": "Personnel",           "level": "low",      "icon": "🚶"},
        {"type": "Structure",           "level": "low",      "icon": "🏠"},
        {"type": "Armed Vehicle",       "level": "high",     "icon": "🔴"},
        {"type": "Aircraft",            "level": "critical", "icon": "✈️"},
        {"type": "Encampment",          "level": "medium",   "icon": "⛺"},
        {"type": "Radar Installation",  "level": "high",     "icon": "📡"},
        {"type": "Supply Depot",        "level": "medium",   "icon": "📦"},
    ]
    for _ in range(num_threats):
        tt = random.choice(threat_types)
        threats.append({
            "id":         str(uuid.uuid4())[:8],
            "type":       tt["type"],
            "level":      tt["level"],
            "icon":       tt["icon"],
            "x":          random.randint(int(w * 0.1), int(w * 0.9)),
            "y":          random.randint(int(h * 0.1), int(h * 0.9)),
            "confidence": round(random.uniform(0.65, 0.99), 2),
            "timestamp":  datetime.now().isoformat(),
        })
    return threats

def generate_terrain_analysis(img_array):
    """Analyze terrain from fused image"""
    gray    = np.mean(img_array[:, :, :3], axis=2)
    mean_val = float(np.mean(gray))
    std_val  = float(np.std(gray))
    terrain_types = {
        "urban":        round(random.uniform(10, 35), 1),
        "vegetation":   round(random.uniform(15, 40), 1),
        "water":        round(random.uniform(2,  15), 1),
        "bare_soil":    round(random.uniform(5,  20), 1),
        "road_network": round(random.uniform(3,  12), 1),
    }
    total = sum(terrain_types.values())
    terrain_types = {k: round(v / total * 100, 1) for k, v in terrain_types.items()}
    return {
        "terrain_composition":      terrain_types,
        "avg_elevation_estimate":   round(mean_val * 2.5 + random.uniform(-50, 50), 1),
        "roughness_index":          round(std_val / 50, 2),
        "visibility_score":         round(random.uniform(0.5, 1.0), 2),
        "cover_rating":             random.choice(["Minimal", "Moderate", "Good", "Excellent"]),
        "traversability":           random.choice(["Easy", "Moderate", "Difficult", "Impassable"]),
    }

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload source image(s)"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    img_id = str(uuid.uuid4())[:12]
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'png'
    filename = f"{img_id}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    img = Image.open(filepath).convert('RGB')
    img_array = np.array(img)
    sensor_data = {}
    for sensor_name, generator in SENSOR_GENERATORS.items():
        overlay          = generator(img_array)
        overlay_filename = f"{img_id}_{sensor_name}.png"
        overlay_path     = os.path.join(UPLOAD_FOLDER, overlay_filename)
        Image.fromarray(overlay).save(overlay_path)
        sensor_data[sensor_name] = overlay_filename
    sensor_feeds[img_id] = {
        "original":    filename,
        "sensors":     sensor_data,
        "width":       img.width,
        "height":      img.height,
        "upload_time": datetime.now().isoformat(),
    }
    return jsonify({
        "id":       img_id,
        "filename": filename,
        "width":    img.width,
        "height":   img.height,
        "sensors":  list(sensor_data.keys()),
        "message":  "Image uploaded and sensor overlays generated",
    })

@app.route('/api/fuse', methods=['POST'])
def fuse_images():
    """Perform image fusion"""
    data    = request.json
    img_id  = data.get('image_id')
    method  = data.get('method', 'weighted_average')
    sensors = data.get('sensors', ['thermal', 'infrared'])
    enhance = data.get('enhance', True)

    if img_id not in sensor_feeds:
        return jsonify({"error": "Image not found"}), 404
    if method not in ALL_METHODS:
        return jsonify({"error": f"Unknown fusion method: {method}"}), 400

    start_time = time.time()
    feed = sensor_feeds[img_id]
    original_path = os.path.join(UPLOAD_FOLDER, feed['original'])

    if method == 'wavelet':
        sar_key = next(
            (s for s in ('sar', 'infrared', 'thermal') if s in feed['sensors']),
            None,
        )
        if sar_key:
            sar_path = os.path.join(UPLOAD_FOLDER, feed['sensors'][sar_key])
        else:
            sar_path = original_path   

        try:
            result = run_fusion(
                eo_path=original_path,
                sar_path=sar_path,
                output_base=os.path.join(os.path.dirname(__file__), 'output'),
                save_graphs=False,          
            )
        except Exception as exc:
            return jsonify({"error": f"Fusion engine error: {exc}"}), 500

        fused_array = result["fused_array"]        
        scene_mode  = result["scene_mode"]
        extra_metrics = {
            "psnr_wavelet": result["psnr_wavelet"],
            "ssim_wavelet": result["ssim_wavelet"],
            "psnr_dl":      result["psnr_dl"],
            "ssim_dl":      result["ssim_dl"],
            "psnr_final":   result["psnr_final"],
            "ssim_final":   result["ssim_final"],
        }

    else:
        original = np.array(Image.open(original_path).convert('RGB'))
        images_to_fuse = [original]
        for sensor in sensors:
            if sensor in feed['sensors']:
                sensor_path = os.path.join(UPLOAD_FOLDER, feed['sensors'][sensor])
                images_to_fuse.append(np.array(Image.open(sensor_path).convert('RGB')))

        fused_rgb = PIL_FUSION_METHODS[method](images_to_fuse)

        if enhance:
            pil_img  = Image.fromarray(fused_rgb)
            pil_img  = ImageEnhance.Contrast(pil_img).enhance(1.2)
            pil_img  = ImageEnhance.Sharpness(pil_img).enhance(1.3)
            fused_rgb = np.array(pil_img)

        fused_array   = cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR)
        scene_mode    = None
        extra_metrics = {}

    processing_time = time.time() - start_time

    fused_id       = str(uuid.uuid4())[:12]
    fused_filename = f"fused_{fused_id}.png"
    fused_path     = os.path.join(FUSED_FOLDER, fused_filename)
    cv2.imwrite(fused_path, fused_array)

    fused_rgb_out = cv2.cvtColor(fused_array, cv2.COLOR_BGR2RGB)
    threats  = detect_threats(fused_rgb_out)
    terrain  = generate_terrain_analysis(fused_rgb_out)

    fa_f = fused_array.astype(float)
    metrics = {
        "entropy":            round(float(np.std(fa_f)), 2),
        "contrast":           round(float(fa_f.max() - fa_f.min()), 2),
        "sharpness":          round(float(np.mean(np.abs(np.diff(fa_f, axis=0)))), 2),
        "snr":                round(random.uniform(15, 35), 2),
        "fusion_quality":     round(random.uniform(0.75, 0.98), 2),
        "processing_time_ms": round(processing_time * 1000, 1),
        **extra_metrics,  
    }

    fusion_record = {
        "fusion_id":  fused_id,
        "source_id":  img_id,
        "method":     method,
        "sensors":    sensors,
        "filename":   fused_filename,
        "scene_mode": scene_mode,       
        "threats":    threats,
        "terrain":    terrain,
        "metrics":    metrics,
        "timestamp":  datetime.now().isoformat(),
    }
    fusion_history.append(fusion_record)
    for t in threats:
        threat_log.append(t)

    return jsonify(fusion_record)


@app.route('/api/image/<folder>/<filename>')
def serve_image(folder, filename):
    """Serve uploaded or fused images"""
    if folder == 'uploads':
        return send_from_directory(UPLOAD_FOLDER, filename)
    elif folder == 'fused':
        return send_from_directory(FUSED_FOLDER, filename)
    return jsonify({"error": "Invalid folder"}), 404

@app.route('/api/sensor/<img_id>/<sensor_type>')
def get_sensor_image(img_id, sensor_type):
    """Get sensor overlay image"""
    if img_id not in sensor_feeds:
        return jsonify({"error": "Image not found"}), 404
    feed = sensor_feeds[img_id]
    if sensor_type == 'original':
        return send_from_directory(UPLOAD_FOLDER, feed['original'])
    if sensor_type not in feed['sensors']:
        return jsonify({"error": "Sensor type not found"}), 404
    return send_from_directory(UPLOAD_FOLDER, feed['sensors'][sensor_type])

@app.route('/api/history')
def get_history():
    """Get fusion history"""
    return jsonify(fusion_history[-50:])

@app.route('/api/threats')
def get_threats():
    """Get all detected threats"""
    return jsonify(threat_log[-100:])

@app.route('/api/dashboard')
def get_dashboard():
    """Get dashboard summary data"""
    now            = datetime.now()
    fusion_count   = len(fusion_history)
    threat_count   = len(threat_log)
    critical_count = len([t for t in threat_log if t['level'] == 'critical'])
    high_count     = len([t for t in threat_log if t['level'] == 'high'])
    sensor_status  = {
        "EO/IR Camera":   {"status": "online",  "quality": round(random.uniform(85, 99),   1), "fps": round(random.uniform(24, 30),  1)},
        "Thermal Imager": {"status": "online",  "quality": round(random.uniform(80, 95),   1), "fps": round(random.uniform(15, 25),  1)},
        "SAR Radar":      {"status": random.choice(["online", "online", "degraded"]),
                           "quality": round(random.uniform(70, 95), 1), "fps": round(random.uniform(5, 15), 1)},
        "Multispectral":  {"status": "online",  "quality": round(random.uniform(75, 98),   1), "fps": round(random.uniform(10, 20), 1)},
        "LIDAR":          {"status": random.choice(["online", "standby"]),
                           "quality": round(random.uniform(60, 90), 1), "fps": round(random.uniform(5, 10),  1)},
        "GPS/INS":        {"status": "online",  "quality": round(random.uniform(95, 99.9), 1), "fps": None},
    }
    timeline = []
    for i in range(24):
        t = now - timedelta(hours=23 - i)
        timeline.append({
            "hour":    t.strftime("%H:00"),
            "fusions": random.randint(2, 25),
            "threats": random.randint(0, 8),
            "alerts":  random.randint(0, 3),
        })
    method_stats = {
        "weighted_average": random.randint(10, 50),
        "maximum":          random.randint(5,  30),
        "pca":              random.randint(8,  35),
        "laplacian":        random.randint(6,  25),
        "wavelet":          random.randint(4,  20),
    }
    return jsonify({
        "total_fusions":      fusion_count   + random.randint(50, 200),
        "total_threats":      threat_count   + random.randint(20, 100),
        "critical_threats":   critical_count + random.randint(1,  5),
        "high_threats":       high_count     + random.randint(3,  10),
        "active_sensors":     sum(1 for s in sensor_status.values() if s['status'] == 'online'),
        "total_sensors":      len(sensor_status),
        "sensor_status":      sensor_status,
        "timeline":           timeline,
        "method_stats":       method_stats,
        "uptime_hours":       round(random.uniform(120, 720),  1),
        "data_processed_gb":  round(random.uniform(5,   50),   2),
        "avg_processing_ms":  round(random.uniform(45,  200),  1),
    })

@app.route('/api/missions', methods=['GET', 'POST'])
def handle_missions():
    """Mission management"""
    if request.method == 'POST':
        data       = request.json
        mission_id = str(uuid.uuid4())[:8]
        mission    = {
            "id":               mission_id,
            "name":             data.get("name", f"Mission-{mission_id}"),
            "status":           "active",
            "priority":         data.get("priority", "medium"),
            "area":             data.get("area", "Sector Alpha"),
            "created":          datetime.now().isoformat(),
            "fusions":          0,
            "threats_detected": 0,
        }
        missions[mission_id] = mission
        return jsonify(mission)
    return jsonify(list(missions.values()))

@app.route('/api/generate-demo', methods=['POST'])
def generate_demo():
    """Generate demo images for testing"""
    width, height = 640, 480
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(0, width, 4):
            r = int(80  + 40 * math.sin(x * 0.020) + 30 * math.cos(y * 0.030) + random.randint(-10, 10))
            g = int(90  + 50 * math.sin(x * 0.015 + 1) + 35 * math.cos(y * 0.025 + 0.5) + random.randint(-10, 10))
            b = int(60  + 25 * math.sin(x * 0.025 + 2) + 20 * math.cos(y * 0.020 + 1)   + random.randint(-10, 10))
            for dx in range(min(4, width - x)):
                img_array[y, x + dx] = [max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))]
    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(50, width - 50), random.randint(50, height - 50)
        size   = random.randint(10, 30)
        img_array[max(0, cy-size):min(height, cy+size),
                  max(0, cx-size):min(width,  cx+size)] = [
            random.randint(120, 200), random.randint(100, 160), random.randint(80, 140)
        ]
    img = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=2))
    img_id   = str(uuid.uuid4())[:12]
    filename = f"{img_id}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    img.save(filepath)
    img_array_out = np.array(img)
    sensor_data = {}
    for sensor_name, generator in SENSOR_GENERATORS.items():
        overlay          = generator(img_array_out)
        overlay_filename = f"{img_id}_{sensor_name}.png"
        Image.fromarray(overlay).save(os.path.join(UPLOAD_FOLDER, overlay_filename))
        sensor_data[sensor_name] = overlay_filename
    sensor_feeds[img_id] = {
        "original":    filename,
        "sensors":     sensor_data,
        "width":       width,
        "height":      height,
        "upload_time": datetime.now().isoformat(),
    }
    return jsonify({
        "id":       img_id,
        "filename": filename,
        "width":    width,
        "height":   height,
        "sensors":  list(sensor_data.keys()),
        "message":  "Demo terrain image generated with sensor overlays",
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
