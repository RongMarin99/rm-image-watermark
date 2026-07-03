from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
app = Flask(__name__, static_folder=ROOT, static_url_path="")


def b64_to_bgr(b64: str):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def b64_to_gray(b64: str):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def bgr_to_b64(img) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def with_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/api/process", methods=["POST", "OPTIONS"])
def process():
    if request.method == "OPTIONS":
        return with_cors(jsonify({}))

    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode")
    img = b64_to_bgr(body.get("image", ""))

    if img is None:
        return with_cors(jsonify({"error": "Invalid image"})), 400

    H, W = img.shape[:2]

    # ── Crop: inpaint the selected rectangle (watermark area) ────────────────
    if mode == "crop":
        x = max(0, int(body.get("x", 0)))
        y = max(0, int(body.get("y", 0)))
        w = int(body.get("w", 0))
        h = int(body.get("h", 0))
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        if x2 <= x or y2 <= y:
            return with_cors(jsonify({"error": "Invalid region"})), 400
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y:y2, x:x2] = 255
        result = cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

    # ── Paint: manual brush mask → inpaint ───────────────────────────────────
    elif mode == "paint":
        mask = b64_to_gray(body.get("mask", ""))
        if mask is None:
            return with_cors(jsonify({"error": "Invalid mask"})), 400
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(img, binary, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # ── Auto: detect watermark via brightness + edge-pattern, then inpaint ───
    elif mode == "auto":
        sensitivity = float(body.get("sensitivity", 50))  # 0-100
        preview     = body.get("preview", False)

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)

        # 1. Bright regions (white / light watermarks)
        bright_thresh = max(100, int(255 - sensitivity * 1.1))
        _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

        # 2. High-frequency text / logo patterns vs local background
        diff = cv2.absdiff(gray, blurred)
        edge_thresh = max(5, int(30 - sensitivity * 0.2))
        _, edge_mask = cv2.threshold(diff, edge_thresh, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(bright_mask, edge_mask)

        # Morphological close: connect nearby strokes into solid blobs
        close_k = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=3)

        # Remove blobs > 25 % of image area (those are image content, not watermark)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((H, W), dtype=np.uint8)
        max_blob = H * W * 0.25
        for cnt in contours:
            if cv2.contourArea(cnt) < max_blob:
                cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

        if preview:
            overlay = img.copy()
            overlay[mask > 0] = [0, 0, 220]   # red tint where watermark detected
            preview_img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
            return with_cors(jsonify({"preview": bgr_to_b64(preview_img)}))

        result = cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

    else:
        return with_cors(jsonify({"error": f"Unknown mode: {mode}"})), 400

    return with_cors(jsonify({"result": bgr_to_b64(result)}))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
