from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import json
import os

# 🔥 NEW
import tflite_runtime.interpreter as tflite
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==============================
# APP INIT
# ==============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# STATIC FILES
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ==============================
# LOAD MODEL (TFLITE ✅ LIGHT)
# ==============================
model_path = os.path.join(BASE_DIR, "bird_model.tflite")
class_path = os.path.join(BASE_DIR, "class_names.json")

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(class_path) as f:
    class_names = json.load(f)

class_names = {v: k for k, v in class_names.items()}

print("✅ TFLite Model Loaded")

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/health")
def health():
    return {"status": "ok"}

# ==============================
# PREPROCESS
# ==============================
def preprocess(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid image file")

    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# TFLITE PREDICT
# ==============================
def predict_tflite(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# ==============================
# LIME EXPLAIN
# ==============================
explainer = lime_image.LimeImageExplainer()

def lime_explain(img):
    def predict_fn(images):
        preds = []
        for im in images:
            im = np.expand_dims(im.astype(np.float32), axis=0)
            pred = predict_tflite(im)
            preds.append(pred)
        return np.array(preds)

    explanation = explainer.explain_instance(
        img[0],
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=50  # 🔥 keep low for memory
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return temp.tolist()

# ==============================
# PREDICT
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = preprocess(img_bytes)

        preds = predict_tflite(img)

        top_indices = preds.argsort()[-3:][::-1]
        top_scores = preds[top_indices]
        top_labels = [class_names[i] for i in top_indices]

        top1_conf = float(top_scores[0])
        top2_conf = float(top_scores[1])

        if top1_conf < 0.75:
            return {
                "label": "Unknown Bird",
                "confidence": top1_conf,
                "similar": top_labels
            }

        if abs(top1_conf - top2_conf) < 0.15:
            return {
                "label": "Uncertain Bird",
                "confidence": top1_conf,
                "similar": top_labels
            }

        return {
            "label": top_labels[0],
            "confidence": top1_conf,
            "similar": top_labels
        }

    except Exception as e:
        return {"error": str(e)}

# ==============================
# EXPLAIN (LIME 🔥)
# ==============================
@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = preprocess(img_bytes)

        preds = predict_tflite(img)

        top_indices = preds.argsort()[-3:][::-1]
        top_scores = preds[top_indices]
        top_labels = [class_names[i] for i in top_indices]

        top1_conf = float(top_scores[0])
        top2_conf = float(top_scores[1])

        lime_map = lime_explain(img)

        if top1_conf < 0.75:
            label = "Unknown Bird"
        elif abs(top1_conf - top2_conf) < 0.15:
            label = "Uncertain Bird"
        else:
            label = top_labels[0]

        return {
            "label": label,
            "confidence": top1_conf,
            "similar": top_labels,
            "ig_map": lime_map  # keep same name for frontend
        }

    except Exception as e:
        return {"error": str(e)}


