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





# works but it is not working well as becasuee of timelimit  on render using tensorflow , we use tensorflow for both training model and integrated gradients . for model we can train locally usiing tensorflow and then we can push the trained model without need of using tensorflow on render but for xai explanaton tensor flow is needed which is approx300-400mb .so switching to lime .which doesnot use tf
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse

# import numpy as np
# import tensorflow as tf
# from PIL import Image, UnidentifiedImageError
# import io
# import json
# import os

# # ==============================
# # APP INIT
# # ==============================
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==============================
# # SERVE ANGULAR STATIC FILES ✅ FIXED
# # ==============================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# STATIC_DIR = os.path.join(BASE_DIR, "static")

# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # ✅ FIX: ONLY ROOT SERVES INDEX.HTML
# @app.get("/")
# def serve_frontend():
#     return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# # ==============================
# # LOAD MODEL (FIXED PATH ✅)
# # ==============================
# model_path = os.path.join(BASE_DIR, "bird_model.h5")
# class_path = os.path.join(BASE_DIR, "class_names.json")

# model = tf.keras.models.load_model(model_path, compile=False)

# with open(class_path) as f:
#     class_names = json.load(f)

# class_names = {v: k for k, v in class_names.items()}

# print("✅ Model Loaded")

# # ==============================
# # HEALTH CHECK (FOR RENDER)
# # ==============================
# @app.get("/health")
# def health():
#     return {"status": "ok"}

# # ==============================
# # PREPROCESS
# # ==============================
# def preprocess(img_bytes):
#     try:
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     except UnidentifiedImageError:
#         raise ValueError("Invalid image file")

#     img = img.resize((224, 224))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# # ==============================
# # INTEGRATED GRADIENTS
# # ==============================
# def integrated_gradients(model, x, steps=20):
#     baseline = tf.zeros_like(x)

#     scaled_inputs = [
#         baseline + (i / steps) * (x - baseline)
#         for i in range(steps + 1)
#     ]
#     scaled_inputs = tf.concat(scaled_inputs, axis=0)

#     with tf.GradientTape() as tape:
#         tape.watch(scaled_inputs)
#         preds = model(scaled_inputs)

#     grads = tape.gradient(preds, scaled_inputs)

#     grads = tf.reshape(grads, (steps + 1,) + x.shape[1:])
#     avg_grads = tf.reduce_mean(grads[:-1], axis=0)

#     ig = (x[0] - baseline[0]) * avg_grads
#     ig = tf.abs(ig)

#     max_val = tf.reduce_max(ig)
#     if max_val != 0:
#         ig = ig / max_val

#     return ig.numpy()

# # ==============================
# # PREDICT
# # ==============================
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         img_bytes = await file.read()
#         img = preprocess(img_bytes)

#         preds = model.predict(img, verbose=0)[0]

#         top_indices = preds.argsort()[-3:][::-1]
#         top_scores = preds[top_indices]
#         top_labels = [class_names[i] for i in top_indices]

#         top1_conf = float(top_scores[0])
#         top2_conf = float(top_scores[1])

#         if top1_conf < 0.75:
#             return {
#                 "label": "Unknown Bird",
#                 "confidence": top1_conf,
#                 "similar": top_labels
#             }

#         if abs(top1_conf - top2_conf) < 0.15:
#             return {
#                 "label": "Uncertain Bird",
#                 "confidence": top1_conf,
#                 "similar": top_labels
#             }

#         return {
#             "label": top_labels[0],
#             "confidence": top1_conf,
#             "similar": top_labels
#         }

#     except Exception as e:
#         return {"error": str(e)}

# # ==============================
# # EXPLAIN (XAI)
# # ==============================
# @app.post("/explain")
# async def explain(file: UploadFile = File(...)):
#     try:
#         img_bytes = await file.read()
#         img = preprocess(img_bytes)

#         preds = model.predict(img, verbose=0)[0]

#         top_indices = preds.argsort()[-3:][::-1]
#         top_scores = preds[top_indices]
#         top_labels = [class_names[i] for i in top_indices]

#         top1_conf = float(top_scores[0])
#         top2_conf = float(top_scores[1])

#         ig = integrated_gradients(model, img)

#         if top1_conf < 0.75:
#             label = "Unknown Bird"
#         elif abs(top1_conf - top2_conf) < 0.15:
#             label = "Uncertain Bird"
#         else:
#             label = top_labels[0]

#         return {
#             "label": label,
#             "confidence": top1_conf,
#             "similar": top_labels,
#             "ig_map": ig.tolist()
#         }

#     except Exception as e:
#         return {"error": str(e)}




# # works good but even if we give unknown bird it is picking it from the some cls and with high confidence
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import tensorflow as tf
# from PIL import Image, UnidentifiedImageError
# import io
# import json

# # ==============================
# # APP INIT
# # ==============================
# app = FastAPI()

# # ✅ CORS (MUST BE AFTER app)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==============================
# # LOAD MODEL
# # ==============================
# model = tf.keras.models.load_model("bird_model.keras")

# with open("class_names.json") as f:
#     class_names = json.load(f)

# # reverse mapping
# class_names = {v: k for k, v in class_names.items()}

# # ==============================
# # PREPROCESS
# # ==============================
# def preprocess(img_bytes):
#     try:
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     except UnidentifiedImageError:
#         raise ValueError("Invalid image file")

#     img = img.resize((224, 224))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# # ==============================
# # INTEGRATED GRADIENTS
# # ==============================
# def integrated_gradients(model, x, steps=20):
#     baseline = tf.zeros_like(x)

#     scaled_inputs = [
#         baseline + (i / steps) * (x - baseline)
#         for i in range(steps + 1)
#     ]
#     scaled_inputs = tf.concat(scaled_inputs, axis=0)

#     with tf.GradientTape() as tape:
#         tape.watch(scaled_inputs)
#         preds = model(scaled_inputs)

#     grads = tape.gradient(preds, scaled_inputs)

#     grads = tf.reshape(grads, (steps + 1,) + x.shape[1:])
#     avg_grads = tf.reduce_mean(grads[:-1], axis=0)

#     ig = (x[0] - baseline[0]) * avg_grads
#     ig = tf.abs(ig)
#     ig = ig / tf.reduce_max(ig)

#     return ig.numpy()

# # ==============================
# # ROUTES
# # ==============================

# @app.get("/")
# def home():
#     return {"message": "Bird Classifier API 🚀"}


# # 🔹 PREDICT
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         img_bytes = await file.read()
#         img = preprocess(img_bytes)

#         preds = model.predict(img)
#         class_index = int(np.argmax(preds[0]))
#         confidence = float(np.max(preds[0]))

#         return {
#             "label": class_names[class_index],
#             "confidence": confidence
#         }

#     except Exception as e:
#         return {"error": str(e)}


# # 🔹 EXPLAIN
# @app.post("/explain")
# async def explain(file: UploadFile = File(...)):
#     try:
#         img_bytes = await file.read()
#         img = preprocess(img_bytes)

#         preds = model.predict(img)
#         class_index = int(np.argmax(preds[0]))
#         confidence = float(np.max(preds[0]))

#         ig = integrated_gradients(model, img)

#         return {
#             "label": class_names[class_index],
#             "confidence": confidence,
#             "ig_map": ig.tolist()
#         }

#     except Exception as e:
#         return {"error": str(e)}
