from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import io
from PIL import Image

from db import get_use_from_db, store_use_to_db
from gemini_fetch import get_use_from_gemini

predict_plant = Blueprint("predict_plant", __name__)

# MODEL + LABEL FILE
MODEL_PATH = "model/leaf_model.h5"
LABEL_FILE = "model/labels.txt"

model = load_model(MODEL_PATH)

# Load labels from labels.txt
labels = []
with open(LABEL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            labels.append(parts[1])


def extract_names(folder_name):
    scientific = folder_name
    common = "Unknown"

    if "(" in folder_name and ")" in folder_name:
        scientific = folder_name.split("(")[0].strip().rstrip("_")
        common = folder_name[folder_name.find("(")+1 : folder_name.find(")")]

    return scientific, common


@predict_plant.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    folder_name = labels[class_idx]

    scientific, common = extract_names(folder_name)

    uses = get_use_from_db(scientific)
    if not uses:
        uses = get_use_from_gemini(scientific)
        if uses:
            store_use_to_db(scientific, uses)

    return jsonify({
        "scientific_name": scientific,
        "common_name": common,
        "uses": uses if uses else "Not available"
    })
