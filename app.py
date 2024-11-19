import base64
import logging
import os

import cv2
import numpy as np
from flask import Flask, g, jsonify, request
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model paths defined using environment variables for flexibility
MODEL_PATHS = {
    "model_n": "models/yolo11n.pt",
    "faces_model": "models/yolov8n-face.pt",
}


def get_model(model_name):
    """Load and return the YOLO model from Flask's application context."""
    if model_name not in g:
        if model_name in MODEL_PATHS:
            logger.info(f"Loading {model_name} model")
            g[model_name] = YOLO(MODEL_PATHS[model_name], verbose=False)
        else:
            raise ValueError(f"Model {model_name} not supported.")
    return g[model_name]


def load_image(input_data):
    """Load an image from a file path or image bytes."""
    if isinstance(input_data, str):
        return cv2.imread(input_data)
    elif isinstance(input_data, bytes):
        np_arr = np.frombuffer(input_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        raise ValueError(
            "Input should be either a file path (str) or image bytes (bytes)."
        )


def crop_padded_box(image, box, padding, image_width, image_height):
    """Crop a box from an image with padding, ensuring the coordinates are within bounds."""
    x_min, y_min, x_max, y_max = map(int, box)
    x_min_padded = max(0, x_min - padding)
    y_min_padded = max(0, y_min - padding)
    x_max_padded = min(image_width, x_max + padding)
    y_max_padded = min(image_height, y_max + padding)
    cropped_object = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    return cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)


@app.route("/detect_people", methods=["POST"])
def detect_persons_route():
    """Flask route to detect persons in an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected for uploading"}), 400

    padding = request.form.get("padding", default=200, type=int)
    image_bytes = file.read()

    try:
        model_n = get_model("model_n")
        image = load_image(image_bytes)
        results = model_n(image, classes=0, line_width=100)
        image_height, image_width = image.shape[:2]

        persons = [
            crop_padded_box(
                image, box.xyxy.cpu().numpy(), padding, image_width, image_height
            )
            for result in results
            for box in result.boxes
        ]

        encoded_images = [
            base64.b64encode(
                cv2.imencode(".jpg", cv2.cvtColor(person, cv2.COLOR_RGB2BGR))[1]
            ).decode("utf-8")
            for person in persons
        ]

        return jsonify({"persons": encoded_images}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect_faces", methods=["POST"])
def detect_faces_route():
    """Flask route to detect faces in an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected for uploading"}), 400

    confidence_threshold = request.form.get("confidence", default=0.75, type=float)
    padding = request.form.get("padding", default=20, type=int)
    image_bytes = file.read()

    try:
        face_model = get_model("faces_model")
        image = load_image(image_bytes)
        results = face_model(image)
        image_height, image_width = image.shape[:2]

        faces = []
        for result in results:
            boxes = [x for x in result.boxes if x.conf[0] > confidence_threshold]
            if not boxes:  # If no boxes match the threshold, fallback to all detections
                boxes = result.boxes

            for box in boxes:
                box_coords = box.xyxy.cpu().numpy()[0]
                faces.append(
                    crop_padded_box(
                        image, box_coords, padding, image_width, image_height
                    )
                )

        encoded_faces = [
            base64.b64encode(
                cv2.imencode(".jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))[1]
            ).decode("utf-8")
            for face in faces
        ]

        return jsonify({"faces": encoded_faces}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.teardown_appcontext
def teardown(exception):
    """Clean up models from Flask's application context after request."""
    for model_name in MODEL_PATHS.keys():
        model = g.pop(model_name, None)
        if model is not None:
            del model


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
