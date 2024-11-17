import base64
import logging

import cv2
import numpy as np
from flask import Flask, g, jsonify, request
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def get_model():
    if "model_n" not in g:
        logger.info("Loading YOLO model...")
        g.model_n = YOLO("dev/yolo11n.pt")
    return g.model_n


def detect_persons(input_data, model_n, padding=200):
    """
    Detect persons in an image using YOLO model. Accepts either a file path or image bytes.

    Args:
        input_data (str or bytes): Path to the image file or image in bytes.
        model_n: YOLO model instance.
        padding (int): Padding applied around detected persons.

    Returns:
        list: List of cropped person images as numpy arrays in RGB format.
    """
    # Determine if the input is an image path or bytes and load the image accordingly
    if isinstance(input_data, str):  # Path to image
        image = cv2.imread(input_data)
        results = model_n(input_data, classes=0, line_width=100)
    elif isinstance(input_data, bytes):  # Image bytes
        np_arr = np.frombuffer(input_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results = model_n(image, classes=0, line_width=100)
    else:
        raise ValueError(
            "Input should be either a file path (str) or image bytes (bytes)."
        )

    # Ensure the image was loaded correctly
    if image is None:
        raise ValueError("Failed to load image. Check the input path or image bytes.")

    # Get the image dimensions
    image_height, image_width = image.shape[:2]

    persons = []

    # Loop over the detections and extract the bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            # Apply padding, ensuring the new coordinates are within image bounds
            x_min_padded = max(0, x_min - padding)
            y_min_padded = max(0, y_min - padding)
            x_max_padded = min(image_width, x_max + padding)
            y_max_padded = min(image_height, y_max + padding)

            # Crop the padded object from the image
            cropped_object = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

            # Convert BGR to RGB and add to persons list
            persons.append(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))

    return persons


@app.route("/detect_persons", methods=["POST"])
def detect_persons_route():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected for uploading"}), 400

    # Get padding parameter from the request, default to 200 if not provided
    padding = request.form.get("padding", default=200, type=int)

    image_bytes = file.read()
    try:
        model_n = get_model()
        persons = detect_persons(image_bytes, model_n, padding)
        # Encode the cropped images to base64 strings
        encoded_images = []
        for person in persons:
            # Convert numpy array (RGB) to image bytes
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(person, cv2.COLOR_RGB2BGR))
            # Encode to base64 string
            encoded_image = base64.b64encode(buffer).decode("utf-8")
            encoded_images.append(encoded_image)

        return jsonify({"persons": encoded_images}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.teardown_appcontext
def teardown(exception):
    model_n = g.pop("model_n", None)
    if model_n is not None:
        del model_n


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
