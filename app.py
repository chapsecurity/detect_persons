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

# Model paths defined using environment variables for flexibility
MODEL_PATHS = {
    "model_n": "models/yolo11n.pt",
    "faces_model": "models/yolov8n-face.pt",
    "license_plates_model": "models/license_plate_detector.pt",
}


def get_model(model_name):
    """Load and return the YOLO model from Flask's application context."""
    if not hasattr(g, model_name):
        if model_name in MODEL_PATHS:
            logger.info(f"Loading {model_name} model")
            setattr(g, model_name, YOLO(MODEL_PATHS[model_name], verbose=False))
        else:
            raise ValueError(f"Model {model_name} not supported.")
    return getattr(g, model_name)


def crop_padded_box(image, box, padding, image_width, image_height):
    """Crop a box from an image with padding, ensuring the coordinates are within bounds."""
    x_min, y_min, x_max, y_max = map(int, box)
    x_min_padded = max(0, x_min - padding)
    y_min_padded = max(0, y_min - padding)
    x_max_padded = min(image_width, x_max + padding)
    y_max_padded = min(image_height, y_max + padding)
    cropped_object = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    return cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)


def get_image(base64_string):
    # Decode the base64 string
    image_bytes = base64.b64decode(base64_string)
    # Convert the bytes into a NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image using OpenCV
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.route("/detect_people", methods=["POST"])
def detect_people_route():
    """Flask route to detect people in an image."""
    padding = 20

    try:
        model_n = get_model("model_n")
        image = get_image(request.json["image"])
        results = model_n(image, classes=0, line_width=100)
        image_height, image_width = image.shape[:2]

        people = []
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
                cropped_object = image[
                    y_min_padded:y_max_padded, x_min_padded:x_max_padded
                ]
                # Convert BGR to RGB and add to persons list
                people.append(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))

        encoded_images = [
            base64.b64encode(
                cv2.imencode(".jpg", cv2.cvtColor(person, cv2.COLOR_RGB2BGR))[1]
            ).decode("utf-8")
            for person in people
        ]

        return jsonify({"result": encoded_images}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect_faces", methods=["POST"])
def detect_faces_route():
    """Flask route to detect faces in an image."""
    confidence_threshold = request.form.get("confidence", default=0.75, type=float)
    padding = 20

    try:
        face_model = get_model("faces_model")
        image = get_image(request.json["image"])
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

        return jsonify({"result": encoded_faces}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect_license_plates", methods=["POST"])
def detect_license_plates():
    """Flask route to detect faces in an image."""
    try:
        frame = get_image(request.json["image"])

        license_plates_model = get_model("license_plates_model")
        license_plates = license_plates_model(frame)[0]

        plates = []

        # detect license plates
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = license_plate

            # crop license plate
            license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

            license_plate_crop_base64 = base64.b64encode(
                cv2.imencode(
                    ".jpg", cv2.cvtColor(license_plate_crop, cv2.COLOR_RGB2BGR)
                )[1]
            ).decode("utf-8")

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY
            )
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            license_plate_crop_thresh = base64.b64encode(
                cv2.imencode(
                    ".jpg", cv2.cvtColor(license_plate_crop_thresh, cv2.COLOR_RGB2BGR)
                )[1]
            ).decode("utf-8")

            plates.append(
                {
                    "colored": license_plate_crop_base64,
                    "thresh": license_plate_crop_thresh,
                }
            )

        return jsonify({"result": plates}), 200

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.teardown_appcontext
def teardown(exception):
    """Clean up models from Flask's application context after request."""
    for model_name in MODEL_PATHS.keys():
        if hasattr(g, model_name):
            delattr(g, model_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5678)
