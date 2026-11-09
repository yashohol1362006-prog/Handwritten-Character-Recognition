
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

MODEL_PATH = "models/character_model.keras"

# Load trained CNN model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run python src/train.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):

    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image.")

    # Convert white background to black and character to white
    _, image = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find character boundaries
    coords = cv2.findNonZero(image)

    if coords is None:
        raise ValueError("No character detected.")

    x, y, w, h = cv2.boundingRect(coords)

    image = image[y:y+h, x:x+w]

    # Add padding
    size = max(w, h) + 20
    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[y_offset:y_offset+h, x_offset:x_offset+w] = image

    # Resize to 28x28
    image = cv2.resize(square, (28, 28))

    # Normalize
    image = image.astype("float32") / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    return image


def predict_character(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction[0])

    character = chr(predicted_index + 65)

    confidence = float(prediction[0][predicted_index]) * 100

    print("\n===== PREDICTION RESULT =====")
    print("Predicted Character:", character)
    print("Confidence:", round(confidence, 2), "%")

    # Display top 3 predictions
    top_indices = np.argsort(prediction[0])[-3:][::-1]

    print("\nTop 3 Predictions:")

    for index in top_indices:
        letter = chr(index + 65)
        probability = prediction[0][index] * 100

        print(f"{letter}: {probability:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py image.png")
    else:
        predict_character(sys.argv[1])
