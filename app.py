import base64
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, request
from flask_socketio import SocketIO, emit
from PIL import Image
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5, color=(0, 0, 255))
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=3)

# TensorFlow model setup
model = tf.keras.models.load_model('final.h5')
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'NEXT', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Lock for thread safety
thread_lock = threading.Lock()

def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load image in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    img = cv2.resize(img, (256, 256))  # Resize image
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class = classes[np.argmax(predictions)]
    return predicted_class

def save_cropped_image(cropped_hand):
    cropped_hand_resized = cv2.resize(cropped_hand, (256, 256))
    flipped_hand = cv2.flip(cropped_hand_resized, 1)  # Horizontal flip
    is_success, buffer = cv2.imencode(".jpg", flipped_hand)
    if is_success:
        with open('cropped_hand.jpg', 'wb') as f:
            f.write(buffer)
        print('Cropped, resized, horizontally flipped, and saved image as JPG: cropped_hand.jpg')
    else:
        print('Error in converting image to JPG')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message', {'data': 'Connected to the server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image')
def handle_image(data):
    image_data = data['image']
    sid = request.sid
    try:
        image_bytes = base64.b64decode(image_data)
        with open(f'received_image_{sid}.png', 'wb') as f:
            f.write(image_bytes)
            print(f'Image received and saved for client {sid}.')

        # Process the image in a separate thread
        threading.Thread(target=process_image, args=(f'received_image_{sid}.png', sid)).start()

    except Exception as e:
        print('Error saving or processing image:', e)

def process_image(image_path, sid):
    global thread_lock
    with thread_lock:
        image = cv2.imread(image_path)
        if image is None:
            print('Error: Image not loaded.')
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec, connection_drawing_spec)
                image_height, image_width, _ = image.shape
                landmarks = np.zeros((len(hand_landmarks.landmark), 2))
                for idx, lm in enumerate(hand_landmarks.landmark):
                    landmarks[idx] = [lm.x * image_width, lm.y * image_height]
                hand_bbox = calculate_hand_bbox(landmarks, image_width, image_height)
                cropped_hand = image[int(hand_bbox[1]):int(hand_bbox[3]), int(hand_bbox[0]):int(hand_bbox[2])]
                save_cropped_image(cropped_hand)
                predicted_class = predict_image('cropped_hand.jpg')
                socketio.emit('prediction', {'class': predicted_class}, room=sid)

            cv2.imwrite(f'annotated_image_{sid}.png', image)
            print(f'Annotated image with landmarks saved for client {sid}: annotated_image_{sid}.png')
        else:
            print('No hands detected.')

def calculate_hand_bbox(landmarks, image_width, image_height):
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    padding = 50  # Increased margin for better cropping
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(image_width, int(x_max) + padding)
    y_max = min(image_height, int(y_max) + padding)
    return [x_min, y_min, x_max, y_max]

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
