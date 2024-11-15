import cv2
import mediapipe as mp
import numpy as np
import base64

# Define image size
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def detect_hand_landmarks(data):
  np_data = np.frombuffer(data, np.uint8)
  frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

  if frame is None:
    return
  
  # Resize the frame to the defined image size
  frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
  # Convert the BGR image to RGB for MediaPipe processing
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Procesa la imagen y detecta las manos
  result = hands.process(rgb_frame)

  # Dibuja las anotaciones de las manos
  if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          frame,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing.DrawingSpec(color=(0,0,0), thickness=4, circle_radius=2),
          mp_drawing.DrawingSpec(color=(0,0,0), thickness=4, circle_radius=2),
  )
  
  _, buffer = cv2.imencode('.jpg', frame)

  return buffer
