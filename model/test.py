import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import json

COORDS_SIZE = 21
TOTAL_POINTS = COORDS_SIZE * 3 * 2 # 21 points * 3 coordinates * 2 hands
CONFIDENCE_THRESHOLD = 0.6
SEQUENCE_LENGTH = 30

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Allow detection of up to two hands
mp_drawing = mp.solutions.drawing_utils

def test_model(model_path = './hand_detection.h5'):
  with open('dataset.json', 'r') as f:
    dataset = json.load(f)
  labels = [word['label'] for word in dataset['words']]

  label_encoder = LabelEncoder()
  labels_encoded = label_encoder.fit_transform(labels)
  labels_encoded = to_categorical(labels_encoded)

  # Cargar el modelo entrenado
  model = tf.keras.models.load_model(model_path)

  # Configuración para capturar frames y predicción
  cap = cv2.VideoCapture(0)
  frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
  prediction_buffer = deque(maxlen=10)

  while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
      break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
      hand_positions = []
      for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for landmark in hand_landmarks.landmark:
          hand_positions.extend([landmark.x, landmark.y, landmark.z])

      # Ensure the hand_positions list is padded to accommodate two hands
      while len(hand_positions) < TOTAL_POINTS:
        hand_positions.extend([0] * (TOTAL_POINTS - len(hand_positions)))

      if len(hand_positions) == TOTAL_POINTS:
        frame_sequence.append(hand_positions)

        if len(frame_sequence) == SEQUENCE_LENGTH:
          input_data = np.array(frame_sequence).reshape(1, SEQUENCE_LENGTH, TOTAL_POINTS)
          prediction = model.predict(input_data)

          print(prediction)

          confidence = np.max(prediction)
          predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

          if confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(predicted_label)
          
          if len(prediction_buffer) > 0:
            most_common_prediction = Counter(prediction_buffer).most_common(1)[0][0]
            
            if confidence > CONFIDENCE_THRESHOLD:
              cv2.putText(frame, f'Palabra: {most_common_prediction} (Conf: {confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
              print(f"Predicción: {most_common_prediction}, Confianza: {confidence:.2f}")

    cv2.imshow('Detección de Gestos en Tiempo Real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  test_model()
