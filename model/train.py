import os
import json
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


# Init MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Allow detection of up to two hands
mp_drawing = mp.solutions.drawing_utils

# Logging configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  
tf.debugging.set_log_device_placement(True)

HAND_COORDS_SIZE = 21
TOTAL_POINTS = HAND_COORDS_SIZE * 3 * 2 # 21 points * 3 coordinates * 2 hands

# Data augmentation
def augment_sequence(sequence):
  augmented_sequences = [sequence]

  # rotate the sequence
  rotated_sequence = np.where(np.any(sequence != 0), sequence + np.random.normal(0, 0.01, size=sequence.shape), sequence)
  augmented_sequences.append(rotated_sequence)

  # shift the sequence
  shifted_sequence = np.where(np.any(sequence != 0), sequence + np.random.normal(0, 0.005, size=sequence.shape), sequence)
  augmented_sequences.append(shifted_sequence)

  return augmented_sequences

def normalize_landmarks(landmarks):
  landmarks = np.array(landmarks)
  min_vals = np.min(landmarks, axis=0)
  max_vals = np.max(landmarks, axis=0)
  normalized_landmarks = (landmarks - min_vals) / (max_vals - min_vals)
  return normalized_landmarks

def train_model(output = './hand_detection.h5'):
  # Load dataset details from the JSON file
  with open('dataset.json', 'r') as f:
    dataset = json.load(f)

  # Variables to store data and labels
  label_landmarks_data = []
  labels = []

  # Process each word in the dataset
  for word in dataset['words']:
    print(f"Processing word: {word['label']}")
    label = word['label']
    word_path = word['path']
    print(f"Procesando carpeta: {word_path}")
    for video_name in os.listdir(word_path):
      video_path = os.path.join(word_path, video_name)
      print(f"Procesando video: {video_name}")
      cap = cv2.VideoCapture(video_path)
      video_landmarks = []

      # Define the codec and create VideoWriter object
      os.makedirs('../dataset/processed', exist_ok=True)

      fourcc = cv2.VideoWriter_fourcc(*'H264')
      out = cv2.VideoWriter(f'../dataset/processed/{label}_{video_name}', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
          frame_landmarks = []

          for hand_landmarks in result.multi_hand_landmarks:
            if len(hand_landmarks.landmark) == HAND_COORDS_SIZE:
              for landmark in hand_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
              mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

          if len(result.multi_hand_landmarks) == 1: # just one hand
            frame_landmarks.extend([0] * (TOTAL_POINTS - len(frame_landmarks)))
          
          if len(frame_landmarks) == TOTAL_POINTS:
            normalized_landmarks = normalize_landmarks(frame_landmarks)
            video_landmarks.append(normalized_landmarks) 

        # Write the frame with hand landmarks
        out.write(frame)

      # Solo añadir secuencias válidas para el aumento de datos
      if video_landmarks:
        video_landmarks = np.array(video_landmarks)
        print(f"Estructura original de secuencia video_landmarks: {video_landmarks.shape}")
        augmented_sequences = augment_sequence(video_landmarks)

        for augmented_sequence in augmented_sequences:
          # print(f"Estructura de secuencia aumentada: {augmented_sequence.shape}")
          if augmented_sequence.shape[1] == TOTAL_POINTS:
            label_landmarks_data.append(augmented_sequence)
            labels.append(label)

      cap.release()
      out.release()

  # Convertir a arrays de numpy para usarlos en el modelo
  label_landmarks_data = np.array(label_landmarks_data, dtype=object)
  labels = np.array(labels)

  # Verificar que tenemos datos y etiquetas
  # print(f"Estructura completa de landmarks_data: {landmarks_data.shape}")
  # print(f"Estructura completa de labels: {labels.shape}")
  if len(labels) == 0 or len(label_landmarks_data) == 0:
    print("No se encontraron datos. Verifica la estructura de carpetas y archivos.")
    return
  
  # Paso 2: Preparar los Datos para Entrenamiento
  label_encoder = LabelEncoder()
  labels_encoded = label_encoder.fit_transform(labels)
  labels_encoded = to_categorical(labels_encoded)

  max_sequence_length = max([len(video_landmarks) for video_landmarks in label_landmarks_data])

  print(f'label_landmarks_data.shape: {label_landmarks_data.shape}')
  print(f'Max sequence length: {max_sequence_length}')

  # Ajustar todos los videos a la misma longitud de secuencia
  landmarks_data_padded = np.zeros((len(label_landmarks_data), max_sequence_length, TOTAL_POINTS))

  for i, video_landmarks in enumerate(label_landmarks_data):
    processed_landmarks = []
    for frame in video_landmarks:
      if len(frame) == TOTAL_POINTS:
        processed_landmarks.append(np.array(frame))
 
    padded_sequence = np.zeros((max_sequence_length, TOTAL_POINTS))
    padded_sequence[:len(processed_landmarks)] = np.array(processed_landmarks)
    landmarks_data_padded[i] = padded_sequence

  print(f"Estructura de landmarks_data_padded: {landmarks_data_padded.shape}")
  print(f"Estructura de labels_encoded: {labels_encoded.shape}")

  X_train, X_test, y_train, y_test = train_test_split(landmarks_data_padded, labels_encoded, test_size=0.2, random_state=42)

  # Paso 3: Entrenar el Modelo LSTM
  model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, activation='tanh')),
    Dense(128, activation='tanh'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
  ])

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Define early stopping
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

  # Entrenar el modelo con más épocas y logs detallados
  model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping])

  # Guardar el modelo entrenado
  model.save(output)

  # Generate predictions for the test set
  y_pred = model.predict(X_test)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_true_classes = np.argmax(y_test, axis=1)

  # Compute the confusion matrix
  cm = confusion_matrix(y_true_classes, y_pred_classes)

  # Compute the precision score
  precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
  print(f'Precision: {precision:.4f}')

  # Plot the confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()

if __name__ == "__main__":
  train_model()