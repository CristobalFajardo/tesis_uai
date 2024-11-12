import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Ruta principal donde están las carpetas de cada palabra
dataset_folder = "C:/Users/crist/Downloads/hand_pose_detection_project/dataset-20241112T120213Z-001/dataset"

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Variables para almacenar datos y etiquetas
landmarks_data = []
labels = []

# Función para aplicar aumentos de datos a la secuencia de landmarks
def augment_sequence(sequence):
    augmented_sequences = [sequence]

    # Rotación leve en el eje X e Y
    rotated_sequence = sequence + np.random.normal(0, 0.01, size=sequence.shape)
    augmented_sequences.append(rotated_sequence)

    # Invertir la secuencia
    inverted_sequence = sequence[::-1]
    augmented_sequences.append(inverted_sequence)

    # Pequeños desplazamientos aleatorios
    shifted_sequence = sequence + np.random.normal(0, 0.005, size=sequence.shape)
    augmented_sequences.append(shifted_sequence)

    return augmented_sequences

# Procesar cada carpeta en la carpeta principal del dataset
for word_folder in os.listdir(dataset_folder):
    word_path = os.path.join(dataset_folder, word_folder)
    if os.path.isdir(word_path):
        label = word_folder
        print(f"Procesando carpeta: {word_folder}")
        for video_name in os.listdir(word_path):
            video_path = os.path.join(word_path, video_name)
            print(f"Procesando video: {video_name}")
            cap = cv2.VideoCapture(video_path)
            video_landmarks = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    frame_landmarks = []
                    for hand_landmarks in result.multi_hand_landmarks:
                        if len(hand_landmarks.landmark) == 21:
                            for landmark in hand_landmarks.landmark:
                                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

                    # Solo añadir el frame si tiene exactamente 63 valores (21 puntos x 3 coordenadas)
                    if len(frame_landmarks) == 63:
                        video_landmarks.append(frame_landmarks)

            # Solo añadir secuencias válidas para el aumento de datos
            if video_landmarks:
                video_landmarks = np.array(video_landmarks)
                print(f"Estructura original de secuencia video_landmarks: {video_landmarks.shape}")
                for augmented_sequence in augment_sequence(video_landmarks):
                    print(f"Estructura de secuencia aumentada: {augmented_sequence.shape}")
                    if augmented_sequence.shape[1] == 63:
                        landmarks_data.append(augmented_sequence)
                        labels.append(label)

            cap.release()

# Convertir a arrays de numpy para usarlos en el modelo
landmarks_data = np.array(landmarks_data, dtype=object)
labels = np.array(labels)

# Verificar que tenemos datos y etiquetas
print(f"Estructura completa de landmarks_data: {landmarks_data.shape}")
print(f"Estructura completa de labels: {labels.shape}")
if len(labels) == 0 or len(landmarks_data) == 0:
    print("No se encontraron datos. Verifica la estructura de carpetas y archivos.")
    exit()

# Paso 2: Preparar los Datos para Entrenamiento
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Ajustar todos los videos a la misma longitud de secuencia
max_length = 50
landmarks_data_padded = np.zeros((len(landmarks_data), max_length, 63))

for i, video_landmarks in enumerate(landmarks_data):
    processed_landmarks = []
    for frame in video_landmarks:
        if len(frame) == 63:  # Asegurar que cada frame tiene 63 valores
            processed_landmarks.append(np.array(frame))

    if len(processed_landmarks) > max_length:
        landmarks_data_padded[i] = np.array(processed_landmarks[:max_length])
    else:
        padded_sequence = np.zeros((max_length, 63))
        padded_sequence[:len(processed_landmarks)] = np.array(processed_landmarks)
        landmarks_data_padded[i] = padded_sequence

print(f"Estructura de landmarks_data_padded: {landmarks_data_padded.shape}")
print(f"Estructura de labels_encoded: {labels_encoded.shape}")

X_train, X_test, y_train, y_test = train_test_split(landmarks_data_padded, labels_encoded, test_size=0.2, random_state=42)

# Paso 3: Entrenar el Modelo LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con más épocas y logs detallados
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=2)

# Guardar el modelo entrenado
model.save('modelo_lsa64_mejorado.h5')

# Paso 4: Predicción en Tiempo Real

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_lsa64_mejorado.h5')

# Configuración para capturar frames y predicción
cap = cv2.VideoCapture(0)
sequence_length = 30
frame_sequence = deque(maxlen=sequence_length)
confidence_threshold = 0.5
prediction_buffer = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
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

        if len(hand_positions) == 63:
            frame_sequence.append(hand_positions)

            if len(frame_sequence) == sequence_length:
                input_data = np.array(frame_sequence).reshape(1, sequence_length, 63)
                prediction = model.predict(input_data)
                confidence = np.max(prediction)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                if confidence > confidence_threshold:
                    prediction_buffer.append(predicted_label)
                
                if len(prediction_buffer) > 0:
                    most_common_prediction = Counter(prediction_buffer).most_common(1)[0][0]
                    cv2.putText(frame, f'Palabra: {most_common_prediction} (Conf: {confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    print(f"Predicción: {most_common_prediction}, Confianza: {confidence:.2f}")

    cv2.imshow('Detección de Gestos en Tiempo Real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
