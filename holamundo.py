import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM

# Recolección y Preprocesamiento de Datos

# Cargar el archivo .mat usando h5py
file_path = 'lsa64_preprocessed/lsa64_positions.mat'
print("Cargando archivo .mat...")
mat = h5py.File(file_path, 'r')
print("Archivo .mat cargado.")

# Acceder a las posiciones de las manos y clases
hand_positions_left = np.array(mat['db']['hand_positions_left'])
hand_positions_right = np.array(mat['db']['hand_positions_right'])
print(f"hand_positions_left shape: {hand_positions_left.shape}")
print(f"hand_positions_right shape: {hand_positions_right.shape}")

# Verificar el contenido de class_refs
class_refs = mat['db']['class']
print(f"class_refs shape: {class_refs.shape}")

classes = []
for i, ref in enumerate(class_refs):
    try:
        ref_value = mat[ref[0]][()]  # Acceder al valor de la referencia
        if isinstance(ref_value, np.ndarray):
            ref_value = ref_value[0]  # Acceder al primer elemento si es un array
        actual_class = int(ref_value)
        classes.append(actual_class)
        if i < 5:
            print(f"Clase {i}: {actual_class}")
    except Exception as e:
        print(f"Error en la clase {i}: {e}")
        break

print(f"Número total de clases procesadas: {len(classes)}")

classes = np.array(classes)

# Combinación de posiciones de mano izquierda y derecha
positions = np.concatenate([hand_positions_left, hand_positions_right], axis=-1)
print(f"positions shape: {positions.shape}")

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(positions, classes, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Ajustar la forma de X_train y X_test
if len(X_train.shape) == 2:  # Si X_train es 2D, agregamos una dimensión de tiempo
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
print(f"Forma de X_train después del ajuste: {X_train.shape}")

# One-hot encoding de las etiquetas
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"y_train shape after one-hot encoding: {y_train.shape}")
print(f"y_test shape after one-hot encoding: {y_test.shape}")

# Cerrar el archivo .mat
mat.close()
print("Archivo .mat cerrado.")

# Entrenamiento del Modelo

# Definir el modelo
model = Sequential()

# Capa LSTM para secuencias temporales
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(y_train.shape[1], activation='softmax'))
print("Modelo definido.")

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Modelo compilado.")

# Entrenar el modelo
try:
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    print("Modelo entrenado.")
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")

# Guardar el modelo entrenado
model.save('modelo_lsa64.h5')
print("Modelo guardado.")

# Predicción en Tiempo Real

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_lsa64.h5')
print("Modelo cargado para predicción en tiempo real.")

# Inicializar MediaPipe y la cámara
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa la imagen y detecta las manos
    result = hands.process(rgb_frame)

    # Extraer posiciones de las manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_positions = []
            for landmark in hand_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                hand_positions.append([x, y])
            hand_positions = np.array(hand_positions)

            # Asegurarse de que la entrada tenga la forma adecuada
            if hand_positions.shape[0] == 1:  # Se espera 1 punto para cada mano
                hand_positions_combined = np.concatenate([hand_positions.flatten(), hand_positions.flatten()])
                hand_positions_combined = hand_positions_combined.reshape(1, 1, -1)  # Ajustar dimensiones
                print(f"Dimensión de entrada para predicción: {hand_positions_combined.shape}")

                # Realizar la predicción
                prediccion = model.predict(hand_positions_combined)
                clase_predicha = np.argmax(prediccion, axis=1)

                # Mostrar la clase predicha en el frame
                cv2.putText(frame, f'Clase: {clase_predicha[0]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar el frame
    cv2.imshow('Detección de Señales en Tiempo Real', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
print("Predicción en tiempo real finalizada.")
