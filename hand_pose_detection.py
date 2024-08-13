import cv2
import mediapipe as mp

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Abre la cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa la imagen y detecta las manos
    result = hands.process(rgb_frame)

    # Dibuja las anotaciones de las manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Muestra el frame con las anotaciones
    cv2.imshow('Hand Pose Detection', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
