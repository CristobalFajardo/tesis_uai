import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Abre la cámara web
cap = cv2.VideoCapture(0)
white_bg_on = True

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # blank bg
    frame_w_bg = np.empty(frame.shape)
    frame_w_bg.fill(255)

    if not ret:
        break

    # Convierte la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Procesa la imagen y detecta las manos
    result = hands.process(rgb_frame)

    base_frame = frame_w_bg if white_bg_on else frame

    # Dibuja las anotaciones de las manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                base_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,0), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,0), thickness=4, circle_radius=2),
            )
    # Muestra el frame con las anotaciones
    cv2.imshow('Hand Pose Detection', base_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
