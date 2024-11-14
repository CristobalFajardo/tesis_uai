import cv2
import numpy as np
import socketio
import mediapipe as mp
import flask
import eventlet
import base64

# Initialize Flask and Socket.IO
app = flask.Flask(__name__)
sio = socketio.Server(cors_allowed_origins="*")  # Allow CORS
app = socketio.WSGIApp(sio, app)
PORT = 8080

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Event listener for client connection
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

# Event listener for client disconnection
@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')

# Event listener for receiving video frames
@sio.on('video_frame')
def handle_video_frame(sid, data):
    # Decode the received frame
    np_data = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    if frame is None:
        return
    
    # Convert the BGR image to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Encode the processed frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Emit the processed frame back to the client
    sio.emit('processed_frame', jpg_as_text, room=sid)

# Run the app
if __name__ == '__main__':
    # Start the socket server with eventlet
    eventlet.wsgi.server(eventlet.listen(('', PORT)), app)
