import socketio
import flask
from aiohttp import web
from sign_detection import detect_hand_landmarks
from sklearn.preprocessing import LabelEncoder
import base64

# Initialize Flask and Socket.IO
app = flask.Flask(__name__)
sio = socketio.AsyncServer(cors_allowed_origins="*")  # Allow CORS
app = web.Application()
sio.attach(app)
PORT = 8080

# Event listener for client connection
@sio.event
async def connect(sid, environ):
  print(f'Client connected: {sid}')

# Event listener for client disconnection
@sio.event
async def disconnect(sid):
  print(f'Client disconnected: {sid}')

# Event listener for receiving video frames
@sio.event
async def video_frame(sid, data):
  buffer = detect_hand_landmarks(data)
  # Encode the processed frame to JPEG
  jpg_as_text = base64.b64encode(buffer).decode('utf-8')

  # Emit the processed frame back to the client
  await sio.emit('processed_frame', jpg_as_text, room=sid)

# Run the app
if __name__ == '__main__':
  web.run_app(app, port=PORT)
