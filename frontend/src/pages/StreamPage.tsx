import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import { serverUrl } from '../config';

const StreamPage = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [socket, setSocket] = useState(null);
  
  useEffect(() => {
    // Connect to the Socket.IO server
    const socket = io(`${serverUrl}`);  // Adjust the URL according to your server settings
    setSocket(socket);

    // Get access to the user's webcam
    const getUserMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });

        // Capture video frames and send them to the server
        const captureFrames = () => {
          const video = document.createElement('video');
          video.srcObject = stream;
          video.play();

          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');

          const sendFrame = () => {
            if (!video.paused && !video.ended) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              context.drawImage(video, 0, 0, canvas.width, canvas.height);

              // Convert canvas to Blob and emit to server
              canvas.toBlob((blob) => {
                if (socket && blob) {
                  socket.emit('video_frame', blob);
                }
              }, 'image/jpeg');

              // Continue capturing frames
              requestAnimationFrame(sendFrame);
            }
          };

          video.addEventListener('playing', sendFrame);
        };

        captureFrames();
      } catch (error) {
        console.error('Error accessing webcam: ', error);
      }
    };

    getUserMedia();

    // Listen for processed frames from the server
    socket.on('processed_frame', (imageData) => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.src = `data:image/jpeg;base64,${imageData}`;
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
      };
    });

    // Clean up the socket connection on unmount
    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <canvas ref={canvasRef} style={{ width: '100%', height: 'auto' }}></canvas>
    </div>
  );
};

export default StreamPage