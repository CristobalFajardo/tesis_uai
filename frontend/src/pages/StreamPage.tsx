import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import { serverUrl } from '../config';

const FRAME_INTERVAL = 100;
const QUALITY = .5;

const StreamPage = () => {
  const canvasRef = useRef(null);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const socket = io(serverUrl); // Adjust the URL according to your server settings
    setSocket(socket);

    const getUserMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
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

            canvas.toBlob((blob) => {
              if (socket && blob) {
                socket.emit('video_frame', blob);
              }
            }, 'image/jpeg', QUALITY);
          }
          // Schedule the next frame capture
          setTimeout(sendFrame, FRAME_INTERVAL);
        };

        video.addEventListener('playing', sendFrame);
      } catch (error) {
        console.error('Error accessing webcam: ', error);
      }
    };

    getUserMedia();

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

    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  return (
    <div className="flex flex-col justify-center items-center h-screen">
      <canvas
        ref={canvasRef}
        width="800"
        height="600"
        style={{
          width: '800px',
          height: '600px',
          transform: 'scaleX(-1)',
        }}></canvas>

        <div className="text-2xl pt-8">
          Gracias
        </div>
    </div>
  );
};

export default StreamPage