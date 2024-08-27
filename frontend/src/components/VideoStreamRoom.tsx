import {
  ControlBar,
  LiveKitRoom,
  LiveKitRoomProps,
} from "@livekit/components-react";
import "@livekit/components-styles";
import WebCamStream from "./WebCamStream";

const serverUrl = 'wss://signlanguagelsch-t2topxeb.livekit.cloud';
const token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MjQ4MDYwNjUsImlzcyI6IkFQSWVHbXEzNWNwUXJWVyIsIm5iZiI6MTcyNDc5ODg2NSwic3ViIjoicXVpY2tzdGFydCB1c2VyIGx1cHJpZyIsInZpZGVvIjp7ImNhblB1Ymxpc2giOnRydWUsImNhblB1Ymxpc2hEYXRhIjp0cnVlLCJjYW5TdWJzY3JpYmUiOnRydWUsInJvb20iOiJxdWlja3N0YXJ0IHJvb20iLCJyb29tSm9pbiI6dHJ1ZX19.9DXKYYVJlQJPWr8FssQJWLlirA_giM0e7dhGpQLj4sc';

const SafeLiveKitRoom = LiveKitRoom as unknown as React.FC<LiveKitRoomProps>;

export default function VideoStreamRoom() {
  return (
    <SafeLiveKitRoom
      video={true}
      audio={false}
      token={token}
      serverUrl={serverUrl}
      data-lk-theme="default"
      style={{ height: '100vh' }}
    >
      <WebCamStream />
      <ControlBar
        controls={{
          microphone: false,
          screenShare: false,
          leave: false,
        }}
      />
    </SafeLiveKitRoom>
  )
}
