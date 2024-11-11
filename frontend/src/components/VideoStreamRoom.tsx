import {
  ControlBar,
  LiveKitRoom,
  LiveKitRoomProps,
} from "@livekit/components-react";
import "@livekit/components-styles";
import WebCamStream from "./WebCamStream";
import { serverUrl } from "../config";


const SafeLiveKitRoom = LiveKitRoom as unknown as React.FC<LiveKitRoomProps>;

export default function VideoStreamRoom({ token }: { token: string }) {
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
