import {
  GridLayout,
  ParticipantTile,
  useTracks,
} from '@livekit/components-react'
import '@livekit/components-styles'
import { Track } from 'livekit-client'

const SafeParticipantTile = ParticipantTile as React.FC

export default function WebCamStream() {
  const tracks = useTracks(
    [
      {
        source: Track.Source.Camera,
        withPlaceholder: true,
      },
    ],
    {
      onlySubscribed: false,
    },
  )

  return (
    <GridLayout
      tracks={tracks}
      style={{ height: 'calc(100vh - var(--lk-control-bar-height))' }}
    >
      <SafeParticipantTile />
    </GridLayout>
  )
}