import VideoStreamRoom from '../components/VideoStreamRoom'

export default function StreamPage ({ token }: { token: string }) {
  return (
    <div className="p-1 w-full h-screen flex items-center justify-center">
      <VideoStreamRoom token={token} />
    </div>
  )
}
