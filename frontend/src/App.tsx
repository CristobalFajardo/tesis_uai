import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import StreamPage from './pages/StreamPage'
import { serverUrl } from './config'

function App() {
  const [isLoading, setIsLoading] = useState(true)
  const [token, setToken] = useState('')

  useEffect(() => {
    const loadToken = async () => {
      try {
        const response = await fetch(`${serverUrl}/getToken`)
        const data = await response.json()
        setToken(data.token)
        setIsLoading(false)
      } catch (error) {
        console.error('Error fetching token:', error)
      }
    }

    loadToken()
  }, [])

  if (isLoading) {
    return <div>Loading...</div>
  }

  return (
    <Router>
      <Routes>
        <Route index element={<HomePage />} />
        <Route path="/stream" element={<StreamPage token={token} />} />
      </Routes>
    </Router>
  )
}

export default App
