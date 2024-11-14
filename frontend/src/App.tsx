import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import StreamPage from './pages/StreamPage'

function App() {

  return (
    <Router>
      <Routes>
        <Route index element={<HomePage />} />
        <Route path="/stream" element={<StreamPage />} />
      </Routes>
    </Router>
  )
}

export default App
