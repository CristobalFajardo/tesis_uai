import { Link } from 'react-router-dom'

export default function HomePage () {
  return (
    <div className="p-1">
      <h1 className="text-lg">
        Combinando visión artificial y modelos de lenguaje para interpretación de lenguaje de señas
      </h1>


      <Link to="/" className="btn btn-primary">
        Comenzar
      </Link>
    </div>
  )
}