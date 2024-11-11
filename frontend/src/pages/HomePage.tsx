import { Link } from 'react-router-dom'

export default function HomePage () {
  return (
    <div className="p-1 w-full h-screen flex items-center justify-center">

      <div className="max-w-2xl mx-auto p-2">
        <h1 className="text-4xl mb-8">
          Combinando visión artificial y modelos de lenguaje para interpretación de lengua de señas
        </h1>
        <h2 className="text-2xl mb-4">
          Proyecto de Tesis MIA. Universidad Adolfo Ibáñez
        </h2>

        <div className="mt-2 text-slate-500">
          <h4>Desarrollado por:</h4>
          <ul className="list list-disc pl-4 my-2">
            <li>
              Cristóbal Fajardo
            </li>
            <li>
              Pablo Gaete
            </li>
            <li>
              Rubén Torres
            </li>
          </ul>
        </div>

        <div className="mt-10 w-80 mx-auto">
          <Link to="/stream" className="btn btn-primary btn-lg">
            Comenzar
          </Link>
        </div>
      </div>
    </div>
  )
}