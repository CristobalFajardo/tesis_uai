from openai import OpenAI
client = OpenAI(api_key = "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")

# Palabras aisladas que quieres relacionar
palabras_aisladas = "Por favor, Gracias, Nombre,Yo, Tú, Bien, Sentir, Cómo, Gustar, Ver, Hola, Chao, Bueno, Día, Tarde, Noche, Enamorado, Comer, Conversar, Médico"

def obtener_respuesta(prompt):
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
          }
        ]
      }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
      "type": "text"
    }
  )
  return response
  

promt1 = "Responde un ejemplo sin explicación al relacionar las palabras aisladas "
respuesta = obtener_respuesta(promt1 + "\"Pablo\" \"nombre\" \"yo\"")

print("Respuesta:", respuesta.choices[0].message.content.strip())