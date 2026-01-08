import json
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

app = Flask(__name__)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def build_prompt(respuestas: dict) -> str:
    """
    Prompt para que Bedrock devuelva un JSON útil y limpio.
    """
    return f"""
Eres un Chatbot de Orientación Laboral, que apoye la orientación académica y laboral del alumnado adolescente, facilitando la toma de decisiones informadas sobre su futuro formativo y profesional mediante itinerarios personalizados, interacción conversacional y análisis de intereses y competencias. 
Objetivos a cumplir:
Interactuar de forma natural con el alumnado, recogiendo información sobre intereses, habilidades, preferencias y expectativas profesionales.
Analizar las respuestas del alumnado y generar recomendaciones personalizadas de itinerarios formativos y salidas profesionales, evitando enfoques deterministas.
Fomentar el autoconocimiento y la reflexión vocacional, ayudando al alumnado a identificar sus fortalezas, motivaciones y áreas de interés mediante preguntas guiadas y dinámicas interactivas.
Generar informes automáticos de orientación adaptados al lenguaje del alumnado y del profesorado, facilitando el seguimiento tutorial y la toma de decisiones pedagógicas.

Recoger evidencias del proceso de orientación, permitiendo analizar tendencias, dudas frecuentes y necesidades formativas de manera agregada.(sin texto extra, sin markdown).

Estructura EXACTA:
{{
  "perfil_vocacional": "Resumen de 3-5 líneas",
  "opciones": [
    {{
      "nombre": "Opción 1",
      "motivo": "2-3 frases",
      "encaje": 0-100
    }}
  ],
  "recomendacion_educativa": "Qué estudiar y por qué (breve)",
  "proximos_pasos": ["Paso 1","Paso 2","Paso 3"]
}}

Respuestas del formulario (JSON):
{json.dumps(respuestas, ensure_ascii=False, indent=2)}
""".strip()

def invoke_titan(prompt: str) -> str:
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 800,
            "temperature": 0.4,
            "topP": 0.9
        }
    }

    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    raw = resp["body"].read().decode("utf-8")
    data = json.loads(raw)

    # Titan devuelve results[0].outputText
    return data["results"][0]["outputText"]

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "region": AWS_REGION,
        "modelId": MODEL_ID
    })

@app.post("/api/orientacion")
def orientacion():
    """
    Recibe JSON del frontend y llama a Bedrock.
    """
    respuestas = request.get_json(silent=True)
    if not respuestas:
        return jsonify({"ok": False, "error": "No llegó JSON válido"}), 400

    prompt = build_prompt(respuestas)

    try:
        output_text = invoke_titan(prompt)

        # Intentamos parsear el JSON devuelto por el modelo
        try:
            parsed = json.loads(output_text)
            return jsonify({"ok": True, "data": parsed})
        except json.JSONDecodeError:
            return jsonify({
                "ok": False,
                "error": "El modelo no devolvió JSON puro.",
                "raw": output_text
            }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # CORS fácil para desarrollo: permite llamadas desde index.html
    # Flask 3 + navegador: si te da guerra, te lo ajusto
    app.run(host="0.0.0.0", port=8000, debug=True)
