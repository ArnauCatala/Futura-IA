import json
import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import boto3

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

# --- Configuración Flask ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "web"))

app = Flask(
    __name__,
    static_folder=WEB_DIR,
    static_url_path=""
)

# --- Cliente Bedrock ---
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# --- Funciones ---
def build_prompt(respuestas: dict) -> str:
    """Prompt para que Bedrock devuelva un JSON limpio."""
    return f"""
Eres un Chatbot de Orientación Laboral...
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
    return data["results"][0]["outputText"]

# --- Rutas Frontend ---
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

# --- Rutas API ---
@app.get("/health")
def health():
    return jsonify({"ok": True, "region": AWS_REGION, "modelId": MODEL_ID})

@app.post("/api/orientacion")
def orientacion():
    respuestas = request.get_json(silent=True)
    if not respuestas:
        return jsonify({"ok": False, "error": "No llegó JSON válido"}), 400

    prompt = build_prompt(respuestas)

    try:
        output_text = invoke_titan(prompt)
        try:
            parsed = json.loads(output_text)
            return jsonify({"ok": True, "data": parsed})
        except json.JSONDecodeError:
            return jsonify({"ok": False, "error": "El modelo no devolvió JSON puro.", "raw": output_text}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# --- Run local ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
