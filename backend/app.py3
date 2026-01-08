import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080", "*"]},
        r"/health": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080", "*"]},
        r"/": {"origins": ["*"]},
    },
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def build_prompt(respuestas: dict) -> str:
    """
    Pedimos 3 ciclos de FP de la Comunidad Valenciana.
    Respuesta: SOLO JSON, exactamente 3 recomendaciones.
    """
    respuestas_json = json.dumps(respuestas, ensure_ascii=False, indent=2)

    return f"""
Eres un orientador académico especializado en Formación Profesional (FP) en España.
Tienes que recomendar EXACTAMENTE 3 ciclos formativos de la COMUNIDAD VALENCIANA.

REGLAS OBLIGATORIAS:
- Devuelve SIEMPRE y SOLO un JSON válido.
- No uses markdown.
- No escribas texto fuera del JSON.
- EXACTAMENTE 3 recomendaciones (ni 2 ni 4).
- Cada recomendación debe ser un ciclo real y típico de FP en la Comunidad Valenciana.
- Si el perfil no encaja perfecto, igualmente elige los 3 mejores para el alumno según sus respuestas.

FORMATO EXACTO (no añadas campos extra):
{{
  "recomendaciones": [
    {{
      "ciclo": "Nombre del ciclo",
      "grado": "Medio o Superior",
      "familia_profesional": "Familia profesional",
      "motivo": "2-3 frases claras y personalizadas",
      "encaje": 0-100
    }},
    {{
      "ciclo": "...",
      "grado": "...",
      "familia_profesional": "...",
      "motivo": "...",
      "encaje": 0-100
    }},
    {{
      "ciclo": "...",
      "grado": "...",
      "familia_profesional": "...",
      "motivo": "...",
      "encaje": 0-100
    }}
  ]
}}

RESPUESTAS DEL ALUMNO (JSON):
{respuestas_json}
""".strip()


def invoke_nova(prompt: str) -> str:
    body = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "inferenceConfig": {
            "maxTokens": 900,
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

    try:
        return data["output"]["message"]["content"][0]["text"]
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def try_parse_json(text: str):
    # Intento directo
    try:
        return json.loads(text), None
    except Exception:
        pass

    # Rescate simple: primer bloque { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"No se pudo parsear JSON rescatado: {e}"

    return None, "No se encontró un bloque JSON en la respuesta."


def normalize_output(obj: dict) -> dict:
    """
    Asegura que devolvemos SOLO las 3 recomendaciones finales.
    """
    recs = obj.get("recomendaciones", [])
    if not isinstance(recs, list):
        recs = []

    # Recorta a 3 por seguridad
    recs = recs[:3]

    # Normaliza campos mínimos
    cleaned = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        cleaned.append({
            "ciclo": str(r.get("ciclo", "")).strip(),
            "grado": str(r.get("grado", "")).strip(),
            "familia_profesional": str(r.get("familia_profesional", "")).strip(),
            "motivo": str(r.get("motivo", "")).strip(),
            "encaje": int(r.get("encaje", 0)) if str(r.get("encaje", "0")).isdigit() else 0
        })

    # Si faltan (por si el modelo hace cosas raras), seguimos devolviendo lo que haya
    return {"recomendaciones": cleaned}


@app.get("/")
def index():
    return jsonify({
        "mensaje": "Backend activo (Nova Pro).",
        "endpoints": ["/health", "/api/orientacion"]
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "region": AWS_REGION,
        "modelId": MODEL_ID
    })


@app.post("/api/orientacion")
def orientacion():
    respuestas = request.get_json(silent=True)

    if not respuestas or not isinstance(respuestas, dict):
        return jsonify({
            "ok": False,
            "error": "No llegó JSON válido. Asegúrate de enviar Content-Type: application/json"
        }), 400

    prompt = build_prompt(respuestas)

    try:
        output_text = invoke_nova(prompt)

        parsed, parse_error = try_parse_json(output_text)
        if parsed is None:
            return jsonify({
                "ok": False,
                "error": "Nova no devolvió JSON parseable.",
                "detalle": parse_error,
                "raw": output_text
            }), 200

        final_data = normalize_output(parsed)

        return jsonify({
            "ok": True,
            "data": final_data
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
