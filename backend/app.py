import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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
    Pedimos EXACTAMENTE 3 ciclos de FP en la Comunidad Valenciana,
    y para cada uno: trabajos relacionados + rango salarial estimado.
    """
    respuestas_json = json.dumps(respuestas, ensure_ascii=False, indent=2)

    return f"""
Eres un orientador académico experto en Formación Profesional (FP) en España.
Tu tarea: recomendar EXACTAMENTE 3 ciclos formativos de la COMUNIDAD VALENCIANA.

REGLAS OBLIGATORIAS:
- Devuelve SIEMPRE y SOLO un JSON válido.
- No uses markdown.
- No escribas texto fuera del JSON.
- EXACTAMENTE 3 recomendaciones.
- Deben ser ciclos reales y habituales de FP (CV).
- Incluye salidas laborales concretas y rangos salariales ORIENTATIVOS (no cifras “oficiales”).
- Rangos salariales: en euros y preferiblemente ANUAL BRUTO (p. ej. "18.000–24.000 €/año").
- Añade un campo "nota_salarios" aclarando que son estimaciones.

FORMATO EXACTO (no añadas campos extra):
{{
  "nota_salarios": "Texto breve aclarando que son rangos estimados en España/CV.",
  "recomendaciones": [
    {{
      "ciclo": "Nombre del ciclo",
      "grado": "Medio o Superior",
      "familia_profesional": "Familia profesional",
      "motivo": "2-3 frases personalizadas",
      "salidas_laborales": ["Trabajo 1", "Trabajo 2", "Trabajo 3"],
      "rango_salarial": "18.000–24.000 €/año",
      "encaje": 0-100
    }},
    {{
      "ciclo": "...",
      "grado": "...",
      "familia_profesional": "...",
      "motivo": "...",
      "salidas_laborales": ["...","...","..."],
      "rango_salarial": "...",
      "encaje": 0-100
    }},
    {{
      "ciclo": "...",
      "grado": "...",
      "familia_profesional": "...",
      "motivo": "...",
      "salidas_laborales": ["...","...","..."],
      "rango_salarial": "...",
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
            "maxTokens": 1100,
            "temperature": 0.35,
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
    try:
        return json.loads(text), None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"No se pudo parsear JSON rescatado: {e}"

    return None, "No se encontró un bloque JSON en la respuesta."


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def normalize_output(obj: dict) -> dict:
    """
    Asegura salida FINAL:
    - 3 recomendaciones
    - campos limpios
    """
    nota = str(obj.get("nota_salarios", "")).strip()
    if not nota:
        nota = "Rangos salariales orientativos (pueden variar por provincia, experiencia y empresa)."

    recs = obj.get("recomendaciones", [])
    if not isinstance(recs, list):
        recs = []

    recs = recs[:3]

    cleaned = []
    for r in recs:
        if not isinstance(r, dict):
            continue

        salidas = r.get("salidas_laborales", [])
        if isinstance(salidas, str):
            salidas = [salidas]
        if not isinstance(salidas, list):
            salidas = []
        salidas = [str(s).strip() for s in salidas if str(s).strip()][:6]  # máximo 6 por estética

        cleaned.append({
            "ciclo": str(r.get("ciclo", "")).strip(),
            "grado": str(r.get("grado", "")).strip(),
            "familia_profesional": str(r.get("familia_profesional", "")).strip(),
            "motivo": str(r.get("motivo", "")).strip(),
            "salidas_laborales": salidas,
            "rango_salarial": str(r.get("rango_salarial", "")).strip(),
            "encaje": safe_int(r.get("encaje", 0), 0)
        })

    return {"nota_salarios": nota, "recomendaciones": cleaned}


@app.get("/")
def index():
    return jsonify({
        "mensaje": "Backend activo (Bedrock Nova).",
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
            "error": "No llegó JSON válido. Envía Content-Type: application/json"
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

        return jsonify({"ok": True, "data": final_data}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

