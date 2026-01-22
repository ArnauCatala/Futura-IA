import csv
import io
import json
import os
import time
import urllib.request
import threading
from typing import Dict, List, Set, Tuple, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from dotenv import load_dotenv

# Fuzzy match (mejor) con RapidFuzz si está instalado
try:
    from rapidfuzz import fuzz, process  # type: ignore
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False
    import difflib

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

# CSV principal (2025) + fallback (2024)
GVA_FP_CSV_URL_2025 = os.getenv(
    "GVA_FP_CSV_URL_2025",
    "https://dadesobertes.gva.es/dataset/a2183efe-f62c-48ec-bdbe-22a4b63c3832/resource/79af67de-71a2-48b1-bd6d-57a2996e2669/download/alumnos-matriculados-fp_2025.csv"
)
GVA_FP_CSV_URL_2024 = os.getenv(
    "GVA_FP_CSV_URL_2024",
    "https://dadesobertes.gva.es/dataset/04b2a721-9256-40f9-b45e-fa0c8e7000b5/resource/7ac929a5-9138-4791-924b-2f1f4c6777fc/download/alumnos-matriculados-fp_2024.csv"
)

# ✅ NUEVO: CSV centros docentes (dirección/teléfono/web/coords)
# Recurso oficial:
# https://dadesobertes.gva.es/dataset/68eb1d94-76d3-4305-8507-e1aab7717d0e/resource/1aa53c3a-4639-41aa-ac85-d58254c428c0/download/centros-docentes-de-la-comunitat-valenciana.csv
# (viene del dataset edu-centros) :contentReference[oaicite:2]{index=2}
GVA_CENTROS_CSV_URL = os.getenv(
    "GVA_CENTROS_CSV_URL",
    "https://dadesobertes.gva.es/dataset/68eb1d94-76d3-4305-8507-e1aab7717d0e/resource/1aa53c3a-4639-41aa-ac85-d58254c428c0/download/centros-docentes-de-la-comunitat-valenciana.csv"
)

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

# ----------------------------
# Cache / índices FP
# ----------------------------
_cycle_city_index: Dict[Tuple[str, str], Set[str]] = {}
_cycle_anygrade_index: Dict[str, Set[str]] = {}
_municipios_all: List[str] = []

_last_index_load_ts: float = 0.0
_last_index_source: str = ""
_last_index_error: str = ""

_INDEX_TTL_SECONDS = 24 * 3600  # refresca 1 vez al día
_fp_index_lock = threading.Lock()


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# ----------------------------
# Prompt Bedrock Nova Pro
# ----------------------------
def build_prompt(respuestas: dict) -> str:
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
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 1100, "temperature": 0.35, "topP": 0.9},
    }

    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
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
        candidate = text[start : end + 1]
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
        salidas = [str(s).strip() for s in salidas if str(s).strip()][:6]

        cleaned.append(
            {
                "ciclo": str(r.get("ciclo", "")).strip(),
                "grado": str(r.get("grado", "")).strip(),
                "familia_profesional": str(r.get("familia_profesional", "")).strip(),
                "motivo": str(r.get("motivo", "")).strip(),
                "salidas_laborales": salidas,
                "rango_salarial": str(r.get("rango_salarial", "")).strip(),
                "encaje": safe_int(r.get("encaje", 0), 0),
            }
        )

    return {"nota_salarios": nota, "recomendaciones": cleaned}


# ----------------------------
# CSV GVA FP (ciudades por ciclo + municipios CV)
# ----------------------------
def _download_csv(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (ProyectoIA-FP/1.0)",
            "Accept": "text/csv,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read()


def _read_gva_csv_bytes() -> Tuple[bytes, str, str]:
    """
    Devuelve (csv_bytes, source_url, error_msg)
    """
    csv_bytes = None
    source = ""
    last_err = None

    for url in (GVA_FP_CSV_URL_2025, GVA_FP_CSV_URL_2024):
        try:
            csv_bytes = _download_csv(url)
            source = url
            break
        except Exception as e:
            last_err = e

    if csv_bytes is None:
        return b"", "", f"No se pudo descargar CSV GVA (2025/2024). Error: {last_err}"

    return csv_bytes, source, ""


def _load_gva_fp_index(force: bool = False) -> None:
    global _cycle_city_index, _cycle_anygrade_index, _municipios_all
    global _last_index_load_ts, _last_index_source, _last_index_error

    now = time.time()
    if (not force) and _cycle_city_index and (now - _last_index_load_ts) < _INDEX_TTL_SECONDS:
        return

    with _fp_index_lock:
        now = time.time()
        if (not force) and _cycle_city_index and (now - _last_index_load_ts) < _INDEX_TTL_SECONDS:
            return

        _last_index_error = ""
        _last_index_source = ""

        csv_bytes, source, err = _read_gva_csv_bytes()
        if err:
            _cycle_city_index = {}
            _cycle_anygrade_index = {}
            _municipios_all = []
            _last_index_load_ts = now
            _last_index_source = ""
            _last_index_error = err
            return

        try:
            text = csv_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = csv_bytes.decode("latin-1")

        f = io.StringIO(text)

        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames and len(reader.fieldnames) == 1 and "," in reader.fieldnames[0]:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=",")

        needed = {"NOM_CICLO", "NOM_MUN", "NOM_GRADO"}
        fields = set(reader.fieldnames or [])
        if not needed.issubset(fields):
            _cycle_city_index = {}
            _cycle_anygrade_index = {}
            _municipios_all = []
            _last_index_load_ts = now
            _last_index_source = source
            _last_index_error = (
                f"CSV descargado pero faltan columnas: {sorted(list(needed - fields))}. "
                f"Columnas detectadas: {sorted(list(fields))[:30]}"
            )
            return

        new_index: Dict[Tuple[str, str], Set[str]] = {}
        new_anygrade: Dict[str, Set[str]] = {}
        municipios_set: Set[str] = set()

        for row in reader:
            nom_ciclo = _norm(row.get("NOM_CICLO", ""))
            nom_grado = _norm(row.get("NOM_GRADO", ""))
            nom_mun_raw = (row.get("NOM_MUN", "") or "").strip()

            if nom_mun_raw:
                municipios_set.add(nom_mun_raw)

            if not nom_ciclo or not nom_mun_raw:
                continue

            new_index.setdefault((nom_ciclo, nom_grado), set()).add(nom_mun_raw)
            new_anygrade.setdefault(nom_ciclo, set()).add(nom_mun_raw)

        _cycle_city_index = new_index
        _cycle_anygrade_index = new_anygrade
        _municipios_all = sorted(municipios_set)

        _last_index_load_ts = now
        _last_index_source = source
        _last_index_error = ""


def _best_match_with_rapidfuzz(query: str, choices: List[str]) -> Tuple[str, int]:
    best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)  # type: ignore
    if not best:
        return "", 0
    matched_name, score, _ = best
    return matched_name, int(score)


def _best_match_with_difflib(query: str, choices: List[str]) -> Tuple[str, int]:
    if not choices:
        return "", 0
    best_list = difflib.get_close_matches(query, choices, n=1, cutoff=0.0)
    if not best_list:
        return "", 0
    matched = best_list[0]
    score = int(difflib.SequenceMatcher(None, query, matched).ratio() * 100)
    return matched, score


def _best_cycle_match(ciclo_n: str) -> Tuple[str, int]:
    if not ciclo_n or not _cycle_anygrade_index:
        return "", 0
    choices = list(_cycle_anygrade_index.keys())
    if RAPIDFUZZ_OK:
        return _best_match_with_rapidfuzz(ciclo_n, choices)
    return _best_match_with_difflib(ciclo_n, choices)


def _best_grado_match(grado_n: str, grados_choices: List[str]) -> Tuple[str, int]:
    if not grado_n or not grados_choices:
        return "", 0
    if RAPIDFUZZ_OK:
        return _best_match_with_rapidfuzz(grado_n, grados_choices)
    return _best_match_with_difflib(grado_n, grados_choices)


def _find_cities_for_cycle(ciclo: str, grado: str) -> Tuple[List[str], dict]:
    ciclo_n = _norm(ciclo)
    grado_n = _norm(grado)

    _load_gva_fp_index(force=False)

    info = {
        "rapidfuzz": RAPIDFUZZ_OK,
        "matched_ciclo": "",
        "match_score": 0,
        "matched_grado": "",
        "grado_score": 0,
    }

    if not _cycle_city_index and not _cycle_anygrade_index:
        return [], info

    if ciclo_n and grado_n:
        exact = _cycle_city_index.get((ciclo_n, grado_n))
        if exact:
            info["matched_ciclo"] = ciclo_n
            info["match_score"] = 100
            info["matched_grado"] = grado_n
            info["grado_score"] = 100
            return sorted(exact), info

    if ciclo_n:
        anyg = _cycle_anygrade_index.get(ciclo_n)
        if anyg:
            info["matched_ciclo"] = ciclo_n
            info["match_score"] = 100
            return sorted(anyg), info

    matched_ciclo, score = _best_cycle_match(ciclo_n)
    info["matched_ciclo"] = matched_ciclo
    info["match_score"] = score

    if not matched_ciclo or score < 55:
        return [], info

    if grado_n:
        grados_posibles: List[str] = []
        for (c_name, g_name) in _cycle_city_index.keys():
            if c_name == matched_ciclo:
                grados_posibles.append(g_name)

        if grados_posibles:
            matched_grado, gscore = _best_grado_match(grado_n, grados_posibles)
            info["matched_grado"] = matched_grado
            info["grado_score"] = gscore

            if matched_grado and gscore >= 55:
                cities = _cycle_city_index.get((matched_ciclo, matched_grado), set())
                if cities:
                    return sorted(cities), info

    cities_any = _cycle_anygrade_index.get(matched_ciclo, set())
    return sorted(cities_any), info


# ----------------------------
# ✅ NUEVO: Índice de CENTROS (dirección/teléfono/web)
# ----------------------------
_centros_by_localidad: Dict[str, List[dict]] = {}
_centros_last_load_ts: float = 0.0
_centros_last_source: str = ""
_centros_last_error: str = ""
_CENTROS_TTL_SECONDS = 7 * 24 * 3600  # 7 días (sobra, el dataset se actualiza continuo)
_centros_lock = threading.Lock()


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _centro_fp_score(row: dict) -> int:
    """
    No siempre hay un campo 'imparte FP'. Así que ordenamos "probables FP" arriba
    por heurística (CIPFP, FP, IES, etc.).
    """
    hay = " ".join(
        [
            str(row.get("denominacion_generica_es", "")),
            str(row.get("denominacion_especifica", "")),
            str(row.get("denominacion", "")),
        ]
    ).upper()

    score = 0
    if "CIPFP" in hay or "CENTRE INTEGRAT" in hay or "CENTRO INTEGRADO" in hay:
        score += 50
    if "FORMACIÓN PROFESIONAL" in hay or "FORMACION PROFESIONAL" in hay:
        score += 35
    if "FP" in hay:
        score += 10
    if "IES" in hay or "INSTITUTO" in hay:
        score += 6
    return score


def _download_centros_csv() -> bytes:
    return _download_csv(GVA_CENTROS_CSV_URL)


def _load_centros_index(force: bool = False) -> None:
    global _centros_by_localidad, _centros_last_load_ts, _centros_last_source, _centros_last_error

    now = time.time()
    if (not force) and _centros_by_localidad and (now - _centros_last_load_ts) < _CENTROS_TTL_SECONDS:
        return

    with _centros_lock:
        now = time.time()
        if (not force) and _centros_by_localidad and (now - _centros_last_load_ts) < _CENTROS_TTL_SECONDS:
            return

        _centros_last_error = ""
        _centros_last_source = ""

        try:
            csv_bytes = _download_centros_csv()
            _centros_last_source = GVA_CENTROS_CSV_URL
        except Exception as e:
            _centros_by_localidad = {}
            _centros_last_load_ts = now
            _centros_last_error = f"No se pudo descargar CSV centros. Error: {e}"
            return

        try:
            text = csv_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = csv_bytes.decode("latin-1", errors="replace")

        f = io.StringIO(text)
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames and len(reader.fieldnames) == 1 and "," in reader.fieldnames[0]:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=",")

        # Campos relevantes (según diccionario del recurso) :contentReference[oaicite:3]{index=3}
        needed = {"denominacion", "direccion", "localidad"}
        fields = set(reader.fieldnames or [])
        if not needed.issubset(fields):
            _centros_by_localidad = {}
            _centros_last_load_ts = now
            _centros_last_error = (
                f"CSV centros descargado pero faltan columnas: {sorted(list(needed - fields))}. "
                f"Columnas detectadas: {sorted(list(fields))[:30]}"
            )
            return

        idx: Dict[str, List[dict]] = {}

        for row in reader:
            localidad_raw = (row.get("localidad", "") or "").strip()
            if not localidad_raw:
                continue

            item = {
                "codigo": row.get("codigo", ""),
                "nombre": (row.get("denominacion", "") or "").strip(),
                "tipo": (row.get("denominacion_generica_es", "") or "").strip(),
                "regimen": (row.get("regimen", "") or "").strip(),
                "direccion": (row.get("direccion", "") or "").strip(),
                "numero": (row.get("numero", "") or "").strip(),
                "cp": row.get("codigo_postal", ""),
                "localidad": localidad_raw,
                "provincia": (row.get("provincia", "") or "").strip(),
                "telefono": str(row.get("telefono", "") or "").strip(),
                "url": (row.get("url_es", "") or "").strip(),
                "lat": _safe_float(row.get("latitud", None)),
                "lon": _safe_float(row.get("longitud", None)),
            }

            # Limpieza rápida
            if item["telefono"] in ("0", "0.0", "nan", "None"):
                item["telefono"] = ""
            if item["url"] in ("nan", "None"):
                item["url"] = ""

            key = _norm(localidad_raw)
            idx.setdefault(key, []).append(item)

        # Orden interno: probables FP primero, luego por nombre
        for k, arr in idx.items():
            arr.sort(key=lambda r: (-_centro_fp_score(r), (r.get("nombre") or "").lower()))

        _centros_by_localidad = idx
        _centros_last_load_ts = now
        _centros_last_error = ""


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def root():
    return jsonify(
        {
            "mensaje": "Backend activo (Bedrock Nova).",
            "endpoints": [
                "/health",
                "/api/orientacion",
                "/api/ciudades",
                "/api/ciudades/debug",
                "/api/municipios",
                "/api/centros",
            ],
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True, "region": AWS_REGION, "modelId": MODEL_ID, "rapidfuzz": RAPIDFUZZ_OK})


@app.post("/api/orientacion")
def orientacion():
    respuestas = request.get_json(silent=True)

    if not respuestas or not isinstance(respuestas, dict):
        return jsonify({"ok": False, "error": "No llegó JSON válido. Envía Content-Type: application/json"}), 400

    prompt = build_prompt(respuestas)

    try:
        output_text = invoke_nova(prompt)

        parsed, parse_error = try_parse_json(output_text)
        if parsed is None:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Nova no devolvió JSON parseable.",
                        "detalle": parse_error,
                        "raw": output_text,
                    }
                ),
                200,
            )

        final_data = normalize_output(parsed)
        return jsonify({"ok": True, "data": final_data}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/ciudades")
def ciudades():
    ciclo = request.args.get("ciclo", "").strip()
    grado = request.args.get("grado", "").strip()

    if not ciclo:
        return jsonify({"ok": False, "error": "Falta parámetro 'ciclo'"}), 400

    try:
        cities, match_info = _find_cities_for_cycle(ciclo, grado)

        extra = {}
        if _last_index_source:
            extra["source"] = _last_index_source
        if _last_index_error:
            extra["warning"] = _last_index_error

        return jsonify(
            {
                "ok": True,
                "ciclo": ciclo,
                "grado": grado,
                "ciudades": cities,
                "count": len(cities),
                "match": match_info,
                **extra,
            }
        ), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/municipios")
def municipios():
    try:
        _load_gva_fp_index(force=False)

        if _last_index_error:
            return jsonify({"ok": False, "error": _last_index_error}), 500

        return jsonify(
            {
                "ok": True,
                "count": len(_municipios_all),
                "municipios": _municipios_all,
                "source": _last_index_source,
            }
        ), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/ciudades/debug")
def ciudades_debug():
    _load_gva_fp_index(force=True)
    return jsonify(
        {
            "ok": True,
            "index_pairs": len(_cycle_city_index),
            "index_cycles": len(_cycle_anygrade_index),
            "municipios_count": len(_municipios_all),
            "source": _last_index_source,
            "error": _last_index_error,
            "rapidfuzz": RAPIDFUZZ_OK,
        }
    ), 200


# ✅ NUEVO endpoint: centros por municipio
@app.get("/api/centros")
def centros():
    municipio = request.args.get("municipio", "").strip()
    limit = request.args.get("limit", "").strip()
    only_fp = request.args.get("only_fp", "0").strip()  # "1" para filtrar “probables FP”

    if not municipio:
        return jsonify({"ok": False, "error": "Falta parámetro 'municipio'"}), 400

    try:
        _load_centros_index(force=False)

        if _centros_last_error:
            return jsonify({"ok": False, "error": _centros_last_error}), 500

        key = _norm(municipio)
        items = _centros_by_localidad.get(key, [])

        # Si piden solo FP, filtramos por score mínimo
        if only_fp == "1":
            items = [x for x in items if _centro_fp_score(x) >= 10]

        # limit
        try:
            lim = int(limit) if limit else 25
        except Exception:
            lim = 25
        lim = max(1, min(lim, 100))

        out = items[:lim]

        return jsonify(
            {
                "ok": True,
                "municipio": municipio,
                "count": len(out),
                "centros": out,
                "source": _centros_last_source,
            }
        ), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
