import os
import time
from collections import defaultdict, deque

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
MODELO = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ORIGENS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
LIMITE = int(os.environ.get("LIMITE_POR_MINUTO", "15"))
MAX_CARACTERES = 8000

# Só o teu site pode chamar este servidor (define ALLOWED_ORIGINS no Railway)
CORS(app, origins=ORIGENS or "*")
pedidos = defaultdict(deque)


def ip_do_cliente():
    return (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()


def excedeu_limite(ip):
    agora = time.time()
    fila = pedidos[ip]
    while fila and agora - fila[0] > 60:
        fila.popleft()
    if len(fila) >= LIMITE:
        return True
    fila.append(agora)
    return False


def normalizar(entrada):
    if isinstance(entrada, str):
        return [{"role": "user", "content": entrada}] if entrada.strip() else None
    if isinstance(entrada, list) and 0 < len(entrada) <= 12:
        msgs = []
        for m in entrada:
            if (not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                    or not isinstance(m.get("content"), str)):
                return None
            msgs.append({"role": m["role"], "content": m["content"]})
        return msgs
    return None


@app.get("/")
def estado():
    return jsonify(estado="ok")


@app.post("/api/sample")
def sample():
    if not GROQ_KEY:
        return jsonify(erro="GROQ_API_KEY em falta"), 500
    if excedeu_limite(ip_do_cliente()):
        return jsonify(erro="limite"), 429
    dados = request.get_json(silent=True) or {}
    msgs = normalizar(dados.get("input"))
    if not msgs or sum(len(m["content"]) for m in msgs) > MAX_CARACTERES:
        return jsonify(erro="pedido invalido"), 400
    corpo = {"model": MODELO, "messages": msgs, "temperature": 0.4, "max_tokens": 500}
    if dados.get("json"):
        corpo["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": "Bearer " + GROQ_KEY}, json=corpo, timeout=30)
        r.raise_for_status()
        texto = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return jsonify(erro="falha no modelo"), 502
    return jsonify(text=texto)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
