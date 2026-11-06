import os
import time
from collections import defaultdict, deque

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
# A Groq desativou o llama-3.3-70b-versatile em agosto de 2026.
# Este e o substituto recomendado (podes mudar com a variavel GROQ_MODEL no Vercel).
MODELO = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
ORIGENS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
LIMITE = int(os.environ.get("LIMITE_POR_MINUTO", "15"))
MAX_CARACTERES = 8000

# Só o teu site pode chamar este servidor (define ALLOWED_ORIGINS no Vercel)
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


def montar_corpo(msgs, modo_json):
    corpo = {"model": MODELO, "messages": msgs, "temperature": 0.4, "max_completion_tokens": 1500}
    if MODELO.startswith("openai/gpt-oss"):
        corpo["reasoning_effort"] = "low"      # pensa pouco, responde depressa
        corpo["include_reasoning"] = False     # devolve só a resposta final
    if modo_json:
        corpo["response_format"] = {"type": "json_object"}
    return corpo


def chamar_groq(corpo):
    r = requests.post(GROQ_URL, headers={"Authorization": "Bearer " + GROQ_KEY}, json=corpo, timeout=30)
    if r.status_code == 400:  # se algum parametro opcional nao for aceite, tenta sem eles
        simples = {k: v for k, v in corpo.items() if k not in ("response_format", "include_reasoning", "reasoning_effort")}
        r = requests.post(GROQ_URL, headers={"Authorization": "Bearer " + GROQ_KEY}, json=simples, timeout=30)
    return r


@app.get("/")
def estado():
    return jsonify(estado="ok")


@app.get("/api/teste")
def teste():
    """Diagnostico: diz se a chave existe e o que a Groq responde (nunca mostra a chave)."""
    if excedeu_limite(ip_do_cliente()):
        return jsonify(erro="limite"), 429
    info = {"modelo": MODELO, "chave": "presente" if GROQ_KEY else "em falta"}
    if not GROQ_KEY:
        return jsonify(info), 500
    try:
        r = chamar_groq(montar_corpo([{"role": "user", "content": "Responde apenas: ok"}], False))
        info["groq_status"] = r.status_code
        info["resposta"] = r.json()["choices"][0]["message"]["content"] if r.ok else r.text[:300]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        info["erro"] = type(e).__name__
        return jsonify(info), 502
    return jsonify(info)


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
    try:
        r = chamar_groq(montar_corpo(msgs, bool(dados.get("json"))))
        r.raise_for_status()
        texto = r.json()["choices"][0]["message"]["content"] or ""
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return jsonify(erro="falha no modelo"), 502
    if not texto.strip():
        return jsonify(erro="resposta vazia"), 502
    return jsonify(text=texto)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
