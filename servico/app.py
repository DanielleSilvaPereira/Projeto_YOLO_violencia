"""Serviço HTTP de IA do Safe Evidence (contrato em docs/contrato-ia.md).

    uvicorn servico.app:app --host 0.0.0.0 --port 8090

Variáveis de ambiente: veja .env.example.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Header, HTTPException

from ia import __version__
from servico.processamento import Processador

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

IA_TOKEN = os.environ.get("IA_TOKEN", "")

app = FastAPI(title="Safe Evidence - Serviço de IA", version=__version__)
processador = Processador()


@app.on_event("startup")
def carregar_modelo():
    # carrega já na subida pra primeira análise não pagar o custo
    if os.environ.get("CARREGAR_NA_SUBIDA", "1") == "1":
        _ = processador.classificar


@app.get("/saude")
def saude():
    return {"ok": True, "versao_modelo": processador.cfg.versao_modelo,
            "modelo_carregado": processador._classificar is not None}


@app.post("/analisar", status_code=202)
def analisar(pedido: dict, authorization: str = Header(default="")):
    if IA_TOKEN and authorization != f"Bearer {IA_TOKEN}":
        raise HTTPException(401, "token inválido")
    try:
        novo = processador.enfileirar(pedido)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"aceito": True, "id_gravacao": pedido["id_gravacao"],
            "observacao": None if novo else "já em análise"}
