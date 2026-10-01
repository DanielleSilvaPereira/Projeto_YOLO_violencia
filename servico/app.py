"""Serviço HTTP de IA do Safe Evidence (contrato em docs/contrato-ia.md).

    uvicorn servico.app:app --host 0.0.0.0 --port 8090

Variáveis de ambiente: veja .env.example.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response

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


def _autorizar(authorization: str):
    if IA_TOKEN and authorization != f"Bearer {IA_TOKEN}":
        raise HTTPException(401, "token inválido")


@app.post("/analisar", status_code=202)
def analisar(pedido: dict, authorization: str = Header(default="")):
    _autorizar(authorization)
    try:
        novo = processador.enfileirar(pedido)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"aceito": True, "id_gravacao": pedido["id_gravacao"],
            "observacao": None if novo else "já em análise"}


@app.post("/laudo")
def laudo(pedido: dict, authorization: str = Header(default="")):
    """Devolve o laudo em PDF ("Download de Segurança").

    Corpo: {"resultado": <JSON do callback>, "metadados": {...}, "video_url": "...(opcional)"}
    Com video_url, o vídeo é baixado, o hash conferido com metadados.hash_sha256
    e os quadros de pico entram no PDF.
    """
    _autorizar(authorization)
    from ia.laudo import gerar_laudo

    resultado = pedido.get("resultado")
    meta = pedido.get("metadados") or {}
    if not isinstance(resultado, dict) or resultado.get("status", "concluido") != "concluido":
        raise HTTPException(400, "envie 'resultado' com uma análise concluída")
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as d:
        video = None
        if pedido.get("video_url"):
            video = os.path.join(d, "video.mp4")
            try:
                h = processador.baixar(pedido["video_url"], video)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(502, f"não consegui baixar o vídeo: {e}")
            if meta.get("hash_sha256") and h.lower() != meta["hash_sha256"].lower():
                raise HTTPException(409, "hash do vídeo não confere com o registrado")
        codigo = gerar_laudo(resultado, buf, meta, video=video, cfg=processador.cfg)
    nome = f"laudo_gravacao_{meta.get('id_gravacao', 'x')}.pdf"
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{nome}"',
                             "X-Codigo-Verificacao": codigo})
