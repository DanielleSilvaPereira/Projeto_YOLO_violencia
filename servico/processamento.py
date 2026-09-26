"""Lógica do serviço, separada do FastAPI (dá pra testar sem subir servidor).

Segue docs/contrato-ia.md: baixa o vídeo, confere o SHA-256, roda o modelo e
manda o resultado para o callback_url do backend.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import requests

from ia.config import ConfigInferencia
from ia.video import ClassificadorFrames, analisar_video

log = logging.getLogger("servico_ia")

CAMPOS_OBRIGATORIOS = ("id_gravacao", "video_url", "hash_sha256", "callback_url")


class Processador:
    def __init__(self, cfg: ConfigInferencia | None = None, callback_token: str | None = None,
                 classificar: Callable | None = None, max_paralelo: int = 1):
        self.cfg = cfg or ConfigInferencia()
        self.callback_token = callback_token if callback_token is not None else os.environ.get("IA_CALLBACK_TOKEN", "")
        self._classificar = classificar
        self._lock_modelo = threading.Lock()
        self._em_andamento: set = set()
        self._lock_ids = threading.Lock()
        # fila: a análise usa GPU/CPU pesado, então por padrão uma de cada vez
        self._fila = ThreadPoolExecutor(max_workers=max_paralelo, thread_name_prefix="analise")

    # ---------- modelo ----------
    @property
    def classificar(self):
        if self._classificar is None:
            with self._lock_modelo:
                if self._classificar is None:
                    log.info("carregando modelo %s", self.cfg.pesos)
                    self._classificar = ClassificadorFrames(self.cfg)
        return self._classificar

    # ---------- fila ----------
    def enfileirar(self, pedido: dict) -> bool:
        """Devolve False se esse id já está sendo analisado (pedido repetido)."""
        faltando = [c for c in CAMPOS_OBRIGATORIOS if not pedido.get(c)]
        if faltando:
            raise ValueError(f"campos obrigatórios ausentes: {', '.join(faltando)}")
        vid = pedido["id_gravacao"]
        with self._lock_ids:
            if vid in self._em_andamento:
                return False
            self._em_andamento.add(vid)
        self._fila.submit(self._executar, pedido)
        return True

    def _executar(self, pedido: dict):
        try:
            resultado = self.processar(pedido)
            self.enviar_callback(pedido["callback_url"], resultado)
        except Exception:  # nunca derruba a thread da fila
            log.exception("falha inesperada na gravação %s", pedido.get("id_gravacao"))
        finally:
            with self._lock_ids:
                self._em_andamento.discard(pedido["id_gravacao"])

    # ---------- etapas ----------
    @staticmethod
    def baixar(url: str, destino: str) -> str:
        """Baixa em streaming calculando o SHA-256 no caminho."""
        h = hashlib.sha256()
        with requests.get(url, stream=True, timeout=(10, 120)) as r:
            r.raise_for_status()
            with open(destino, "wb") as f:
                for bloco in r.iter_content(1 << 20):
                    f.write(bloco)
                    h.update(bloco)
        return h.hexdigest()

    def processar(self, pedido: dict) -> dict:
        inicio = time.time()
        vid = pedido["id_gravacao"]
        caminho = None
        try:
            fd, caminho = tempfile.mkstemp(suffix=".mp4", prefix=f"grav{vid}_")
            os.close(fd)
            h = self.baixar(pedido["video_url"], caminho)
            if h.lower() != str(pedido["hash_sha256"]).lower():
                raise ValueError(f"hash não confere: esperado {pedido['hash_sha256']}, recebido {h}")
            r = analisar_video(caminho, self.classificar, self.cfg)
            r.pop("_linha_do_tempo", None)
            r["tempo_processamento_ms"] = int((time.time() - inicio) * 1000)
            log.info("gravação %s: %s (%.2f) em %d ms", vid, r["veredicto"], r["score"],
                     r["tempo_processamento_ms"])
            return {"status": "concluido", **r, "hash_sha256_verificado": h}
        except Exception as e:
            log.warning("gravação %s falhou: %s", vid, e)
            return {"status": "erro", "erro": str(e)[:500]}
        finally:
            if caminho and os.path.exists(caminho):
                os.remove(caminho)

    def enviar_callback(self, url: str, resultado: dict, tentativas: int = 3):
        headers = {"Authorization": f"Bearer {self.callback_token}"} if self.callback_token else {}
        for n in range(1, tentativas + 1):
            try:
                r = requests.post(url, json=resultado, headers=headers, timeout=30)
                if r.status_code < 500:
                    # 4xx não adianta repetir (409 = já concluída/hash divergente, 401 = token)
                    if r.status_code >= 400:
                        log.warning("callback %s respondeu %s: %s", url, r.status_code, r.text[:200])
                    return r.status_code
            except requests.RequestException as e:
                log.warning("callback falhou (tentativa %d): %s", n, e)
            time.sleep(2 ** n)
        # se não conseguiu, o backend estoura IA_TIMEOUT_MINUTOS e pede de novo (RE08)
        return None
