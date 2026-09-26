"""Leitura de vídeo, classificação dos frames com YOLOv9 e agregação."""

from __future__ import annotations

import time
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from .agregacao import ResultadoVideo, agregar
from .config import CLASSE_VIOLENCIA, ConfigInferencia


def amostrar_frames(caminho: str, fps_amostragem: float) -> Iterator[tuple[float, np.ndarray]]:
    """Gera (tempo_s, frame BGR) a cada 1/fps_amostragem segundos."""
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise ValueError(f"não consegui abrir o vídeo: {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:  # 0, NaN ou lixo no cabeçalho
        fps = 30.0
    passo = 1.0 / fps_amostragem
    proximo = 0.0
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            t = idx / fps
            idx += 1
            if t + 1e-6 < proximo:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield t, frame
            proximo += passo
            while proximo <= t:  # vídeo com fps baixo: não acumula atraso
                proximo += passo
    finally:
        cap.release()


def duracao_video(caminho: str) -> float | None:
    cap = cv2.VideoCapture(caminho)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and n and fps == fps and n > 0:
            return float(n / fps)
        return None
    finally:
        cap.release()


class ClassificadorFrames:
    """Envolve o modelo YOLO de classificação e devolve P(violência) por frame."""

    def __init__(self, cfg: ConfigInferencia):
        from ultralytics import YOLO  # import tardio: os testes não precisam de torch

        self.cfg = cfg
        self.modelo = YOLO(cfg.pesos, task="classify")
        nomes = {int(k): v for k, v in self.modelo.names.items()}
        idx = [k for k, v in nomes.items() if v == CLASSE_VIOLENCIA]
        if not idx:
            raise ValueError(f"o modelo não tem a classe '{CLASSE_VIOLENCIA}': {nomes}")
        self.idx_violencia = idx[0]

    def __call__(self, frames: Sequence[np.ndarray]) -> list[float]:
        if not frames:
            return []
        res = self.modelo.predict(list(frames), imgsz=self.cfg.imgsz, device=self.cfg.dispositivo,
                                  verbose=False)
        return [float(r.probs.data[self.idx_violencia]) for r in res]


FuncaoProb = Callable[[Sequence[np.ndarray]], list[float]]


def probabilidades_video(caminho: str, classificar: FuncaoProb, cfg: ConfigInferencia
                         ) -> tuple[list[float], list[float]]:
    tempos: list[float] = []
    probs: list[float] = []
    lote_t: list[float] = []
    lote_f: list[np.ndarray] = []
    for t, frame in amostrar_frames(caminho, cfg.fps_amostragem):
        lote_t.append(t)
        lote_f.append(frame)
        if len(lote_f) >= cfg.tamanho_lote:
            probs += classificar(lote_f)
            tempos += lote_t
            lote_t, lote_f = [], []
    if lote_f:
        probs += classificar(lote_f)
        tempos += lote_t
    return tempos, probs


def analisar_video(caminho: str, classificar: FuncaoProb, cfg: ConfigInferencia) -> dict:
    """Roda o pipeline inteiro e devolve os campos do contrato-ia."""
    inicio = time.time()
    tempos, probs = probabilidades_video(caminho, classificar, cfg)
    if not probs:
        raise ValueError("nenhum frame lido do vídeo")
    r: ResultadoVideo = agregar(tempos, probs, cfg, duracao_video(caminho))
    return {
        "score": r.score,
        "veredicto": r.veredicto,
        "segmentos_criticos": r.segmentos_criticos,
        "versao_modelo": cfg.versao_modelo,
        "tempo_processamento_ms": int((time.time() - inicio) * 1000),
        # extra (não vai pro backend): útil pra gráfico/depuração
        "_linha_do_tempo": {"tempos_s": [round(t, 3) for t in tempos], "prob": r.probs_suavizadas},
    }
