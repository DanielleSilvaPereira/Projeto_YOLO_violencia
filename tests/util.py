"""Vídeos sintéticos para testes: frames 'violentos' são vermelhos, normais verdes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def criar_video(caminho: Path, n_frames: int, fps: float, violentos: list[tuple[int, int]] = (),
                tamanho=(96, 64)) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    w, h = tamanho
    out = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        viol = any(a <= i <= b for a, b in violentos)
        cor = (0, 0, 255) if viol else (0, 255, 0)  # BGR
        out.write(np.full((h, w, 3), cor, dtype=np.uint8))
    out.release()
    return caminho


def classificador_falso(frames):
    """P(violência) = quanto o frame é mais vermelho que verde."""
    saida = []
    for f in frames:
        b, g, r = (float(f[..., i].mean()) for i in range(3))
        saida.append(r / (r + g + 1e-6))
    return saida
