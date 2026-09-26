"""Leitura dos rótulos do XD-Violence.

Nome dos vídeos:  <titulo>_label_<C1>-<C2>-<C3>.mp4
  A  = normal         B1 = luta (fighting)   B2 = tiro (shooting)
  B4 = tumulto (riot) B5 = abuso (abuse)     B6 = acidente de carro
  G  = explosão       "0" = posição vazia (vídeos têm até 3 categorias)

Anotação do teste (test_annotations.txt), uma linha por vídeo violento:
  <nome_do_video_sem_.mp4> ini1 fim1 ini2 fim2 ...   (índices de frame)
"""

from __future__ import annotations

import re
from pathlib import Path

CATEGORIAS = {
    "A": "normal",
    "B1": "luta",
    "B2": "tiro",
    "B4": "tumulto",
    "B5": "abuso",
    "B6": "acidente_carro",
    "G": "explosao",
}
# Padrão do Safe Evidence: só o que é agressão entre pessoas.
CATEGORIAS_AGRESSAO = ("B1", "B2", "B4", "B5")

_RE_LABEL = re.compile(r"label_([A-Z0-9]+(?:-[A-Z0-9]+)*)")


def id_video(caminho: str | Path) -> str:
    nome = Path(caminho).name
    for ext in (".mp4", ".avi", ".mkv", ".webm"):
        if nome.lower().endswith(ext):
            return nome[: -len(ext)]
    return nome


def categorias_do_nome(caminho: str | Path) -> list[str]:
    """'filme__#00-01_label_B1-B2-0.mp4' -> ['B1', 'B2']."""
    m = _RE_LABEL.search(Path(caminho).name)
    if not m:
        raise ValueError(f"nome sem 'label_': {caminho}")
    return [c for c in m.group(1).split("-") if c != "0"]


def eh_normal(cats: list[str]) -> bool:
    return cats == ["A"]


def ler_anotacoes(caminho: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Devolve {id_video: [(frame_ini, frame_fim), ...]} (intervalos inclusivos)."""
    anot: dict[str, list[tuple[int, int]]] = {}
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if not partes:
            continue
        vid = id_video(partes[0])
        nums = [int(float(x)) for x in partes[1:]]
        if len(nums) % 2:
            raise ValueError(f"número ímpar de índices na anotação de {vid}")
        anot[vid] = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    return anot


def frame_violento(idx_frame: int, intervalos: list[tuple[int, int]]) -> bool:
    return any(a <= idx_frame <= b for a, b in intervalos)
