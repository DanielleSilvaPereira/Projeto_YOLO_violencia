"""Parâmetros de inferência. Todos podem ser trocados por variável de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

CLASSE_VIOLENCIA = "violencia"
CLASSE_NORMAL = "nao_violencia"


def _f(nome: str, padrao: float) -> float:
    return float(os.environ.get(nome, padrao))


@dataclass
class ConfigInferencia:
    # Caminho dos pesos treinados (saída de scripts/treinar.py).
    pesos: str = field(default_factory=lambda: os.environ.get("MODELO_PESOS", "pesos/melhor.pt"))
    # Nome/versão gravado no laudo (RN07 - rastreabilidade).
    versao_modelo: str = field(default_factory=lambda: os.environ.get("VERSAO_MODELO", "yolov9c-cls-xdviolence-v1"))
    # Quantos frames por segundo do vídeo são analisados.
    fps_amostragem: float = field(default_factory=lambda: _f("FPS_AMOSTRAGEM", 4))
    # Tamanho da imagem de entrada do modelo (igual ao usado no treino).
    imgsz: int = field(default_factory=lambda: int(_f("IMGSZ", 224)))
    # Janela (s) da média móvel que suaviza a probabilidade frame a frame.
    janela_suavizacao_s: float = field(default_factory=lambda: _f("JANELA_SUAVIZACAO_S", 1.0))
    # Probabilidade (0-1) a partir da qual um frame conta como violento.
    limiar_frame: float = field(default_factory=lambda: _f("LIMIAR_FRAME", 0.5))
    # Trechos mais curtos que isto são descartados; buracos menores que isto são unidos.
    duracao_min_segmento_s: float = field(default_factory=lambda: _f("DURACAO_MIN_SEGMENTO_S", 1.0))
    uniao_max_s: float = field(default_factory=lambda: _f("UNIAO_MAX_S", 1.0))
    # Score do vídeo = média das maiores probabilidades suavizadas que somam
    # esta duração (s). Fixo em segundos: 4 s de briga pesam igual num vídeo
    # de 10 s ou de 10 min.
    topo_s: float = field(default_factory=lambda: _f("TOPO_S", 2.0))
    # Faixas do veredicto (score 0-100), conforme dicionário de dados.
    limiar_alerta_alto: float = field(default_factory=lambda: _f("LIMIAR_ALERTA_ALTO", 70))
    limiar_anomalia: float = field(default_factory=lambda: _f("LIMIAR_ANOMALIA", 40))
    tamanho_lote: int = field(default_factory=lambda: int(_f("TAMANHO_LOTE", 32)))
    dispositivo: str | None = field(default_factory=lambda: os.environ.get("DISPOSITIVO") or None)
