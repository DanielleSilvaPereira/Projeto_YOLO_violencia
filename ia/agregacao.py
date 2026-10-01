"""Transforma a probabilidade de cada frame no resultado do vídeo inteiro.

Não depende de torch/YOLO, então dá pra testar sem GPU.

Entrada: lista de (tempo_s, prob_violencia) dos frames amostrados.
Saída:   score 0-100, veredicto e trechos críticos (formato do contrato-ia).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConfigInferencia


@dataclass
class ResultadoVideo:
    score: float
    veredicto: str
    segmentos_criticos: list[dict]
    probs_suavizadas: list[float]
    linha_do_tempo: list[dict] | None = None


def suavizar(probs: np.ndarray, janela: int) -> np.ndarray:
    """Média móvel centrada; nas bordas usa só os vizinhos existentes."""
    if janela <= 1 or len(probs) == 0:
        return probs.astype(float)
    kernel = np.ones(janela)
    soma = np.convolve(probs, kernel, mode="same")
    cont = np.convolve(np.ones_like(probs, dtype=float), kernel, mode="same")
    return soma / cont


def score_video(probs_suav: np.ndarray, k: int) -> float:
    """Média das k maiores probabilidades.

    Um vídeo com 3 s de briga em 60 s precisa ter score alto; a média simples
    diluiria o trecho. Usar só o máximo seria sensível a um frame isolado.
    """
    if len(probs_suav) == 0:
        return 0.0
    k = max(1, min(k, len(probs_suav)))
    topo = np.sort(probs_suav)[-k:]
    return float(topo.mean() * 100)


def veredicto(score: float, cfg: ConfigInferencia) -> str:
    if score >= cfg.limiar_alerta_alto:
        return "Alerta Alto"
    if score >= cfg.limiar_anomalia:
        return "Anomalia"
    return "Normal"


def segmentos(tempos: np.ndarray, probs: np.ndarray, passo_s: float, cfg: ConfigInferencia,
              duracao_video_s: float | None = None) -> list[dict]:
    """Agrupa frames acima do limiar em trechos [inicio_s, fim_s]."""
    acima = probs >= cfg.limiar_frame
    brutos: list[list[float]] = []  # [inicio, fim, indice_ini, indice_fim]
    i = 0
    while i < len(acima):
        if not acima[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(acima) and acima[j + 1]:
            j += 1
        # cada amostra representa passo_s segundos a partir do seu tempo
        brutos.append([float(tempos[i]), float(tempos[j]) + passo_s, i, j])
        i = j + 1

    # une trechos separados por buracos curtos
    unidos: list[list[float]] = []
    for s in brutos:
        if unidos and s[0] - unidos[-1][1] <= cfg.uniao_max_s:
            unidos[-1][1] = s[1]
            unidos[-1][3] = s[3]
        else:
            unidos.append(s)

    saida = []
    for ini, fim, a, b in unidos:
        if duracao_video_s is not None:
            fim = min(fim, duracao_video_s)
        if fim - ini < cfg.duracao_min_segmento_s:
            continue
        trecho = probs[int(a): int(b) + 1]
        pico = int(a) + int(np.argmax(trecho))
        saida.append({
            "inicio_s": round(ini, 2),
            "fim_s": round(fim, 2),
            "score": round(float(trecho.max()) * 100, 2),
            "pico_s": round(float(tempos[pico]), 2),  # momento mais violento do trecho
        })
    return saida


def nivel(score: float, cfg: ConfigInferencia) -> str:
    """Cor da barra no front: "critico" (vermelho), "alerta" (amarelo) ou "normal"."""
    if score >= cfg.limiar_alerta_alto:
        return "critico"
    if score >= cfg.limiar_anomalia:
        return "alerta"
    return "normal"


def linha_do_tempo(tempos: np.ndarray, probs_suav: np.ndarray, probs_brutas: np.ndarray,
                   cfg: ConfigInferencia, duracao_video_s: float | None = None) -> list[dict]:
    """Resume a curva em intervalos fixos (padrão 1 s) para o gráfico do front.

    score       = maior probabilidade suavizada do intervalo (0-100), a mesma
                  curva usada no veredicto e nos trechos críticos
    score_frame = maior probabilidade de um frame isolado, sem suavizar
    pico_s      = tempo exato desse frame (o front pode pular o vídeo pra ele)
    """
    passo = max(cfg.passo_linha_do_tempo_s, 1e-3)
    fim_total = duracao_video_s or (float(tempos[-1]) + 1.0 / cfg.fps_amostragem)
    n = max(1, int(np.ceil(fim_total / passo - 1e-9)))
    idx = np.minimum((tempos / passo).astype(int), n - 1)
    saida = []
    for b in range(n):
        sel = np.nonzero(idx == b)[0]
        if len(sel) == 0:
            continue
        i_suav = sel[int(np.argmax(probs_suav[sel]))]
        i_bruto = sel[int(np.argmax(probs_brutas[sel]))]
        sc = round(float(probs_suav[i_suav]) * 100, 2)
        saida.append({
            "inicio_s": round(b * passo, 2),
            "fim_s": round(min((b + 1) * passo, fim_total), 2),
            "score": sc,
            "score_frame": round(float(probs_brutas[i_bruto]) * 100, 2),
            "pico_s": round(float(tempos[i_bruto]), 2),
            "nivel": nivel(sc, cfg),
        })
    return saida


def agregar(tempos: list[float], probs: list[float], cfg: ConfigInferencia,
            duracao_video_s: float | None = None) -> ResultadoVideo:
    t = np.asarray(tempos, dtype=float)
    p = np.asarray(probs, dtype=float)
    if len(p) == 0:
        return ResultadoVideo(0.0, "Normal", [], [])

    passo_s = 1.0 / cfg.fps_amostragem
    janela = max(1, int(round(cfg.janela_suavizacao_s / passo_s)))
    ps = suavizar(p, janela)
    k = max(1, int(round(cfg.topo_s / passo_s)))
    sc = round(score_video(ps, k), 2)
    return ResultadoVideo(
        score=sc,
        veredicto=veredicto(sc, cfg),
        segmentos_criticos=segmentos(t, ps, passo_s, cfg, duracao_video_s),
        probs_suavizadas=[round(float(x), 4) for x in ps],
        linha_do_tempo=linha_do_tempo(t, ps, p, cfg, duracao_video_s),
    )
