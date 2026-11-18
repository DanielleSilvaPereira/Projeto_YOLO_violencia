"""Analisa um vídeo e imprime o resultado no formato do contrato-ia.

    python scripts/analisar_video.py meu_video.mp4
    python scripts/analisar_video.py meu_video.mp4 --video-saida anotado.mp4 --grafico linha.png
    python scripts/analisar_video.py meu_video.mp4 --laudo laudo.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ia.config import ConfigInferencia  # noqa: E402
from ia.video import ClassificadorFrames, analisar_video  # noqa: E402


def gerar_video_anotado(entrada: str, saida: str, tempos: list[float], probs: list[float], limiar: float):
    cap = cv2.VideoCapture(entrada)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t_arr = np.asarray(tempos)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        p = float(probs[max(0, int(np.searchsorted(t_arr, t, side="right")) - 1)])
        viol = p >= limiar
        cor = (0, 0, 255) if viol else (0, 180, 0)
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(frame, f"{'VIOLENCIA' if viol else 'normal'}  {p * 100:5.1f}%  t={t:6.2f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.rectangle(frame, (0, h - 10), (int(w * p), h), cor, -1)
        out.write(frame)
    cap.release()
    out.release()


def gerar_grafico(tempos, probs, segmentos, limiar, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for s in segmentos:
        ax.axvspan(s["inicio_s"], s["fim_s"], color="tab:red", alpha=0.15)
    ax.plot(tempos, probs, color="tab:red", lw=1.5)
    ax.axhline(limiar, ls="--", color="gray", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("P(violência)")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--pesos", default=None, help="padrão: MODELO_PESOS ou pesos/melhor.pt")
    ap.add_argument("--video-saida", help="grava uma cópia com a probabilidade desenhada")
    ap.add_argument("--grafico", help="PNG com a probabilidade ao longo do tempo")
    ap.add_argument("--laudo", help="gera o laudo em PDF (com os quadros de pico)")
    args = ap.parse_args(argv)

    cfg = ConfigInferencia()
    if args.pesos:
        cfg.pesos = args.pesos
    r = analisar_video(args.video, ClassificadorFrames(cfg), cfg)
    linha = r.pop("_linha_do_tempo")
    h = hashlib.sha256(Path(args.video).read_bytes()).hexdigest()
    r = {"status": "concluido", **r, "hash_sha256_verificado": h}
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if args.video_saida:
        gerar_video_anotado(args.video, args.video_saida, linha["tempos_s"], linha["prob"], cfg.limiar_frame)
        print(f"vídeo anotado: {args.video_saida}")
    if args.grafico:
        gerar_grafico(linha["tempos_s"], linha["prob"], r["segmentos_criticos"], cfg.limiar_frame, args.grafico)
        print(f"gráfico: {args.grafico}")
    if args.laudo:
        from ia.laudo import gerar_laudo
        from ia.video import duracao_video
        meta = {"id_gravacao": "-", "arquivo": Path(args.video).name, "hash_sha256": h,
                "duracao_s": duracao_video(args.video),
                "data_gravacao": datetime.fromtimestamp(Path(args.video).stat().st_mtime).strftime("%d/%m/%Y %H:%M")}
        gerar_laudo(r, args.laudo, meta, video=args.video, cfg=cfg)
        print(f"laudo: {args.laudo}")


if __name__ == "__main__":
    main()
