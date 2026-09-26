"""Avalia o modelo treinado — métricas para o capítulo de resultados do TCC.

1) Por frame (imagens de data/frames/test):
   acurácia, precisão, revocação, F1, ROC-AUC, AP e matriz de confusão.

2) Por vídeo (--por-video): roda o pipeline completo em cada vídeo do split
   de teste (lista vem do manifesto.csv) e mede
   - se o veredicto acerta (violento = "Alerta Alto" ou "Anomalia");
   - o tempo de processamento / duração do vídeo (RN16: <= 50%).

    python scripts/avaliar.py --pesos pesos/melhor.pt --dados data/frames --por-video
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ia.config import CLASSE_NORMAL, CLASSE_VIOLENCIA, ConfigInferencia  # noqa: E402
from ia.video import ClassificadorFrames, analisar_video, duracao_video  # noqa: E402


def metricas(y: np.ndarray, p: np.ndarray, limiar: float) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    pred = (p >= limiar).astype(int)
    m = {
        "n": int(len(y)),
        "positivos": int(y.sum()),
        "limiar": limiar,
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "revocacao": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "matriz_confusao": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if 0 < y.sum() < len(y):
        m["roc_auc"] = roc_auc_score(y, p)
        m["ap"] = average_precision_score(y, p)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def salvar_matriz(mc: list, titulo: str, arquivo: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(mc, cmap="Blues")
    rot = ["não violência", "violência"]
    ax.set_xticks([0, 1], rot)
    ax.set_yticks([0, 1], rot)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    tot = max(1, max(max(l) for l in mc))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mc[i][j], ha="center", va="center",
                    color="white" if mc[i][j] > tot / 2 else "black")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def avaliar_frames(classificar, pasta_teste: Path, cfg: ConfigInferencia, lote: int) -> dict:
    arquivos, y = [], []
    for classe, rot in ((CLASSE_VIOLENCIA, 1), (CLASSE_NORMAL, 0)):
        fs = sorted((pasta_teste / classe).glob("*.jpg"))
        arquivos += fs
        y += [rot] * len(fs)
    if not arquivos:
        sys.exit(f"sem imagens em {pasta_teste}")
    probs: list[float] = []
    for i in range(0, len(arquivos), lote):
        probs += classificar([cv2.imread(str(f)) for f in arquivos[i:i + lote]])
    return metricas(np.array(y), np.array(probs), cfg.limiar_frame)


def avaliar_videos(classificar, manifesto: Path, cfg: ConfigInferencia) -> tuple[dict, list[dict]]:
    videos: dict[str, dict] = {}
    with open(manifesto, encoding="utf-8") as f:
        for l in csv.DictReader(f):
            if l["split"] != "test":
                continue
            v = videos.setdefault(l["caminho_video"], {"violento": False})
            v["violento"] |= l["classe"] == CLASSE_VIOLENCIA
    linhas = []
    for n, (caminho, info) in enumerate(sorted(videos.items()), 1):
        r = analisar_video(caminho, classificar, cfg)
        dur = duracao_video(caminho) or 0
        linhas.append({
            "video": Path(caminho).name,
            "real_violento": int(info["violento"]),
            "score": r["score"],
            "veredicto": r["veredicto"],
            "n_segmentos": len(r["segmentos_criticos"]),
            "duracao_s": round(dur, 2),
            "tempo_proc_s": round(r["tempo_processamento_ms"] / 1000, 2),
        })
        print(f"  {n}/{len(videos)} {linhas[-1]['video'][:60]} -> {r['veredicto']} ({r['score']})")
    y = np.array([l["real_violento"] for l in linhas])
    s = np.array([l["score"] / 100 for l in linhas])
    m = metricas(y, s, cfg.limiar_anomalia / 100)
    razoes = [l["tempo_proc_s"] / l["duracao_s"] for l in linhas if l["duracao_s"] > 0]
    m["rn16_tempo_sobre_duracao_medio"] = round(float(np.mean(razoes)), 4) if razoes else None
    m["rn16_pior_caso"] = round(float(np.max(razoes)), 4) if razoes else None
    m["rn16_ok"] = bool(razoes) and max(razoes) <= 0.5
    return m, linhas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", default="pesos/melhor.pt")
    ap.add_argument("--dados", default="data/frames")
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--por-video", action="store_true")
    ap.add_argument("--lote", type=int, default=64)
    args = ap.parse_args(argv)

    cfg = ConfigInferencia(pesos=args.pesos)
    classificar = ClassificadorFrames(cfg)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    dados = Path(args.dados)

    print("avaliando frames ...")
    mf = avaliar_frames(classificar, dados / "test", cfg, args.lote)
    salvar_matriz(mf["matriz_confusao"], "Por frame", saida / "matriz_frames.png")
    resultado = {"modelo": args.pesos, "versao": cfg.versao_modelo, "por_frame": mf}

    if args.por_video:
        print("avaliando vídeos ...")
        mv, linhas = avaliar_videos(classificar, dados / "manifesto.csv", cfg)
        salvar_matriz(mv["matriz_confusao"], "Por vídeo", saida / "matriz_videos.png")
        resultado["por_video"] = mv
        with open(saida / "videos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)

    (saida / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
