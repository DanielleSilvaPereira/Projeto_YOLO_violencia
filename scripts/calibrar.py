"""Calibra a regra que transforma P(violência) por frame no veredicto do vídeo.

O modelo é bom em ORDENAR frames (ROC-AUC alto), mas com os limiares padrão
quase todo vídeo normal longo tem 2 s de "falso positivo" e vira alerta.
Este script escolhe, nos vídeos de VALIDAÇÃO (nunca no teste), os parâmetros:

  JANELA_SUAVIZACAO_S  quanto suavizar a curva no tempo
  TOPO_S               quantos segundos "mais violentos" compõem o score
  LIMIAR_ANOMALIA      corte que maximiza a acurácia balanceada (Youden)
  LIMIAR_ALERTA_ALTO   corte com no máx. 5% de falso alarme nos normais
  LIMIAR_FRAME         = LIMIAR_ANOMALIA / 100 (para os trechos críticos)

e grava em pesos/calibracao.json, que o serviço e o avaliar.py leem sozinhos.

    python scripts/calibrar.py --pesos pesos/melhor.pt --dados data/frames

A inferência (parte lenta) roda uma vez e fica em cache; rodar de novo só
refaz a busca, em segundos.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ia.agregacao import agregar  # noqa: E402
from ia.config import CLASSE_VIOLENCIA, RAIZ, ConfigInferencia  # noqa: E402

JANELAS = (1.0, 2.0, 3.0, 5.0)
TOPOS = (2.0, 4.0, 8.0, 15.0)
FP_MAX_ALERTA = 0.05


def videos_do_split(manifesto: Path, split: str) -> dict[str, bool]:
    vids: dict[str, bool] = {}
    with open(manifesto, encoding="utf-8") as f:
        for l in csv.DictReader(f):
            if l["split"] == split:
                vids[l["caminho_video"]] = vids.get(l["caminho_video"], False) or l["classe"] == CLASSE_VIOLENCIA
    return vids


def linhas_do_tempo(vids: dict[str, bool], cfg: ConfigInferencia, cache: Path) -> dict:
    dados = json.loads(cache.read_text()) if cache.exists() else {}
    faltam = [v for v in vids if v not in dados]
    if faltam:
        from ia.video import ClassificadorFrames, duracao_video, probabilidades_video
        classificar = ClassificadorFrames(cfg)
        for n, v in enumerate(faltam, 1):
            t, p = probabilidades_video(v, classificar, cfg)
            dados[v] = {"violento": vids[v], "tempos": t, "probs": p, "duracao": duracao_video(v)}
            print(f"  {n}/{len(faltam)} {Path(v).name[:60]}")
            if n % 10 == 0 or n == len(faltam):
                cache.write_text(json.dumps(dados))
    return {v: dados[v] for v in vids}


def escolher_limiares(y: np.ndarray, s: np.ndarray) -> dict:
    """Youden para a anomalia; FP <= 5% para o alerta alto."""
    cortes = np.unique(np.concatenate([s, [0.0, 100.01]]))
    neg, pos = (y == 0).sum(), (y == 1).sum()
    melhor = None
    for c in cortes:
        pred = s >= c
        tpr = (pred & (y == 1)).sum() / max(pos, 1)
        fpr = (pred & (y == 0)).sum() / max(neg, 1)
        bal = (tpr + 1 - fpr) / 2
        if melhor is None or bal > melhor["acuracia_balanceada"]:
            melhor = {"limiar": float(c), "acuracia_balanceada": float(bal), "revocacao": float(tpr),
                      "falso_alarme": float(fpr)}
    alerta = None
    for c in cortes:  # crescente: primeiro corte com FP <= 5%
        if c < melhor["limiar"]:
            continue
        fpr = ((s >= c) & (y == 0)).sum() / max(neg, 1)
        if fpr <= FP_MAX_ALERTA:
            alerta = float(c)
            break
    melhor["limiar_alerta"] = alerta if alerta is not None else melhor["limiar"]
    return melhor


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    maior = (pos[:, None] > neg[None, :]).mean()
    igual = (pos[:, None] == neg[None, :]).mean()
    return float(maior + igual / 2)


def buscar(linhas: dict, cfg_base: ConfigInferencia) -> tuple[dict, list[dict]]:
    y = np.array([int(d["violento"]) for d in linhas.values()])
    tabela = []
    for janela, topo in itertools.product(JANELAS, TOPOS):
        cfg = replace(cfg_base, janela_suavizacao_s=janela, topo_s=topo)
        s = np.array([agregar(d["tempos"], d["probs"], cfg).score for d in linhas.values()])
        r = escolher_limiares(y, s)
        tabela.append({"janela_s": janela, "topo_s": topo, "auc": round(auc(y, s), 4), **r})
    tabela.sort(key=lambda r: (r["acuracia_balanceada"], r["auc"]), reverse=True)
    b = tabela[0]
    params = {
        "JANELA_SUAVIZACAO_S": b["janela_s"],
        "TOPO_S": b["topo_s"],
        "LIMIAR_ANOMALIA": round(b["limiar"], 2),
        "LIMIAR_ALERTA_ALTO": round(max(b["limiar_alerta"], b["limiar"]), 2),
        "LIMIAR_FRAME": round(b["limiar"] / 100, 4),
    }
    return params, tabela


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", default="pesos/melhor.pt")
    ap.add_argument("--dados", default="data/frames")
    ap.add_argument("--split", default="val", help="NÃO use test: o teste é só para a avaliação final")
    ap.add_argument("--saida", default=str(RAIZ / "pesos" / "calibracao.json"))
    args = ap.parse_args(argv)
    if args.split == "test":
        print("AVISO: calibrar no teste infla as métricas finais; use val.")

    cfg = ConfigInferencia(pesos=args.pesos)
    dados = Path(args.dados)
    vids = videos_do_split(dados / "manifesto.csv", args.split)
    print(f"{len(vids)} vídeos de {args.split} ({sum(vids.values())} violentos)")
    linhas = linhas_do_tempo(vids, cfg, dados / f"linhas_do_tempo_{args.split}.json")

    params, tabela = buscar(linhas, cfg)
    print("\n janela  topo   AUC    acur.bal  revoc.  falso_alarme  limiar")
    for r in tabela[:8]:
        print(f"  {r['janela_s']:4.0f}  {r['topo_s']:5.0f}  {r['auc']:.3f}   {r['acuracia_balanceada']:.3f}"
              f"    {r['revocacao']:.3f}   {r['falso_alarme']:.3f}        {r['limiar']:.1f}")
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps({"split": args.split, "n_videos": len(vids),
                                            "parametros": params, "melhor": tabela[0]},
                                           indent=2, ensure_ascii=False))
    print(f"\nparâmetros escolhidos: {json.dumps(params)}\nsalvo em {args.saida}")


if __name__ == "__main__":
    main()
