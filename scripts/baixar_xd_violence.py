"""Baixa o XD-Violence (vídeos + anotação) do espelho no Hugging Face.

O dataset completo tem ~4.750 vídeos / 217 h (dezenas de GB). Para o TCC
normalmente basta um subconjunto; por isso dá pra limitar quantos vídeos
baixar de cada parte.

    # todos os vídeos de teste (têm anotação por frame) + 300 normais do treino
    python scripts/baixar_xd_violence.py --partes test_videos --normais-treino 300

    # tudo (demora e ocupa muito espaço)
    python scripts/baixar_xd_violence.py --partes todas

Espelho: https://huggingface.co/datasets/jherng/xd-violence
Oficial:  https://roc-ng.github.io/XD-Violence/  (OneDrive/Baidu, mesmos arquivos)
Uso: apenas pesquisa/acadêmico — cite o artigo (Wu et al., ECCV 2020).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO = "jherng/xd-violence"
PARTES_TREINO = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954"]
PARTE_TESTE = "test_videos"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--destino", default="data/xd_violence")
    ap.add_argument("--partes", default=PARTE_TESTE,
                    help=f"lista separada por vírgula entre {PARTES_TREINO + [PARTE_TESTE]} ou 'todas'")
    ap.add_argument("--max-por-parte", type=int, default=0, help="0 = todos os vídeos da parte")
    ap.add_argument("--normais-treino", type=int, default=0,
                    help="baixa N vídeos NORMAIS (label_A) sorteados das partes de treino")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("instale: pip install huggingface_hub")

    api = HfApi()
    arquivos = api.list_repo_files(REPO, repo_type="dataset")
    videos = [f for f in arquivos if f.startswith("data/video/") and f.lower().endswith(".mp4")]
    anotacoes = [f for f in arquivos if "annotation" in f.lower() and f.endswith(".txt")]
    rng = random.Random(args.seed)

    partes = PARTES_TREINO + [PARTE_TESTE] if args.partes == "todas" else args.partes.split(",")
    escolhidos: list[str] = []
    for parte in partes:
        da_parte = sorted(v for v in videos if v.startswith(f"data/video/{parte}/"))
        if not da_parte:
            print(f"AVISO: parte '{parte}' não encontrada no repositório")
        if args.max_por_parte and len(da_parte) > args.max_por_parte:
            da_parte = rng.sample(da_parte, args.max_por_parte)
        escolhidos += da_parte

    if args.normais_treino:
        normais = sorted(v for v in videos if any(v.startswith(f"data/video/{p}/") for p in PARTES_TREINO)
                         and "label_A" in v and v not in escolhidos)
        escolhidos += rng.sample(normais, min(args.normais_treino, len(normais)))

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)

    # local_dir grava direto no destino (sem cópia duplicada no cache)
    for f in anotacoes:
        local = hf_hub_download(REPO, f, repo_type="dataset", local_dir=destino)
        if "test" in f.lower():
            shutil.copy(local, destino / "test_annotations.txt")
        print(f"anotação: {local}")

    print(f"baixando {len(escolhidos)} vídeos para {destino} ...")
    for n, f in enumerate(escolhidos, 1):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=destino)
        if n % 25 == 0 or n == len(escolhidos):
            print(f"  {n}/{len(escolhidos)}")
    print(f"pronto. Anotação por frame: {destino / 'test_annotations.txt'}")


if __name__ == "__main__":
    main()
