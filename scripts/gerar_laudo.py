"""Gera o laudo em PDF a partir de um resultado já salvo (JSON do callback).

    python scripts/gerar_laudo.py resultado.json --saida laudo.pdf
    python scripts/gerar_laudo.py resultado.json --metadados gravacao.json --video video.mp4

resultado.json: o JSON que a IA mandou no callback (docs/contrato-ia.md).
gravacao.json (opcional): {"id_gravacao": 12, "arquivo": "...", "data_gravacao": "...",
                           "dispositivo": "...", "duracao_s": 16.4, "hash_sha256": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ia.laudo import gerar_laudo  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resultado")
    ap.add_argument("--metadados")
    ap.add_argument("--video", help="vídeo original, para incluir os quadros de pico")
    ap.add_argument("--saida", default="laudo.pdf")
    args = ap.parse_args(argv)
    resultado = json.loads(Path(args.resultado).read_text(encoding="utf-8"))
    meta = json.loads(Path(args.metadados).read_text(encoding="utf-8")) if args.metadados else {}
    codigo = gerar_laudo(resultado, args.saida, meta, video=args.video)
    print(f"laudo: {args.saida}\ncódigo de verificação: {codigo}")


if __name__ == "__main__":
    main()
