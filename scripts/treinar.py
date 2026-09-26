"""Treina o classificador de frames (YOLOv9-C cls) no dataset gerado por preparar_dataset.py.

    python scripts/treinar.py --dados data/frames --epocas 30 --lote 64

No fim copia o melhor peso para pesos/melhor.pt (usado pelo serviço).

--modelo aceita também um classificador pronto do Ultralytics (ex.:
yolo11n-cls.pt) — útil como baseline para comparar com o YOLOv9 no TCC.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default="data/frames", help="pasta com train/ val/ (test/)")
    ap.add_argument("--modelo", default=str(RAIZ / "configs" / "yolov9c-cls.yaml"),
                    help="yaml da arquitetura ou .pt de um classificador")
    ap.add_argument("--pretreinado", default="yolov9c.pt",
                    help="pesos de onde copiar o backbone (só quando --modelo é .yaml); 'nenhum' treina do zero")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--paciencia", type=int, default=8, help="early stopping (épocas sem melhora)")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--dispositivo", default=None, help="0, 0,1, cpu ... (padrão: automático)")
    ap.add_argument("--projeto", default="runs/violencia")
    ap.add_argument("--nome", default="yolov9c-cls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    if args.modelo.endswith((".yaml", ".yml")):
        modelo = YOLO(args.modelo, task="classify")
        if args.pretreinado.lower() != "nenhum":
            # copia todas as camadas com mesmo nome e formato (backbone 0-9)
            modelo.load(args.pretreinado)
    else:
        modelo = YOLO(args.modelo)

    modelo.train(
        data=str(Path(args.dados).resolve()),
        epochs=args.epocas,
        batch=args.lote,
        imgsz=args.imgsz,
        patience=args.paciencia,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        device=args.dispositivo,
        project=args.projeto,
        name=args.nome,
        workers=args.workers,
        seed=args.seed,
        # aumentos de dados: câmera no pulso/parede -> enquadramento e luz variam
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        erasing=0.3,
        plots=True,
    )

    pasta = Path(modelo.trainer.save_dir)
    melhor = pasta / "weights" / "best.pt"
    destino = RAIZ / "pesos" / "melhor.pt"
    destino.parent.mkdir(exist_ok=True)
    shutil.copy(melhor, destino)
    info = {"origem": str(melhor), "dados": args.dados, "modelo": args.modelo,
            "pretreinado": args.pretreinado, "epocas": args.epocas, "imgsz": args.imgsz,
            "classes": modelo.names}
    (destino.parent / "melhor.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\nmelhor peso: {melhor}\ncopiado para: {destino}")


if __name__ == "__main__":
    main()
