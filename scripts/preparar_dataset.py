"""Extrai frames rotulados dos vídeos e monta o dataset de classificação.

Saída (formato que o YOLO de classificação lê direto):

    data/frames/
      train/violencia/*.jpg   train/nao_violencia/*.jpg
      val/...                 test/...
      manifesto.csv           (de onde veio cada imagem)
      resumo.json             (contagens por split/classe)

A divisão train/val/test é feita POR VÍDEO: frames do mesmo vídeo nunca
aparecem em splits diferentes (senão a acurácia fica artificialmente alta).

Exemplos
--------
XD-Violence (padrão: só categorias de agressão, anotação por frame do teste):

    python scripts/preparar_dataset.py --formato xd \
        --videos data/xd_violence --anotacoes data/xd_violence/test_annotations.txt

Incluindo também os vídeos violentos de treino (rótulo só por vídeo, mais ruído):

    python scripts/preparar_dataset.py --formato xd --videos ... --anotacoes ... --usar-fracos

Qualquer dataset organizado em pastas (ex.: Real Life Violence Situations):

    python scripts/preparar_dataset.py --formato pastas --videos data/rlvs
    # data/rlvs/Violence/*.mp4 e data/rlvs/NonViolence/*.mp4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ia.config import CLASSE_NORMAL, CLASSE_VIOLENCIA  # noqa: E402
from ia import xd_violence as xd  # noqa: E402

EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass
class Tarefa:
    caminho: str
    id: str
    split: str
    modo: str  # "normal" | "violento" | "anotado"
    categorias: str
    intervalos: list | None = None


def split_do_video(vid: str, proporcoes: tuple[float, float, float], seed: int) -> str:
    h = int(hashlib.md5(f"{seed}:{vid}".encode()).hexdigest(), 16) / 16**32
    if h < proporcoes[0]:
        return "train"
    if h < proporcoes[0] + proporcoes[1]:
        return "val"
    return "test"


def listar_videos(raiz: Path) -> list[Path]:
    return sorted(p for p in raiz.rglob("*") if p.suffix.lower() in EXTS)


def tarefas_xd(args) -> tuple[list[Tarefa], Counter]:
    anot = xd.ler_anotacoes(args.anotacoes) if args.anotacoes else {}
    cats_ok = set(args.categorias.split(","))
    ignorados: Counter = Counter()
    tarefas = []
    for p in listar_videos(Path(args.videos)):
        try:
            cats = xd.categorias_do_nome(p)
        except ValueError:
            ignorados["sem_rotulo_no_nome"] += 1
            continue
        vid = xd.id_video(p)
        split = split_do_video(vid, args.proporcoes, args.seed)
        if xd.eh_normal(cats):
            tarefas.append(Tarefa(str(p), vid, split, "normal", "A"))
            continue
        if not cats_ok.intersection(cats):
            ignorados["categoria_fora_do_filtro"] += 1
            continue
        if vid in anot:
            tarefas.append(Tarefa(str(p), vid, split, "anotado", "-".join(cats), anot[vid]))
        elif args.usar_fracos:
            # rótulo só por vídeo: vai sempre pro treino, nunca pra avaliação
            tarefas.append(Tarefa(str(p), vid, "train", "violento", "-".join(cats)))
        else:
            ignorados["violento_sem_anotacao"] += 1
    return tarefas, ignorados


def classe_da_pasta(nome: str) -> str | None:
    n = nome.lower().replace("-", "").replace("_", "").replace(" ", "")
    if n.startswith(("non", "nao", "normal", "no")):
        return CLASSE_NORMAL
    if "viol" in n or "fight" in n or "briga" in n or "agress" in n:
        return CLASSE_VIOLENCIA
    return None


def tarefas_pastas(args) -> tuple[list[Tarefa], Counter]:
    ignorados: Counter = Counter()
    tarefas = []
    raiz = Path(args.videos)
    for p in listar_videos(raiz):
        classe = None
        for parte in p.relative_to(raiz).parts[:-1]:
            classe = classe_da_pasta(parte) or classe
        if classe is None:
            ignorados["pasta_sem_classe"] += 1
            continue
        vid = f"{p.parent.name}/{p.stem}"
        modo = "violento" if classe == CLASSE_VIOLENCIA else "normal"
        tarefas.append(Tarefa(str(p), vid, split_do_video(vid, args.proporcoes, args.seed), modo, classe))
    return tarefas, ignorados


def extrair(t: Tarefa, saida: str, fps_alvo: float, max_frames: int, lado: int, margem_s: float
            ) -> list[dict]:
    cap = cv2.VideoCapture(t.caminho)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps != fps or fps > 240:
        fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    passo = max(1, int(round(fps / fps_alvo)))
    if total > 0 and max_frames > 0:
        passo = max(passo, -(-total // max_frames))  # divisão com teto
    margem = int(round(margem_s * fps))
    nome_seguro = hashlib.md5(t.id.encode()).hexdigest()[:12]

    linhas = []
    idx = 0
    while True:
        if not cap.grab():
            break
        i = idx
        idx += 1
        if i % passo:
            continue
        if t.modo == "anotado":
            viol = xd.frame_violento(i, t.intervalos)
            # perto da borda de um trecho o rótulo é duvidoso: pula
            if margem and (viol != xd.frame_violento(i - margem, t.intervalos)
                           or viol != xd.frame_violento(i + margem, t.intervalos)):
                continue
            classe = CLASSE_VIOLENCIA if viol else CLASSE_NORMAL
        else:
            classe = CLASSE_VIOLENCIA if t.modo == "violento" else CLASSE_NORMAL
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        esc = lado / min(h, w)
        if esc < 1:
            frame = cv2.resize(frame, (round(w * esc), round(h * esc)), interpolation=cv2.INTER_AREA)
        destino = Path(saida) / t.split / classe / f"{nome_seguro}_{i:06d}.jpg"
        cv2.imwrite(str(destino), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        linhas.append({"split": t.split, "classe": classe, "arquivo": str(destino.relative_to(saida)),
                       "video": t.id, "frame": i, "tempo_s": round(i / fps, 3),
                       "categorias": t.categorias, "modo": t.modo, "caminho_video": t.caminho})
        if max_frames and len(linhas) >= max_frames:
            break
    cap.release()
    return linhas


def balancear_treino(saida: Path, linhas: list[dict], seed: int) -> list[dict]:
    treino = [l for l in linhas if l["split"] == "train"]
    por_classe = {c: [l for l in treino if l["classe"] == c] for c in (CLASSE_VIOLENCIA, CLASSE_NORMAL)}
    menor = min(len(v) for v in por_classe.values())
    rng = random.Random(seed)
    remover = []
    for v in por_classe.values():
        if len(v) > menor:
            remover += rng.sample(v, len(v) - menor)
    ids = {id(l) for l in remover}
    for l in remover:
        (saida / l["arquivo"]).unlink(missing_ok=True)
    return [l for l in linhas if id(l) not in ids]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formato", choices=["xd", "pastas"], required=True)
    ap.add_argument("--videos", required=True, help="pasta com os vídeos (busca recursiva)")
    ap.add_argument("--anotacoes", help="XD: test_annotations.txt (rótulo por frame)")
    ap.add_argument("--saida", default="data/frames")
    ap.add_argument("--categorias", default=",".join(xd.CATEGORIAS_AGRESSAO),
                    help="XD: categorias de violência usadas (padrão: B1,B2,B4,B5)")
    ap.add_argument("--usar-fracos", action="store_true",
                    help="XD: inclui vídeos violentos sem anotação por frame (só no treino)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames extraídos por segundo de vídeo")
    ap.add_argument("--max-frames-por-video", type=int, default=64)
    ap.add_argument("--lado", type=int, default=320, help="redimensiona o menor lado para este valor")
    ap.add_argument("--margem-s", type=float, default=0.5,
                    help="XD anotado: ignora frames a menos disto (s) da borda de um trecho violento")
    ap.add_argument("--proporcoes", default="0.7,0.15,0.15", help="train,val,test (por vídeo)")
    ap.add_argument("--balancear", action="store_true", help="iguala as classes no treino")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.proporcoes = tuple(float(x) for x in args.proporcoes.split(","))
    if abs(sum(args.proporcoes) - 1) > 1e-6:
        ap.error("--proporcoes precisa somar 1")

    tarefas, ignorados = tarefas_xd(args) if args.formato == "xd" else tarefas_pastas(args)
    if not tarefas:
        sys.exit(f"nenhum vídeo utilizável em {args.videos} (ignorados: {dict(ignorados)})")

    saida = Path(args.saida)
    for s in ("train", "val", "test"):
        for c in (CLASSE_VIOLENCIA, CLASSE_NORMAL):
            (saida / s / c).mkdir(parents=True, exist_ok=True)

    print(f"{len(tarefas)} vídeos | modos: {dict(Counter(t.modo for t in tarefas))} | ignorados: {dict(ignorados)}")
    linhas: list[dict] = []
    params = (str(saida), args.fps, args.max_frames_por_video, args.lado, args.margem_s)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futuros = [ex.submit(extrair, t, *params) for t in tarefas]
        for n, f in enumerate(as_completed(futuros), 1):
            linhas += f.result()
            if n % 50 == 0 or n == len(futuros):
                print(f"  {n}/{len(futuros)} vídeos, {len(linhas)} frames")

    if args.balancear:
        linhas = balancear_treino(saida, linhas, args.seed)

    linhas.sort(key=lambda l: l["arquivo"])
    with open(saida / "manifesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()) if linhas else ["arquivo"])
        w.writeheader()
        w.writerows(linhas)

    resumo = {
        "formato": args.formato,
        "parametros": {k: v for k, v in vars(args).items()},
        "videos_por_split": dict(Counter((t.split) for t in tarefas)),
        "frames": {s: dict(Counter(l["classe"] for l in linhas if l["split"] == s))
                   for s in ("train", "val", "test")},
        "ignorados": dict(ignorados),
    }
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo["frames"], indent=2))
    for s in ("val", "test"):
        if len(resumo["frames"].get(s, {})) < 2:
            print(f"AVISO: o split '{s}' não tem as duas classes; use mais vídeos ou mude --proporcoes.")


if __name__ == "__main__":
    main()
