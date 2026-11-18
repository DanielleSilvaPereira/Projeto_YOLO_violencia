"""Gera o laudo em PDF de uma gravação analisada ("Download de Segurança").

Entrada: o resultado que a IA mandou no callback (docs/contrato-ia.md) e os
metadados da gravação que o backend guarda. Opcionalmente o próprio vídeo,
para incluir as imagens dos momentos de pico de cada trecho crítico.

    from ia.laudo import gerar_laudo
    gerar_laudo(resultado, "laudo_12.pdf", metadados={"id_gravacao": 12, ...})

O PDF traz um "código de verificação": SHA-256 do resultado da análise em
JSON canônico. Quem tiver o resultado guardado no banco consegue recalcular e
conferir que o laudo não foi editado (ver `codigo_verificacao`).
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from .config import ConfigInferencia

# cores iguais às da tela do front
VERMELHO = "#C62828"
AMARELO = "#F9A825"
CINZA = "#BDBDBD"
ROXO = "#3B2A8C"
CORES_NIVEL = {"critico": VERMELHO, "alerta": AMARELO, "normal": CINZA}
CORES_VEREDICTO = {"Alerta Alto": VERMELHO, "Anomalia": AMARELO, "Normal": "#2E7D32"}

AVISO = ("Este laudo é gerado automaticamente por um modelo de inteligência artificial "
         "e indica a probabilidade de haver agressão física nas imagens. É um indicativo "
         "para triagem e não substitui a análise humana do vídeo. O modelo analisa apenas "
         "a imagem (não analisa áudio nem identifica objetos ou pessoas) e pode errar, "
         "principalmente em cenas de esporte, ação ou movimentos bruscos sem agressão.")


def mmss(s: float | None) -> str:
    if s is None:
        return "-"
    s = max(0.0, float(s))
    m, seg = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{seg:05.2f}" if h else f"{m:02d}:{seg:05.2f}"


def codigo_verificacao(resultado: dict) -> str:
    """SHA-256 do resultado em JSON canônico (chaves ordenadas, sem espaços)."""
    canon = json.dumps(resultado, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _grafico(resultado: dict, cfg: ConfigInferencia, duracao: float | None) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lt = resultado.get("linha_do_tempo") or []
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    if lt:
        ax.bar([b["inicio_s"] for b in lt], [b["score"] for b in lt],
               width=[max(b["fim_s"] - b["inicio_s"], 1e-3) * 0.9 for b in lt], align="edge",
               color=[CORES_NIVEL.get(b.get("nivel"), CINZA) for b in lt])
        fim = duracao or lt[-1]["fim_s"]
    else:  # resultado antigo, sem linha do tempo: só os trechos críticos
        for s in resultado.get("segmentos_criticos") or []:
            ax.bar(s["inicio_s"], s["score"], width=max(s["fim_s"] - s["inicio_s"], 0.5),
                   align="edge", color=VERMELHO)
        fim = duracao or max([s["fim_s"] for s in resultado.get("segmentos_criticos") or []] + [1])
    ax.axhline(cfg.limiar_alerta_alto, ls="--", lw=0.8, color=VERMELHO)
    ax.axhline(cfg.limiar_anomalia, ls="--", lw=0.8, color=AMARELO)
    ax.set_xlim(0, fim)
    ax.set_ylim(0, 100)
    ax.set_ylabel("índice (0-100)", fontsize=8)
    ticks = ax.get_xticks()
    ax.set_xticks([t for t in ticks if 0 <= t <= fim])
    ax.set_xticklabels([mmss(t)[:-3] for t in ax.get_xticks()], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def _frame_png(video: str, t: float, largura: int = 480) -> bytes | None:
    import cv2
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > largura:
        frame = cv2.resize(frame, (largura, int(h * largura / w)))
    ok, buf = cv2.imencode(".png", frame)
    return buf.tobytes() if ok else None


def gerar_laudo(resultado: dict, saida: str | Path | io.BytesIO, metadados: dict | None = None,
                video: str | None = None, cfg: ConfigInferencia | None = None,
                max_imagens: int = 6) -> str:
    """Escreve o PDF em `saida` e devolve o código de verificação."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    if resultado.get("status") not in (None, "concluido"):
        raise ValueError("só dá pra gerar laudo de análise concluída")
    cfg = cfg or ConfigInferencia()
    meta = metadados or {}
    codigo = codigo_verificacao(resultado)
    duracao = meta.get("duracao_s") or meta.get("duracao_segundos")
    emitido = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M:%S %Z")

    est = getSampleStyleSheet()
    normal = ParagraphStyle("n", parent=est["Normal"], fontSize=9, leading=12)
    pequeno = ParagraphStyle("p", parent=normal, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))
    titulo = ParagraphStyle("t", parent=est["Title"], fontSize=16, leading=20, textColor=colors.HexColor(ROXO),
                            spaceAfter=2)
    sub = ParagraphStyle("s", parent=normal, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
    h2 = ParagraphStyle("h2", parent=est["Heading2"], fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=4,
                        textColor=colors.HexColor(ROXO))
    mono = ParagraphStyle("m", parent=normal, fontName="Courier", fontSize=7.5, leading=9.5)

    def P(txt, st=normal):
        return Paragraph(txt, st)

    def tabela(linhas, larguras, cabecalho=False):
        t = Table(linhas, colWidths=larguras)
        estilo = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                  ("FONTSIZE", (0, 0), (-1, -1), 9),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
                  ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD"))]
        if cabecalho:
            estilo += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEAF8")),
                       ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(estilo))
        return t

    hist = []
    hist.append(P("Safe Evidence", titulo))
    hist.append(P("Laudo de Análise de Vídeo por Inteligência Artificial", sub))
    hist.append(Spacer(1, 10))

    # ---- resultado em destaque ----
    ver = resultado.get("veredicto", "-")
    cor = colors.HexColor(CORES_VEREDICTO.get(ver, "#555555"))
    rotulo = ParagraphStyle("r", parent=normal, fontSize=7.5, textColor=colors.HexColor("#555555"))
    valor = ParagraphStyle("v", parent=normal, fontSize=24, leading=28)
    destaque = Table(
        [[P("ÍNDICE DE AGRESSIVIDADE", rotulo), P("VEREDICTO", rotulo), P("TRECHOS CRÍTICOS", rotulo)],
         [P(f"<b>{resultado.get('score', 0):.0f}</b><font size=12>/100</font>", valor),
          P(f"<font color='{CORES_VEREDICTO.get(ver, '#555555')}'><b>{escape(ver)}</b></font>",
            ParagraphStyle("v2", parent=valor, fontSize=18, leading=28)),
          P(f"<b>{len(resultado.get('segmentos_criticos') or [])}</b>", valor)]],
        colWidths=[6 * cm, 6 * cm, 5 * cm])
    destaque.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.2, cor),
                                  ("LINEBEFORE", (1, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    hist.append(destaque)
    hist.append(Spacer(1, 4))
    hist.append(P(f"Faixas: Normal abaixo de {cfg.limiar_anomalia:.0f}; Anomalia de {cfg.limiar_anomalia:.0f} "
                  f"a {cfg.limiar_alerta_alto:.0f}; Alerta Alto a partir de {cfg.limiar_alerta_alto:.0f}.", pequeno))

    # ---- identificação ----
    hist.append(P("1. Identificação da gravação", h2))
    h_reg = meta.get("hash_sha256")
    h_ver = resultado.get("hash_sha256_verificado")
    if h_reg and h_ver:
        integ = ("<font color='#2E7D32'><b>Íntegro</b></font> - o arquivo analisado é idêntico ao registrado"
                 if h_reg.lower() == h_ver.lower() else
                 f"<font color='{VERMELHO}'><b>DIVERGENTE</b></font> - o arquivo analisado não é o registrado")
    else:
        integ = "não verificada (hash ausente)"
    linhas = [
        ["Gravação nº", P(escape(str(meta.get("id_gravacao", "-"))))],
        ["Arquivo", P(escape(str(meta.get("arquivo", "-"))))],
        ["Data da gravação", P(escape(str(meta.get("data_gravacao", "-"))))],
        ["Dispositivo", P(escape(str(meta.get("dispositivo", "-"))))],
        ["Duração", P(mmss(duracao) if duracao else "-")],
        ["SHA-256 registrado", P(escape(h_reg or "-"), mono)],
        ["SHA-256 analisado", P(escape(h_ver or "-"), mono)],
        ["Integridade", P(integ)],
    ]
    hist.append(tabela(linhas, [4 * cm, 13 * cm]))

    # ---- linha do tempo ----
    hist.append(P("2. Linha do tempo de agressividade", h2))
    hist.append(Image(io.BytesIO(_grafico(resultado, cfg, duracao)), width=17 * cm, height=5.2 * cm))
    hist.append(P("Cada barra é um intervalo do vídeo; a altura é o maior índice naquele intervalo. "
                  "Vermelho = nível de Alerta Alto, amarelo = Anomalia, cinza = normal. "
                  "Linhas tracejadas: limiares.", pequeno))

    # ---- trechos críticos ----
    segs = resultado.get("segmentos_criticos") or []
    hist.append(P("3. Trechos críticos", h2))
    if segs:
        linhas = [["#", "Início", "Fim", "Duração", "Momento de pico", "Índice máx."]]
        for i, s in enumerate(segs, 1):
            linhas.append([str(i), mmss(s["inicio_s"]), mmss(s["fim_s"]), f"{s['fim_s'] - s['inicio_s']:.1f} s",
                           mmss(s.get("pico_s")), f"{s['score']:.1f}"])
        hist.append(tabela(linhas, [1 * cm, 2.8 * cm, 2.8 * cm, 2.4 * cm, 4 * cm, 3 * cm], cabecalho=True))
    else:
        hist.append(P("Nenhum trecho ultrapassou o limiar de agressão."))

    # ---- imagens dos picos ----
    if video and segs:
        imgs = []
        for s in sorted(segs, key=lambda s: -s["score"])[:max_imagens]:
            t = s.get("pico_s", (s["inicio_s"] + s["fim_s"]) / 2)
            png = _frame_png(video, t)
            if png:
                idx = segs.index(s) + 1
                imgs.append([Image(io.BytesIO(png), width=8 * cm, height=4.5 * cm, kind="proportional"),
                             P(f"Trecho {idx} - {mmss(t)} - índice {s['score']:.1f}", pequeno)])
        if imgs:
            if len(imgs) % 2:
                imgs.append("")
            grade = [imgs[k:k + 2] for k in range(0, len(imgs), 2)]
            g = Table(grade, colWidths=[8.5 * cm, 8.5 * cm])
            g.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            hist.append(KeepTogether([P("4. Quadros dos momentos de pico", h2), g]))

    # ---- modelo ----
    n = 5 if (video and segs) else 4
    bloco = [P(f"{n}. Dados técnicos da análise", h2)]
    tp = resultado.get("tempo_processamento_ms")
    linhas = [
        ["Modelo", P(escape(str(resultado.get("versao_modelo", "-"))))],
        ["Método", P("YOLOv9-C (backbone GELAN) adaptado para classificação de frames em "
                     "violência / não violência; treinado no dataset XD-Violence")],
        ["Amostragem", P(f"{cfg.fps_amostragem:g} quadros por segundo; suavização de {cfg.janela_suavizacao_s:g} s; "
                         f"índice = média dos {cfg.topo_s:g} s mais altos")],
        ["Tempo de processamento", P(f"{tp / 1000:.1f} s" + (f" ({tp / 1000 / duracao * 100:.1f}% da duração)"
                                                             if tp and duracao else "") if tp else "-")],
        ["Emitido em", P(emitido)],
    ]
    bloco.append(tabela(linhas, [4 * cm, 13 * cm]))
    bloco.append(Spacer(1, 6))
    bloco.append(P("<b>Código de verificação do laudo</b> (SHA-256 do resultado da análise):", pequeno))
    bloco.append(P(codigo, mono))
    hist.append(KeepTogether(bloco))
    hist.append(Spacer(1, 8))
    hist.append(P("<b>Aviso.</b> " + AVISO, pequeno))

    def rodape(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(2 * cm, 1.2 * cm, f"Safe Evidence - gravação {meta.get('id_gravacao', '-')} - "
                                            f"verificação {codigo[:16]}...")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"página {doc.page}")
        canvas.restoreState()

    destino = saida if isinstance(saida, io.BytesIO) else str(saida)
    doc = SimpleDocTemplate(destino, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                            title=f"Laudo Safe Evidence - gravação {meta.get('id_gravacao', '')}",
                            author="Safe Evidence (IA)")
    doc.build(hist, onFirstPage=rodape, onLaterPages=rodape)
    return codigo
