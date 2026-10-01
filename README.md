# Safe Evidence — Detecção de violência em vídeo com YOLOv9

Módulo de IA do TCC **Safe Evidence**. Recebe um vídeo (ESP32-CAM / celular),
classifica os frames como **violência** ou **não violência** com um YOLOv9
adaptado para classificação e devolve um laudo com score, veredicto e os
trechos críticos, no formato combinado com o backend
([docs/contrato-ia.md](docs/contrato-ia.md)).

```
vídeo ──► amostra 4 frames/s ──► YOLOv9-C-cls ──► P(violência) por frame
      ──► média móvel (1 s) ──► trechos acima do limiar ──► score + veredicto
```

## Estrutura

```
configs/yolov9c-cls.yaml     arquitetura: backbone GELAN do YOLOv9-C + cabeça de classificação
ia/
  config.py                  limiares e parâmetros (tudo configurável por variável de ambiente)
  xd_violence.py             leitura dos rótulos/anotações do XD-Violence
  video.py                   leitura do vídeo + classificação dos frames
  agregacao.py               frames -> score, veredicto e segmentos (sem torch; testável)
scripts/
  baixar_xd_violence.py      download do dataset (Hugging Face)
  preparar_dataset.py        extrai frames rotulados e divide train/val/test por vídeo
  treinar.py                 treina e copia o melhor peso para pesos/melhor.pt
  avaliar.py                 métricas por frame e por vídeo (+ checagem do RN16)
  analisar_video.py          analisa um vídeo (JSON, vídeo anotado, gráfico)
servico/                     API FastAPI que implementa o contrato com o backend
notebooks/treino_colab.ipynb passo a passo no Google Colab (GPU grátis)
tests/                       testes (rodam sem GPU e sem o modelo)
```

## Por que "YOLOv9 de classificação"?

O YOLOv9 original é um detector (caixas). Os datasets de violência, porém,
são rotulados por **vídeo/trecho**, não por caixa. Então usamos o **backbone
GELAN do YOLOv9-C** (camadas 0–9 do `yolov9c.yaml` oficial, idênticas) com
uma cabeça `Classify` no lugar da cabeça de detecção. Os pesos do backbone
pré-treinado no COCO (`yolov9c.pt`) são transferidos camada a camada
(*transfer learning*), e só a cabeça começa do zero.

## Dataset: XD-Violence

[XD-Violence](https://roc-ng.github.io/XD-Violence/) (Wu et al., ECCV 2020):
4.754 vídeos, 217 h, 6 tipos de violência. Detalhes que importam aqui:

| Parte | Vídeos | Rótulo | Como usamos |
| --- | --- | --- | --- |
| Treino | 3.954 | só por vídeo (fraco) | vídeos **normais** (`label_A`) viram negativos; violentos só com `--usar-fracos` |
| Teste | 800 | **por frame** (início/fim de cada trecho) | fonte principal dos frames violentos, e dos normais dentro deles |

Categorias no nome do arquivo (`..._label_B1-B2-0.mp4`):
`A` normal · `B1` luta · `B2` tiro · `B4` tumulto · `B5` abuso · `B6` acidente de carro · `G` explosão.
Por padrão usamos **B1, B2, B4, B5** (agressão entre pessoas); acidente e explosão ficam de fora
(`--categorias` muda isso).

Cuidados já tratados no `preparar_dataset.py`:
- **Divisão por vídeo**: frames de um mesmo vídeo nunca caem em splits diferentes (senão a acurácia fica inflada).
- **Margem nas bordas** (`--margem-s 0.5`): frames perto do início/fim de um trecho anotado são descartados, porque o rótulo ali é ambíguo.
- Vídeos com rótulo fraco só entram no **treino**, nunca na avaliação.
- `--balancear` iguala o número de frames das duas classes no treino.

> Como a anotação por frame só existe no conjunto de teste oficial, a
> avaliação aqui é feita num *split nosso* desses vídeos. Os números **não
> são comparáveis** diretamente com o AP publicado do benchmark XD-Violence —
> vale dizer isso no texto do TCC.

Também funciona com qualquer dataset organizado em pastas
(ex.: *Real Life Violence Situations*): `--formato pastas`.

## Como rodar

O jeito mais fácil é o notebook [`notebooks/treino_colab.ipynb`](notebooks/treino_colab.ipynb)
(Colab com GPU T4). Localmente:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. dataset (subconjunto: vídeos de teste + 300 normais do treino)
python scripts/baixar_xd_violence.py --partes test_videos --normais-treino 300

# 2. frames rotulados
python scripts/preparar_dataset.py --formato xd \
    --videos data/xd_violence --anotacoes data/xd_violence/test_annotations.txt --balancear

# 3. treino (GPU recomendada)
python scripts/treinar.py --dados data/frames --epocas 30 --lote 64

# 4. calibra score/veredicto nos vídeos de validação (grava pesos/calibracao.json)
python scripts/calibrar.py --pesos pesos/melhor.pt --dados data/frames

# 5. avaliação final no teste (já usa a calibração)
python scripts/avaliar.py --pesos pesos/melhor.pt --dados data/frames --por-video

# 6. um vídeo qualquer
python scripts/analisar_video.py video.mp4 --video-saida anotado.mp4 --grafico linha.png
```

Baseline para comparação no TCC (mesmo dataset, outro modelo):

```bash
python scripts/treinar.py --modelo yolo11n-cls.pt --nome yolo11n-cls
```

## Serviço para o backend

```bash
cp .env.example .env   # ajuste IA_CALLBACK_TOKEN igual ao do backend
set -a && source .env && set +a
uvicorn servico.app:app --host 0.0.0.0 --port 8090
```

No `.env` do backend: `IA_SERVICE_URL=http://<host>:8090`.
Endpoints: `POST /analisar` (contrato) e `GET /saude`.

Além de score, veredicto e trechos críticos, o callback leva `linha_do_tempo`:
uma barra por segundo com o score daquele segundo, o frame mais forte
(`pico_s`) e o `nivel` (`normal` / `alerta` / `critico`) — pronto para o
gráfico de linha do tempo e para pular o player até o momento suspeito.

O serviço baixa o vídeo em streaming calculando o SHA-256, recusa se o hash
não bater (cadeia de custódia), analisa um vídeo por vez e envia o callback
(com até 3 tentativas). Pedidos repetidos do mesmo `id_gravacao` enquanto ele
ainda está na fila são ignorados; depois de terminar, aceita de novo (RE08).

Docker: `docker build -t safe-evidence-ia . && docker run -p 8090:8090 --env-file .env safe-evidence-ia`
(coloque `pesos/melhor.pt` antes do build).

## Laudo em PDF ("Download de Segurança")

`ia/laudo.py` gera o laudo de uma gravação: índice e veredicto, identificação
do arquivo com os dois SHA-256 (registrado x analisado), gráfico da linha do
tempo, tabela de trechos críticos, quadros dos momentos de pico, dados do
modelo e um **código de verificação** (SHA-256 do resultado da análise em JSON
canônico; recalcule a partir do resultado guardado no banco para provar que o
laudo não foi editado).

```bash
# analisar um vídeo e já gerar o laudo
python scripts/analisar_video.py video.mp4 --laudo laudo.pdf
# a partir de um resultado salvo (JSON do callback)
python scripts/gerar_laudo.py resultado.json --metadados gravacao.json --video video.mp4
```

Pelo serviço, o backend chama `POST /laudo` com
`{"resultado": <JSON do callback>, "metadados": {"id_gravacao", "arquivo", "data_gravacao",
"dispositivo", "duracao_s", "hash_sha256"}, "video_url": "...(opcional)"}` e recebe o PDF
(`application/pdf`). Com `video_url`, o hash do vídeo é conferido antes (409 se não bater)
e os quadros de pico entram no laudo.

## Como o score e o veredicto são calculados

| Etapa | Padrão | Variável |
| --- | --- | --- |
| Frames analisados por segundo | 4 | `FPS_AMOSTRAGEM` |
| Suavização (média móvel) | 1 s | `JANELA_SUAVIZACAO_S` |
| Frame conta como violento se P ≥ | 0,5 | `LIMIAR_FRAME` |
| Trecho mínimo / buraco máximo unido | 1 s / 1 s | `DURACAO_MIN_SEGMENTO_S` / `UNIAO_MAX_S` |
| **Score** = média das maiores probabilidades somando | 2 s | `TOPO_S` |
| Alerta Alto / Anomalia a partir de | 70 / 40 | `LIMIAR_ALERTA_ALTO` / `LIMIAR_ANOMALIA` |

O score usa uma duração fixa (2 s) em vez da média do vídeo inteiro: uma briga
de 4 s num vídeo de 10 minutos tem que dar alerta, e a média simples diluiria
isso. A suavização evita que um único frame isolado dispare alerta.
O repositório já traz `pesos/calibracao.json` com os valores calibrados
(janela 5 s, topo 15 s, Anomalia ≥ 62,76, Alerta Alto ≥ 94,24).
Os valores da tabela são só o ponto de partida: `scripts/calibrar.py` escolhe
janela, duração do topo e limiares nos vídeos de **validação** e grava em
`pesos/calibracao.json`, que o serviço e a avaliação carregam sozinhos
(variável de ambiente continua tendo prioridade). Nunca calibre no teste.

## Testes

```bash
python -m unittest discover -s tests -t .
```

Usam vídeos sintéticos (frames vermelhos = "violência") e um classificador
falso, então rodam em segundos sem GPU. Cobrem: rótulos do XD-Violence,
extração/divisão do dataset, agregação e o fluxo completo do contrato
(download → hash → análise → callback) contra um backend falso.

## Limitações conhecidas

- Classificação **por frame**: não usa movimento entre frames. Um abraço ou
  esporte de contato pode confundir; a suavização temporal ajuda, mas um
  modelo temporal (ex.: LSTM sobre as features do YOLO) seria o próximo passo.
- XD-Violence vem de filmes e YouTube; a câmera do Safe Evidence (ESP32-CAM,
  baixa resolução) tem outra aparência. Vale gravar alguns vídeos próprios
  para validar e, se possível, ajustar (fine-tuning).

## Licenças e citação

- Ultralytics (YOLOv9 nesta implementação): AGPL-3.0.
- XD-Violence: uso acadêmico. Cite:
  Wu, P. et al. *Not only Look, but also Listen: Learning Multimodal Violence
  Detection under Weak Supervision*. ECCV 2020.
- YOLOv9: Wang, C.-Y.; Yeh, I.-H.; Liao, H.-Y. M. *YOLOv9: Learning What You
  Want to Learn Using Programmable Gradient Information*. 2024.
