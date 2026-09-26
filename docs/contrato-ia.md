# Contrato entre o backend e o serviço de IA

Cópia do contrato definido no backend do Safe Evidence. O serviço deste
repositório (`servico/`) implementa exatamente estas duas chamadas.

```
 backend                                   serviço de IA
    │  1. POST /analisar  (pedido)              │
    │ ─────────────────────────────────────────▶│
    │ ◀──────────────── 202 Accepted ───────────│  responde na hora
    │                                           │  baixa o vídeo, roda o modelo...
    │  2. POST /api/ia/resultado/:id (callback) │
    │ ◀─────────────────────────────────────────│
    │ ─────────────── 201 Created ─────────────▶│
```

A análise demora (RN16 permite até 50% da duração do vídeo), então o serviço
aceita o pedido na hora e devolve o resultado quando terminar.

## 1. Pedido de análise (backend → IA)

`POST {IA_SERVICE_URL}/analisar`

```
Content-Type: application/json
Authorization: Bearer <IA_TOKEN>        (se IA_TOKEN estiver configurado)
```

```json
{
  "id_gravacao": 12,
  "video_url": "https://storage.googleapis.com/...(URL assinada, vale 60 min)",
  "video_url_expira_em": "2026-09-22T23:50:00.000Z",
  "hash_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "duracao_segundos": 16.44,
  "callback_url": "https://backend.../api/ia/resultado/12",
  "tentativa": 1
}
```

| Campo | O que é |
| --- | --- |
| `video_url` | Link pra baixar o `.mp4` com um GET simples, sem login. Expira em 60 min. |
| `hash_sha256` | Hash registrado quando o vídeo chegou. **Confira antes de analisar.** |
| `duracao_segundos` | Duração do vídeo. |
| `callback_url` | Para onde mandar o resultado. Use exatamente esta URL. |
| `tentativa` | 1 na primeira vez. Se for maior, é uma nova tentativa depois de um erro. |

**Resposta esperada:** `202` (ou qualquer 2xx) sem esperar a análise terminar.
Outro código ou mais de 10 s → o backend marca `Erro` e tenta de novo depois.

## 2. Resultado (IA → backend)

`POST {callback_url}` com `Authorization: Bearer <IA_CALLBACK_TOKEN>`

### Análise concluída

```json
{
  "status": "concluido",
  "score": 95.24,
  "veredicto": "Alerta Alto",
  "segmentos_criticos": [
    { "inicio_s": 2.0, "fim_s": 5.0, "score": 96.1 },
    { "inicio_s": 12.0, "fim_s": 14.0, "score": 88.7 }
  ],
  "versao_modelo": "yolov9c-cls-xdviolence-v1",
  "tempo_processamento_ms": 4210,
  "hash_sha256_verificado": "e3b0c442...b855"
}
```

| Campo | Obrigatório | Regra |
| --- | --- | --- |
| `status` | sim | `"concluido"` |
| `score` | sim | 0 a 100: probabilidade de agressão no vídeo como um todo (Tabela 15: `indice_probabilidade`) |
| `veredicto` | sim | exatamente `"Alerta Alto"`, `"Anomalia"` ou `"Normal"` |
| `segmentos_criticos` | não | trechos com indício de agressão, em segundos desde o início; `inicio_s <= fim_s` |
| `versao_modelo` | sim | nome/versão do modelo (RN07, rastreabilidade) |
| `tempo_processamento_ms` | não | tempo gasto; usado pra medir o RN16 |
| `hash_sha256_verificado` | recomendado | SHA-256 do arquivo **de fato** analisado |

Hash diferente do registrado → backend recusa (409) e marca `Erro` (cadeia de custódia).

### Erro na análise

```json
{ "status": "erro", "erro": "descrição curta do problema" }
```

### Respostas do backend

| Código | Quando |
| --- | --- |
| `201` | resultado gravado |
| `200` | erro registrado |
| `400` | campo inválido |
| `401` | token errado |
| `409` | gravação já tinha análise concluída, ou hash não confere |

## Novas tentativas (RE08)

- Falhou? A gravação vira `Erro` e o backend tenta de novo após 1, 2, 4, 8... min,
  até `IA_MAX_TENTATIVAS` (padrão 5).
- Aceitou mas não respondeu em `IA_TIMEOUT_MINUTOS` (padrão 30)? Conta como erro.
- Por isso o serviço aceita receber o **mesmo `id_gravacao` mais de uma vez**.

## Rede

O serviço precisa alcançar o backend (download do vídeo e `callback_url`).
Um serviço no Colab não alcança `192.168.x.x`: aí o backend precisa estar
publicado (Cloud Run) ou exposto por um túnel.
