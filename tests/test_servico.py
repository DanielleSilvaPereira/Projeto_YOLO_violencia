"""Testa o fluxo do contrato-ia de ponta a ponta com um backend falso local."""

import hashlib
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ia.config import ConfigInferencia
from servico.processamento import Processador
from tests.util import classificador_falso, criar_video


class BackendFalso(BaseHTTPRequestHandler):
    video = b""
    callbacks: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.video)))
        self.end_headers()
        self.wfile.write(self.video)

    def do_POST(self):
        corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        BackendFalso.callbacks.append((self.path, self.headers.get("Authorization"), corpo))
        self.send_response(201)
        self.end_headers()


class TestServico(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p = criar_video(Path(cls.tmp.name) / "v.mp4", 24 * 20, 24, [(24 * 8, 24 * 12)])  # briga de 8 a 12 s
        BackendFalso.video = p.read_bytes()
        cls.hash = hashlib.sha256(BackendFalso.video).hexdigest()
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), BackendFalso)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        BackendFalso.callbacks = []
        self.proc = Processador(ConfigInferencia(versao_modelo="teste-v1"), callback_token="segredo",
                                classificar=classificador_falso)

    def pedido(self, id_, hash_=None):
        return {"id_gravacao": id_, "video_url": f"{self.base}/video.mp4", "hash_sha256": hash_ or self.hash,
                "duracao_segundos": 20, "callback_url": f"{self.base}/api/ia/resultado/{id_}", "tentativa": 1}

    def esperar_callbacks(self, n, limite=30):
        fim = time.time() + limite
        while len(BackendFalso.callbacks) < n and time.time() < fim:
            time.sleep(0.05)
        return BackendFalso.callbacks

    def test_concluido(self):
        self.assertTrue(self.proc.enfileirar(self.pedido(12)))
        caminho, auth, r = self.esperar_callbacks(1)[0]
        self.assertEqual(caminho, "/api/ia/resultado/12")
        self.assertEqual(auth, "Bearer segredo")
        self.assertEqual(r["status"], "concluido")
        self.assertEqual(r["veredicto"], "Alerta Alto")
        self.assertIn(r["veredicto"], ("Alerta Alto", "Anomalia", "Normal"))
        self.assertTrue(0 <= r["score"] <= 100)
        self.assertEqual(r["versao_modelo"], "teste-v1")
        self.assertEqual(r["hash_sha256_verificado"], self.hash)
        self.assertNotIn("_linha_do_tempo", r)
        seg = r["segmentos_criticos"]
        self.assertEqual(len(seg), 1)
        self.assertAlmostEqual(seg[0]["inicio_s"], 8, delta=0.6)
        self.assertAlmostEqual(seg[0]["fim_s"], 12, delta=0.6)

    def test_hash_divergente_vira_erro(self):
        self.proc.enfileirar(self.pedido(13, hash_="0" * 64))
        _, _, r = self.esperar_callbacks(1)[0]
        self.assertEqual(r["status"], "erro")
        self.assertIn("hash", r["erro"])

    def test_pedido_repetido_e_campos(self):
        self.assertTrue(self.proc.enfileirar(self.pedido(14)))
        self.assertFalse(self.proc.enfileirar(self.pedido(14)))  # já na fila
        self.esperar_callbacks(1)
        time.sleep(0.2)
        self.assertTrue(self.proc.enfileirar(self.pedido(14)))  # nova tentativa depois de terminar
        self.esperar_callbacks(2)
        with self.assertRaises(ValueError):
            self.proc.enfileirar({"id_gravacao": 1})


if __name__ == "__main__":
    unittest.main()
