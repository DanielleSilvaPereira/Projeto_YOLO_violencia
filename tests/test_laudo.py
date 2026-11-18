import tempfile
import unittest
from pathlib import Path

from ia.config import ConfigInferencia
from ia.laudo import codigo_verificacao, gerar_laudo, mmss
from ia.video import analisar_video
from tests.util import classificador_falso, criar_video


class TestLaudo(unittest.TestCase):
    def test_mmss(self):
        self.assertEqual(mmss(75.5), "01:15.50")
        self.assertEqual(mmss(3725), "1:02:05.00")

    def test_codigo_nao_depende_da_ordem(self):
        self.assertEqual(codigo_verificacao({"a": 1, "b": 2}), codigo_verificacao({"b": 2, "a": 1}))
        self.assertNotEqual(codigo_verificacao({"a": 1}), codigo_verificacao({"a": 2}))

    def test_gera_pdf(self):
        cfg = ConfigInferencia(fps_amostragem=4, janela_suavizacao_s=1, topo_s=2, limiar_frame=0.5,
                               limiar_anomalia=40, limiar_alerta_alto=70)
        with tempfile.TemporaryDirectory() as d:
            v = criar_video(Path(d) / "v.mp4", 200, 10, [(80, 120)])
            r = {"status": "concluido", **analisar_video(str(v), classificador_falso, cfg)}
            r.pop("_linha_do_tempo")
            saida = Path(d) / "laudo.pdf"
            cod = gerar_laudo(r, saida, {"id_gravacao": 7, "duracao_s": 20}, video=str(v), cfg=cfg)
            dados = saida.read_bytes()
        self.assertTrue(dados.startswith(b"%PDF"))
        self.assertGreater(len(dados), 10_000)
        self.assertEqual(cod, codigo_verificacao(r))

    def test_recusa_erro(self):
        with self.assertRaises(ValueError):
            gerar_laudo({"status": "erro", "erro": "x"}, "/dev/null")


if __name__ == "__main__":
    unittest.main()
