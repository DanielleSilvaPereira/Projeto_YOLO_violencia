import unittest

from ia.agregacao import agregar
from ia.config import ConfigInferencia


def cfg():
    return ConfigInferencia(fps_amostragem=4, janela_suavizacao_s=0.5, limiar_frame=0.5,
                            duracao_min_segmento_s=1.0, uniao_max_s=1.0, topo_s=2.0,
                            limiar_alerta_alto=70, limiar_anomalia=40)


def serie(dur_s, trechos, fps=4, alto=0.95, baixo=0.05):
    t = [i / fps for i in range(int(dur_s * fps))]
    p = [alto if any(a <= x < b for a, b in trechos) else baixo for x in t]
    return t, p


class TestAgregacao(unittest.TestCase):
    def test_video_normal(self):
        t, p = serie(20, [])
        r = agregar(t, p, cfg(), 20)
        self.assertEqual(r.veredicto, "Normal")
        self.assertEqual(r.segmentos_criticos, [])
        self.assertLess(r.score, 40)

    def test_trecho_curto_em_video_longo_ainda_alerta(self):
        t, p = serie(60, [(10, 14)])
        r = agregar(t, p, cfg(), 60)
        self.assertEqual(r.veredicto, "Alerta Alto")
        self.assertEqual(len(r.segmentos_criticos), 1)
        s = r.segmentos_criticos[0]
        self.assertAlmostEqual(s["inicio_s"], 10, delta=0.5)
        self.assertAlmostEqual(s["fim_s"], 14, delta=0.5)
        self.assertLessEqual(s["inicio_s"], s["fim_s"])

    def test_une_buracos_curtos_e_descarta_picos(self):
        t, p = serie(30, [(2, 5), (5.5, 8), (20, 20.25)])
        r = agregar(t, p, cfg(), 30)
        self.assertEqual(len(r.segmentos_criticos), 1)  # o pico de 0,25 s some
        self.assertAlmostEqual(r.segmentos_criticos[0]["fim_s"], 8, delta=0.5)

    def test_fim_nao_passa_da_duracao(self):
        t, p = serie(5, [(3, 5)])
        r = agregar(t, p, cfg(), 5)
        self.assertLessEqual(r.segmentos_criticos[-1]["fim_s"], 5)

    def test_faixa_anomalia(self):
        t, p = serie(20, [(0, 20)], alto=0.55, baixo=0.55)
        r = agregar(t, p, cfg(), 20)
        self.assertEqual(r.veredicto, "Anomalia")

    def test_vazio(self):
        r = agregar([], [], cfg())
        self.assertEqual((r.score, r.veredicto), (0.0, "Normal"))


if __name__ == "__main__":
    unittest.main()
