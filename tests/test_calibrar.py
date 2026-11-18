import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ia.config import ConfigInferencia
from scripts.calibrar import auc, buscar, escolher_limiares

RAIZ = Path(__file__).resolve().parents[1]


def linha(dur, trechos, ruido, rng, fps=4):
    t = [i / fps for i in range(int(dur * fps))]
    p = [0.95 if any(a <= x < b for a, b in trechos) else float(rng.random() < ruido) * 0.9 for x in t]
    return {"tempos": t, "probs": p, "duracao": dur}


class TestCalibrar(unittest.TestCase):
    def test_limiares(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([10, 20, 30, 60, 55, 70, 80, 90])
        r = escolher_limiares(y, s)
        self.assertEqual(r["limiar"], 55)
        self.assertAlmostEqual(r["acuracia_balanceada"], 0.875)
        self.assertEqual(r["limiar_alerta"], 70)  # primeiro corte sem falso alarme
        self.assertEqual(auc(y, s), 15 / 16)

    def test_busca_separa_ruido_de_briga(self):
        # normais com picos isolados de falso positivo; violentos com 5 s contínuos
        rng = np.random.default_rng(0)
        linhas = {}
        for i in range(20):
            linhas[f"n{i}"] = {"violento": False, **linha(120, [], 0.08, rng)}
            linhas[f"v{i}"] = {"violento": True, **linha(60, [(20, 25)], 0.08, rng)}
        params, tabela = buscar(linhas, ConfigInferencia())
        self.assertGreaterEqual(tabela[0]["acuracia_balanceada"], 0.95)
        self.assertLessEqual(params["LIMIAR_ANOMALIA"], params["LIMIAR_ALERTA_ALTO"])
        self.assertAlmostEqual(params["LIMIAR_FRAME"], params["LIMIAR_ANOMALIA"] / 100, places=3)

    def test_config_le_calibracao(self):
        with tempfile.TemporaryDirectory() as d:
            arq = Path(d) / "c.json"
            arq.write_text(json.dumps({"parametros": {"LIMIAR_ANOMALIA": 61.5, "TOPO_S": 8}}))
            env = {**os.environ, "CALIBRACAO": str(arq), "TOPO_S": "4"}
            cod = "from ia.config import ConfigInferencia as C; c=C(); print(c.limiar_anomalia, c.topo_s)"
            out = subprocess.run([sys.executable, "-c", cod], cwd=RAIZ, env=env,
                                 capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["61.5", "4.0"])  # json aplicado; variável de ambiente tem prioridade


if __name__ == "__main__":
    unittest.main()
