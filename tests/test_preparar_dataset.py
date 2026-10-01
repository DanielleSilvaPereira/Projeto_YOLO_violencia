import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import preparar_dataset
from tests.util import criar_video


class TestPrepararXD(unittest.TestCase):
    def test_xd_anotado_e_filtro(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vids = d / "videos"
            anot = []
            # 12 vídeos violentos anotados (luta), 12 normais, 1 explosão (fora do filtro), 1 sem anotação
            for i in range(12):
                nome = f"filme{i}__#1_label_B1-0-0"
                criar_video(vids / "test_videos" / f"{nome}.mp4", 96, 24, [(24, 71)])
                anot.append(f"{nome} 24 71")
                criar_video(vids / "1-1004" / f"normal{i}__#1_label_A.mp4", 48, 24)
            criar_video(vids / "test_videos" / "bum__#1_label_G-0-0.mp4", 48, 24, [(0, 47)])
            anot.append("bum__#1_label_G-0-0 0 47")
            criar_video(vids / "1-1004" / "luta_treino__#1_label_B1-0-0.mp4", 48, 24, [(0, 47)])
            (d / "anot.txt").write_text("\n".join(anot))

            saida = d / "frames"
            preparar_dataset.main(["--formato", "xd", "--videos", str(vids), "--anotacoes", str(d / "anot.txt"),
                                   "--saida", str(saida), "--fps", "4", "--margem-s", "0.25",
                                   "--workers", "1", "--proporcoes", "0.5,0.25,0.25"])
            resumo = json.loads((saida / "resumo.json").read_text())
            self.assertEqual(resumo["ignorados"], {"categoria_fora_do_filtro": 1, "violento_sem_anotacao": 1})

            with open(saida / "manifesto.csv") as f:
                linhas = list(csv.DictReader(f))
            self.assertTrue(linhas)
            # rótulo confere com a cor do frame e com a anotação
            import cv2
            for l in linhas:
                img = cv2.imread(str(saida / l["arquivo"]))
                vermelho = img[..., 2].mean() > img[..., 1].mean()
                self.assertEqual(vermelho, l["classe"] == "violencia", l)
                if l["modo"] == "anotado":
                    fr = int(l["frame"])
                    self.assertFalse(18 <= fr <= 29 or 66 <= fr <= 77, "margem na borda não respeitada")
            # nenhum vídeo em mais de um split
            por_video = {}
            for l in linhas:
                por_video.setdefault(l["video"], set()).add(l["split"])
            self.assertTrue(all(len(s) == 1 for s in por_video.values()))

    def test_usar_fracos_vai_so_pro_treino(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in range(6):
                criar_video(d / "v" / f"luta{i}__#1_label_B5-0-0.mp4", 24, 24, [(0, 23)])
                criar_video(d / "v" / f"ok{i}__#1_label_A.mp4", 24, 24)
            saida = d / "f"
            preparar_dataset.main(["--formato", "xd", "--videos", str(d / "v"), "--saida", str(saida),
                                   "--usar-fracos", "--workers", "1", "--fps", "2"])
            with open(saida / "manifesto.csv") as f:
                fracos = [l for l in csv.DictReader(f) if l["modo"] == "violento"]
            self.assertTrue(fracos)
            self.assertTrue(all(l["split"] == "train" for l in fracos))


class TestPrepararPastas(unittest.TestCase):
    def test_pastas_e_balanceamento(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in range(8):
                criar_video(d / "rlvs" / "Violence" / f"V_{i}.mp4", 24, 24, [(0, 23)])
            for i in range(16):
                criar_video(d / "rlvs" / "NonViolence" / f"NV_{i}.mp4", 24, 24)
            saida = d / "f"
            preparar_dataset.main(["--formato", "pastas", "--videos", str(d / "rlvs"), "--saida", str(saida),
                                   "--workers", "2", "--fps", "4", "--balancear"])
            tr = json.loads((saida / "resumo.json").read_text())["frames"]["train"]
            self.assertEqual(tr["violencia"], tr["nao_violencia"])
            n_v = len(list((saida / "train" / "violencia").glob("*.jpg")))
            n_n = len(list((saida / "train" / "nao_violencia").glob("*.jpg")))
            self.assertEqual(n_v, n_n)
            va = json.loads((saida / "resumo.json").read_text())["frames"]["val"]
            if len(va) == 2:
                self.assertEqual(va["violencia"], va["nao_violencia"])


if __name__ == "__main__":
    unittest.main()
