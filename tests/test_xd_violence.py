import tempfile
import unittest
from pathlib import Path

from ia import xd_violence as xd


class TestXD(unittest.TestCase):
    def test_categorias(self):
        self.assertEqual(xd.categorias_do_nome("A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4"), ["A"])
        self.assertEqual(xd.categorias_do_nome("x/Bad.Boys__#1_label_B1-B2-0.mp4"), ["B1", "B2"])
        self.assertEqual(xd.categorias_do_nome("v=abc__#1_label_G-0-0.mp4"), ["G"])
        with self.assertRaises(ValueError):
            xd.categorias_do_nome("sem_rotulo.mp4")

    def test_id_video(self):
        self.assertEqual(xd.id_video("pasta/filme_label_B1-0-0.mp4"), "filme_label_B1-0-0")
        self.assertEqual(xd.id_video("filme_label_B1-0-0"), "filme_label_B1-0-0")

    def test_anotacoes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("v1_label_B1-0-0.mp4 10 20 40 50\nv2_label_B5-0-0 0 5\n\n")
            a = xd.ler_anotacoes(p)
        self.assertEqual(a["v1_label_B1-0-0"], [(10, 20), (40, 50)])
        self.assertEqual(a["v2_label_B5-0-0"], [(0, 5)])
        self.assertTrue(xd.frame_violento(20, a["v1_label_B1-0-0"]))
        self.assertFalse(xd.frame_violento(30, a["v1_label_B1-0-0"]))


if __name__ == "__main__":
    unittest.main()
