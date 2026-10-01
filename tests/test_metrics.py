"""Testes das metricas do EASSE-PT."""

import unittest

from easse_pt.metrics import (
    corpus_bleu,
    corpus_features,
    corpus_flesch_brasil,
    corpus_sari,
    flesch_brasil,
    flesch_brasil_level,
    sentence_features,
    sentence_sari,
)
from easse_pt.metrics.readability import count_syllables

ORIG = "O Ministério da Saúde anunciou que a campanha de vacinação será prorrogada."
NAT = "O Ministério da Saúde anunciou que a campanha de vacinação vai durar mais tempo."
FORTE = "A vacinação vai continuar por mais tempo."


class TestSari(unittest.TestCase):
    def test_saida_igual_a_referencia_supera_copia_da_origem(self):
        igual = sentence_sari(ORIG, NAT, [NAT, FORTE])["sari"]
        copia = sentence_sari(ORIG, ORIG, [NAT, FORTE])["sari"]
        self.assertGreater(igual, copia)

    def test_dentro_do_intervalo(self):
        for k, v in sentence_sari(ORIG, FORTE, [NAT, FORTE]).items():
            with self.subTest(metrica=k):
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 100)

    def test_parcelas_compoem_a_nota(self):
        s = sentence_sari(ORIG, NAT, [NAT, FORTE])
        media = (s["sari_keep"] + s["sari_del"] + s["sari_add"]) / 3
        self.assertAlmostEqual(s["sari"], media, places=9)

    def test_exige_referencia(self):
        with self.assertRaises(ValueError):
            sentence_sari(ORIG, NAT, [])

    def test_corpus_valida_tamanhos(self):
        with self.assertRaises(ValueError):
            corpus_sari([ORIG], [NAT, NAT], [[NAT]])

    def test_corpus_e_media_das_sentencas(self):
        c = corpus_sari([ORIG, ORIG], [NAT, FORTE], [[NAT, NAT], [FORTE, FORTE]])["sari"]
        m = (
            sentence_sari(ORIG, NAT, [NAT, FORTE])["sari"]
            + sentence_sari(ORIG, FORTE, [NAT, FORTE])["sari"]
        ) / 2
        self.assertAlmostEqual(c, m, places=9)


class TestBleu(unittest.TestCase):
    def test_perfeito_quando_igual_a_referencia(self):
        self.assertAlmostEqual(corpus_bleu([NAT, FORTE], [[NAT, FORTE]])["bleu"], 100.0, places=4)

    def test_cai_com_saida_diferente(self):
        bom = corpus_bleu([NAT], [[NAT]])["bleu"]
        ruim = corpus_bleu(["Texto completamente diferente aqui."], [[NAT]])["bleu"]
        self.assertLess(ruim, bom)

    def test_registra_o_motor_usado(self):
        self.assertIn(corpus_bleu([NAT], [[NAT]])["bleu_engine"], ("sacrebleu", "interno"))


class TestFleschBrasil(unittest.TestCase):
    def test_texto_simples_maior_que_complexo(self):
        facil = flesch_brasil("O gato dorme. A casa é azul.")
        dificil = flesch_brasil(
            "A implementação de metodologias interdisciplinares pressupõe a "
            "reconfiguração epistemológica das práticas institucionais vigentes."
        )
        self.assertGreater(facil, dificil)

    def test_faixas(self):
        self.assertEqual(flesch_brasil_level(90), "muito facil")
        self.assertEqual(flesch_brasil_level(60), "facil")
        self.assertEqual(flesch_brasil_level(30), "dificil")
        self.assertEqual(flesch_brasil_level(10), "muito dificil")

    def test_texto_vazio(self):
        self.assertEqual(flesch_brasil(""), 0.0)

    def test_contagem_de_silabas(self):
        self.assertEqual(count_syllables("casa"), 2)
        self.assertEqual(count_syllables("pão"), 1)
        self.assertEqual(count_syllables("universidade"), 6)
        self.assertEqual(count_syllables("a"), 1)

    def test_corpus_agrega(self):
        v = corpus_flesch_brasil(["O gato dorme.", "A casa é azul."])
        self.assertTrue(0 < v < 200)


class TestAtributos(unittest.TestCase):
    def test_copia_exata(self):
        f = sentence_features(ORIG, ORIG)
        self.assertEqual(f["exact_copy"], 1.0)
        self.assertAlmostEqual(f["levenshtein_sim"], 1.0)
        self.assertAlmostEqual(f["compression_ratio"], 1.0)

    def test_compressao(self):
        f = sentence_features(ORIG, FORTE)
        self.assertLess(f["compression_ratio"], 1.0)
        self.assertGreater(f["deleted_words"], 0)

    def test_divisao_de_sentencas(self):
        f = sentence_features(
            "O gato dormiu e a casa ficou quieta.", "O gato dormiu. A casa ficou quieta."
        )
        self.assertAlmostEqual(f["sentence_splits"], 2.0)

    def test_corpus_agrega(self):
        out = corpus_features([ORIG, ORIG], [NAT, ORIG])
        self.assertAlmostEqual(out["exact_copy"], 50.0)


if __name__ == "__main__":
    unittest.main()
