"""Testes da tokenizacao PT-BR. Rodam com pytest ou com `python -m unittest`."""

import unittest

from easse_pt.tokenization import (
    _regex_sentences,
    detokenize,
    normalize,
    sentence_tokenize,
    tokenize,
    word_tokenize,
)


class TestTokenizacao(unittest.TestCase):
    def test_word_tokenize_basico(self):
        toks = word_tokenize("O gato, que era preto, dormiu.")
        self.assertIn("gato", toks)
        self.assertIn(",", toks)
        self.assertIn(".", toks)

    def test_preserva_acentuacao(self):
        toks = tokenize("A manutenção é indispensável")
        self.assertIn("manutenção", toks)
        self.assertIn("indispensável", toks)

    def test_clise_com_hifen_nao_quebra(self):
        toks = tokenize("Diz-se que far-lhe-ia bem.", backend="regex")
        self.assertIn("diz-se", toks)
        self.assertIn("far-lhe-ia", toks)

    def test_numeros_e_percentuais(self):
        toks = tokenize("Caiu 12,5% em 2024.", backend="regex")
        self.assertIn("12,5%", toks)
        self.assertIn("2024", toks)

    def test_lowercase_padrao(self):
        self.assertEqual(tokenize("Amazônia"), ["amazônia"])
        self.assertEqual(tokenize("Amazônia", lowercase=False), ["Amazônia"])

    def test_sentencas_multiplas(self):
        self.assertEqual(len(sentence_tokenize("Choveu muito. A rua alagou! E agora?")), 3)

    def test_abreviatura_nao_quebra_sentenca(self):
        sents = _regex_sentences("O Dr. Silva chegou cedo. Ele examinou o paciente.")
        self.assertEqual(len(sents), 2)

    def test_normalize_colapsa_espacos_e_aspas(self):
        self.assertEqual(normalize("  “texto”   com   espaços "), '"texto" com espaços')

    def test_detokenize_aproximado(self):
        self.assertEqual(detokenize(["o", "gato", "dormiu", "."]), "o gato dormiu.")


if __name__ == "__main__":
    unittest.main()
