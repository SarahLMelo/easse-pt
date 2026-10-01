"""Testes do corpus, dos prompts e do pipeline ponta a ponta."""

import json
import tempfile
import unittest
from pathlib import Path

from easse_pt.corpus import Dataset, load_sample
from easse_pt.llm import DummyProvider, get_provider
from easse_pt.llm.base import LLMResponse
from easse_pt.llm.dummy import _extract_sentence
from easse_pt.pipeline import evaluate_references, evaluate_run, run_generation, write_report
from easse_pt.prompts import FINAL_MARKER, build_prompt, postprocess


class TestCorpus(unittest.TestCase):
    def test_amostra_carrega_com_dois_niveis(self):
        ds = load_sample()
        self.assertEqual(len(ds), 20)
        self.assertEqual(ds.available_levels(), ["natural", "forte"])
        self.assertEqual(len(ds.refs_sents()), 2)

    def test_referencias_sao_mais_faceis_que_o_original(self):
        st = load_sample().stats()
        self.assertGreater(st["flesch_natural"], st["flesch_orig"])
        self.assertGreater(st["flesch_forte"], st["flesch_natural"])

    def test_roundtrip_jsonl(self):
        ds = load_sample()
        with tempfile.TemporaryDirectory() as td:
            p = ds.to_jsonl(Path(td) / "ds.jsonl")
            self.assertEqual(Dataset.from_jsonl(p).orig_sents, ds.orig_sents)

    def test_exporta_arquivos_easse(self):
        ds = load_sample()
        with tempfile.TemporaryDirectory() as td:
            files = ds.to_easse_files(td)
            self.assertEqual(set(files), {"orig", "natural", "forte"})
            linhas = files["orig"].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(linhas), len(ds))

    def test_subset_reprodutivel(self):
        ds = load_sample()
        self.assertEqual(ds.subset(5).ids, ds.subset(5).ids)
        self.assertEqual(len(ds.subset(5)), 5)


class TestPrompts(unittest.TestCase):
    def test_prompt_contem_sentenca_e_marcador(self):
        for strategy in ("zero_shot", "cot"):
            for level in ("natural", "forte"):
                with self.subTest(strategy=strategy, level=level):
                    p = build_prompt("O gato dormiu.", strategy=strategy, level=level)
                    self.assertIn("O gato dormiu.", p.text)
                    self.assertIn(FINAL_MARKER, p.text)

    def test_rejeita_estrategia_invalida(self):
        with self.assertRaises(ValueError):
            build_prompt("x", strategy="inexistente")

    def test_extrai_apos_marcador(self):
        raw = "Passo 1 - bla\nPasso 2 - bla\n\nSIMPLIFICADA: O gato dormiu."
        self.assertEqual(postprocess(raw, "cot"), "O gato dormiu.")

    def test_remove_bloco_think(self):
        raw = "<think>raciocinio longo</think>\nSIMPLIFICADA: A casa é azul."
        self.assertEqual(postprocess(raw, "cot"), "A casa é azul.")

    def test_remove_cortesia_em_zero_shot(self):
        self.assertEqual(postprocess("Claro! Aqui está:\nO gato dormiu.", "zero_shot"),
                         "O gato dormiu.")

    def test_remove_prefixo_rotulo(self):
        self.assertEqual(postprocess("Versão simplificada: O gato dormiu.", "zero_shot"),
                         "O gato dormiu.")

    def test_remove_negrito_markdown_do_marcador(self):
        # formato real devolvido pelo llama3.1:8b no teste de fumaca
        raw = "Passo 3...\n\n**Simplificada:**\nA empresa vai contratar 200 pessoas."
        self.assertEqual(postprocess(raw, "cot"), "A empresa vai contratar 200 pessoas.")

    def test_vazio(self):
        self.assertEqual(postprocess("", "cot"), "")


class _Copia(DummyProvider):
    """Baseline que devolve a sentenca de entrada sem alteracao."""

    def generate(self, prompt, system=None):
        return LLMResponse(text=_extract_sentence(prompt))


class TestPipeline(unittest.TestCase):
    def test_end_to_end_com_dummy(self):
        ds = load_sample().subset(6)
        with tempfile.TemporaryDirectory() as td:
            path = run_generation(ds, DummyProvider(), strategy="zero_shot", level="forte",
                                  out_dir=td, progress=False)
            recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(recs), 6)
            self.assertTrue(all(r["sys"] for r in recs))
            self.assertTrue(path.with_suffix(".meta.json").exists())

            scores = evaluate_run(path, ds, semantic=False)
            self.assertTrue(0 <= scores["sari"] <= 100)
            self.assertTrue(0 <= scores["bleu"] <= 100)
            self.assertEqual(scores["n"], 6)
            self.assertTrue((Path(td) / f"{path.stem}.scores.json").exists())
            self.assertTrue((Path(td) / f"{path.stem}.per_sent.csv").exists())

            paths = write_report(td, td)
            md = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("SARI", md)
            self.assertIn("dummy", md)

    def test_geracao_retomavel(self):
        ds = load_sample().subset(4)
        with tempfile.TemporaryDirectory() as td:
            p1 = run_generation(ds, DummyProvider(), out_dir=td, progress=False)
            p2 = run_generation(ds, DummyProvider(), out_dir=td, progress=False)
            self.assertEqual(p1, p2)
            recs = [json.loads(l) for l in p2.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(recs), 4)  # nao duplicou

    def test_referencia_humana_supera_baseline_de_copia(self):
        ds = load_sample()
        refs = evaluate_references(ds)
        self.assertEqual(set(refs), {"natural", "forte"})
        with tempfile.TemporaryDirectory() as td:
            path = run_generation(ds, _Copia(model="copia"), out_dir=td, progress=False)
            copia = evaluate_run(path, ds, semantic=False)
        self.assertGreater(refs["natural"]["sari"], copia["sari"])
        self.assertGreater(refs["forte"]["sari"], copia["sari"])

    def test_get_provider_resolve_spec(self):
        p = get_provider("dummy/rule-based")
        self.assertEqual(p.name, "dummy")
        self.assertEqual(p.model, "rule-based")
        with self.assertRaises(ValueError):
            get_provider("inexistente/modelo")


if __name__ == "__main__":
    unittest.main()
