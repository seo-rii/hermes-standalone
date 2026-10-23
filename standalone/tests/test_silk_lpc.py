"""SILK LPC math matches independently executed, pinned SDK C vectors."""

import json
from pathlib import Path
import unittest

from standalone import silk_lpc


class SilkLPCOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = json.loads((Path(__file__).with_name("fixtures") / "silk-lpc-oracle.json").read_text())

    def test_nlsf_stabilization_disordered_clusters_borders_and_fallback(self):
        for index, case in enumerate(self.oracle["stabilize"]):
            with self.subTest(index=index, order=len(case["input"])):
                values = list(case["input"])
                self.assertIs(silk_lpc.nlsf_stabilize(values, case["deltas"]), values)
                self.assertEqual(values, case["expected"])

    def test_nlsf_polynomial_conversion_and_twenty_step_stable_filter(self):
        for index, case in enumerate(self.oracle["nlsf"]):
            with self.subTest(index=index, order=len(case["input"])):
                self.assertEqual(silk_lpc.nlsf2a(case["input"]), case["expected"])
                self.assertEqual(silk_lpc.nlsf2a_stable(case["input"]), case["stable"])

    def test_rounded_int16_chirp_bandwidth_expansion(self):
        for index, case in enumerate(self.oracle["bandwidth16"]):
            with self.subTest(index=index):
                values = list(case["input"])
                self.assertIs(silk_lpc.bwexpander(values, case["chirp"]), values)
                self.assertEqual(values, case["expected"])

    def test_full_word_int32_chirp_bandwidth_expansion(self):
        for index, case in enumerate(self.oracle["bandwidth32"]):
            with self.subTest(index=index):
                values = list(case["input"])
                self.assertIs(silk_lpc.bwexpander32(values, case["chirp"]), values)
                self.assertEqual(values, case["expected"])

    def test_inverse_prediction_gain_q12_q24_and_early_instability(self):
        for index, case in enumerate(self.oracle["inverse_gain"]):
            with self.subTest(index=index, q=case["q"]):
                actual = silk_lpc.lpc_inverse_pred_gain(case["input"], q=case["q"])
                self.assertEqual(list(actual), case["expected"])


if __name__ == "__main__":
    unittest.main()
