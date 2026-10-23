"""Independent Skype SDK frame, concealment, CNG and filter state vectors."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import zlib

from standalone.silk_synthesis import biquad, lpc_synthesis, ma_prediction, sqrt_approx, sum_sqr_shift, synthesis


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) if isinstance(item, dict) else item for key, item in value.items()})


class SDKFrameSynthesisTests(unittest.TestCase):
    def assert_values(self, actual, expected, path):
        if isinstance(expected, dict):
            for key, value in expected.items():
                self.assert_values(getattr(actual, key), value, path + "." + key)
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path + " length")
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_values(left, right, path + "[" + str(index) + "]")
        else:
            self.assertEqual(actual, expected, path)

    def test_independent_sdk_pcm_and_all_dsp_state(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/silk_synthesis.json").read_text())
        raw = zlib.decompress(base64.b64decode(fixture["data_zlib_base64"]))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), fixture["data_sha256"])
        vectors = json.loads(raw)
        self.assertEqual(len(vectors), fixture["case_count"])
        rates, losses, voiced = set(), set(), set()
        for case in vectors:
            state, control = namespace(case["state"]), namespace(case["control"])
            samples = synthesis(state, control, case["pulses"], lost=case["lost"])
            self.assert_values(samples, case["samples"], case["id"] + ".pcm")
            self.assert_values(state, case["after_state"], case["id"] + ".state")
            self.assert_values(control, case["after_control"], case["id"] + ".control")
            rates.add(state.fs_kHz)
            losses.add(case["lost"])
            voiced.add(control.sigtype)
        self.assertEqual(rates, {8, 12, 16, 24})
        self.assertEqual(losses, {False, True})
        self.assertEqual(voiced, {0, 1})

    def test_independent_sdk_helper_extremes_and_alignment(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/silk_synthesis.json").read_text())
        for index, case in enumerate(fixture["helpers"]):
            kind = case["kind"]
            if kind == "sqrt":
                result = sqrt_approx(case["value"])
            elif kind == "energy":
                result = list(sum_sqr_shift(case["samples"], unaligned=case["unaligned"]))
            else:
                history = case["history"][:]
                if kind == "ma":
                    result = ma_prediction(case["samples"], case["coefficients"], history)
                elif kind == "lpc":
                    result = lpc_synthesis(case["samples"], case["coefficients"], case["gain"], history)
                else:
                    result = biquad(case["samples"], case["numerator"], case["denominator"], history)
                self.assert_values(history, case["after"], kind + ".history[" + str(index) + "]")
            self.assert_values(result, case["expected"], kind + "[" + str(index) + "]")


if __name__ == "__main__":
    unittest.main()
