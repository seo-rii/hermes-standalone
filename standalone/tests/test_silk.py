"""Independent Skype SDK fixtures; tests do not compile or load native code."""

import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
import zlib

from standalone.silk import Decoder, silk_to_wav
from standalone.silk_entropy import RangeDecoder, SilkError, pulses
from standalone import silk_fixed


class EntropyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors = json.loads((Path(__file__).parent / "fixtures/silk_entropy.json").read_text())

    def test_native_range_encoder_vectors_and_padding(self):
        for row in self.vectors["ranges"]:
            with self.subTest(name=row["name"], row=row["row"]):
                decoder = RangeDecoder(base64.b64decode(row["packet"]))
                values = [decoder.symbol(row["cdf"], len(row["cdf"]) // 2) for _ in row["symbols"]]
                self.assertEqual(values, row["symbols"])
                self.assertEqual(decoder.length()[0], row["bits"])
                decoder.finish()


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors = json.loads((Path(__file__).parent / "fixtures/silk.json").read_text())["vectors"]

    def test_native_pcm_packets_at_all_internal_rates_and_packet_durations(self):
        for row in self.vectors:
            with self.subTest(vector=row["id"]):
                decoder = Decoder(row["output_rate"])
                chunks = []
                for index, packet in enumerate(row["packets"]):
                    lost = index in row["lost_packet_indices"]
                    chunks.append(decoder.decode(None if lost else base64.b64decode(packet)))
                    frames = 1 if lost else decoder.state.nFramesDecoded
                    self.assertEqual(frames, row["frames_per_packet"][index])
                    self.assertFalse(decoder.state.moreInternalDecoderFrames)
                pcm = b"".join(chunks)
                self.assertEqual(len(pcm) // 2, row["pcm_samples"])
                self.assertEqual(hashlib.sha256(pcm).hexdigest(), row["pcm_sha256"])
                self.assertEqual(pcm, zlib.decompress(base64.b64decode(row["pcm_zlib_base64"])))

    def _framing(self, row, *, tencent=False, terminator=False):
        payload = bytearray(b"\x02#!SILK_V3" if tencent else b"#!SILK_V3")
        for value in row["packets"]:
            packet = base64.b64decode(value)
            payload.extend(struct.pack("<h", len(packet)))
            payload.extend(packet)
        if terminator:
            payload.extend(b"\xff\xff")
        return bytes(payload)

    def test_standard_and_tencent_files_produce_native_wav_pcm(self):
        rows = [row for row in self.vectors if row["id"] in (
            "8000-40-noise-24000-normal", "16000-20-chirp-16000-normal",
            "24000-100-silence-48000-normal")]
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "voice.dat", Path(directory) / "voice.wav"
            for row in rows:
                for tencent in (False, True):
                    for terminator in (False, True):
                        with self.subTest(vector=row["id"], tencent=tencent, terminator=terminator):
                            source.write_bytes(self._framing(row, tencent=tencent, terminator=terminator))
                            self.assertEqual(silk_to_wav(source, target, row["output_rate"]), row["pcm_samples"])
                            with wave.open(str(target), "rb") as output:
                                self.assertEqual(output.getparams()[:4], (1, 2, row["output_rate"], row["pcm_samples"]))
                                self.assertEqual(output.readframes(output.getnframes()),
                                                 zlib.decompress(base64.b64decode(row["pcm_zlib_base64"])))
            self.assertFalse(list(Path(directory).glob(".hermes-silk-*")))

    def test_invalid_framing_does_not_replace_an_existing_destination(self):
        row = next(row for row in self.vectors if row["id"] == "8000-20-chirp-8000-normal")
        valid = self._framing(row, terminator=True)
        payloads = (b"", b"#!SILK_V2", b"#!SILK_V3", b"#!SILK_V3\0",
                    b"#!SILK_V3\0\0", b"#!SILK_V3\xfe\xff", b"#!SILK_V3\x01\x04",
                    b"#!SILK_V3\x04\0\0", valid + b"x", valid[:-1])
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "bad.silk", Path(directory) / "voice.wav"
            target.write_bytes(b"previous output")
            for payload in payloads:
                with self.subTest(payload=payload[:32]):
                    source.write_bytes(payload)
                    with self.assertRaises(SilkError):
                        silk_to_wav(source, target)
                    self.assertEqual(target.read_bytes(), b"previous output")
                    self.assertFalse(list(Path(directory).glob(".hermes-silk-*")))
            with self.assertRaises(SilkError):
                silk_to_wav(source, source)
            with source.open("wb") as stream:
                stream.truncate(32 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(SilkError, "32 MiB"):
                silk_to_wav(source, target)

    def test_decoder_rejects_invalid_output_rates(self):
        for rate in (True, 7999, 48001, 24000.0, "24000"):
            with self.assertRaises(SilkError):
                Decoder(rate)

    def test_native_frequency_switches_consecutive_losses_and_arbitrary_api_rates(self):
        vectors = json.loads((Path(__file__).parent / "fixtures/silk_sequences.json").read_text())["vectors"]
        for row in vectors:
            with self.subTest(vector=row["id"]):
                decoder = Decoder(row["api_rates"][0])
                chunks = []
                for index, (rate, packet) in enumerate(zip(row["api_rates"], row["packets"])):
                    decoder.rate = rate
                    chunks.append(decoder.decode(None if packet is None else base64.b64decode(packet)))
                    self.assertEqual(len(chunks[-1]) // 2, row["samples_per_packet"][index])
                    self.assertEqual(1 if packet is None else decoder.state.nFramesDecoded,
                                     row["frames_per_packet"][index])
                pcm = b"".join(chunks)
                self.assertEqual(hashlib.sha256(pcm).hexdigest(), row["pcm_sha256"])
                self.assertEqual(pcm, zlib.decompress(base64.b64decode(row["pcm_zlib_base64"])))


class EntropyPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors = json.loads((Path(__file__).parent / "fixtures/silk_entropy.json").read_text())

    def test_native_fixed_point_macros_and_normalized_division(self):
        vectors = json.loads((Path(__file__).parent / "fixtures/silk_fixed.json").read_text())
        implementations = {"wwb": silk_fixed.smulwb, "www": silk_fixed.smulww,
                           "wrnd": silk_fixed.rshift_round, "wdiv": silk_fixed.div32_varq,
                           "winv": silk_fixed.inverse32_varq}
        for name, rows in vectors["cases"].items():
            for values in rows:
                with self.subTest(function=name, values=values[:-1]):
                    self.assertEqual(implementations[name](*values[:-1]), values[-1])

    def test_native_pulse_encoder_all_frame_sizes_signals_and_offsets(self):
        for row in self.vectors["pulses"]:
            with self.subTest(length=row["length"], signal=row["signal"], offset=row["offset"]):
                decoder = RangeDecoder(base64.b64decode(row["packet"]))
                values, _rate = pulses(decoder, row["length"], row["signal"], row["offset"])
                self.assertEqual(values, row["values"])
                self.assertEqual(decoder.length()[0], row["bits"])
                decoder.finish()

    def test_malformed_entropy_inputs_are_bounded(self):
        for data in (b"", b"\0" * 1025):
            with self.assertRaises(SilkError):
                RangeDecoder(data)
        for data in (b"\xff" * 8, b"\0" * 8):
            decoder = RangeDecoder(data)
            with self.assertRaises(SilkError):
                for _ in range(100):
                    decoder.symbol((0, 32768, 65535), 1)
                decoder.finish()
