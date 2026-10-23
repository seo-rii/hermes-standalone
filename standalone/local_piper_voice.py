"""Original Piper1.8.0 voice/configuration/audio-chunk API with owned VITS inference.

Adapted from OHF-Voice/piper1-gpl, commit639388b6317fc4731e91d53da42aea68fd4166ff,
GPL-3.0-or-later. See local-piper-voice-provenance.json and licenses/. All runtime
math, graph execution, phonemization and WAV serialization use the standard library.
"""
from array import array
from dataclasses import dataclass
import itertools
import importlib
import json
import logging
from pathlib import Path
import re
import sys
import threading
import unicodedata
from .local_piper_config import PhonemeType, PiperConfig, SynthesisConfig
from .local_piper_const import BOS, EOS, PAD
from standalone.local_piper_phoneme import ESPEAK_DATA_DIR
from standalone.local_piper_model import PiperModel
from .local_piper_phoneme_ids import phonemes_to_ids
from standalone.local_piper_tensor import Tensor, f32
_LOGGER = logging.getLogger(__name__)
_DEFAULT_SYNTHESIS_CONFIG = SynthesisConfig()
_PHONEME_BLOCK_PATTERN = re.compile('(\\[\\[.*?\\]\\])')
_PHONEMIZER = None
_PHONEMIZER_LOCK = threading.Lock()
_CACHED_PHONEMIZERS = {PhonemeType.HEBREW: ('standalone.local_piper_phonemize_hebrew', 'HebrewPhonemizer', '_hebrew_phonemizer', None), PhonemeType.THAI: ('standalone.local_piper_thai', 'ThaiPhonemizer', '_thai_phonemizer', None)}
_CACHED_PHONEMIZERS[PhonemeType.JAPANESE] = (
    'standalone.local_piper_japanese', 'JapanesePhonemizer', '_japanese_phonemizer', None,
)
_CACHED_PHONEMIZERS[PhonemeType.PINYIN] = (
    'standalone.local_piper_chinese', 'ChinesePhonemizer', '_chinese_phonemizer', 'g2pW',
)

@dataclass
class PhonemeAlignment:
    phoneme: str
    phoneme_ids: object
    num_samples: int

@dataclass
class AudioChunk:
    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_float_array: object
    phonemes: list
    phoneme_ids: list
    phoneme_id_samples: object = None
    phoneme_alignments: object = None
    _audio_int16_array: object = None
    _audio_int16_bytes: object = None
    _phoneme_alignments: object = None

    @property
    def audio_int16_array(self):
        if self._audio_int16_array is None:
            self._audio_int16_array = array('h', (int(max(-32767.0, min(32767.0, f32(v * 32767.0)))) for v in self.audio_float_array))
        return self._audio_int16_array

    @property
    def audio_int16_bytes(self):
        values = self.audio_int16_array
        if sys.byteorder != 'little':
            values = array('h', values)
            values.byteswap()
        return values.tobytes()

@dataclass
class PiperVoice:
    session: object
    config: PiperConfig
    espeak_data_dir: object = ESPEAK_DATA_DIR
    download_dir: Path = Path.cwd()
    use_tashkeel: bool = True
    tashkeel_diacritizier: object = None
    taskeen_threshold: object = 0.8

    @staticmethod
    def load(model_path, config_path=None, use_cuda=False, espeak_data_dir=ESPEAK_DATA_DIR, download_dir=None, include_alignments=False):
        if use_cuda:
            raise ValueError('CUDA execution is unavailable in the standard-library Piper runtime')
        if config_path is None:
            config_path = f'{model_path}.json'
        with open(config_path, encoding='utf-8') as source:
            config = PiperConfig.from_dict(json.load(source))
        model = PiperModel(model_path)
        if include_alignments:
            names = {name for node in model.nodes if node['op'] == 'Ceil' for name in node['outputs']}
            if len(names) == 1 and (not names.intersection(model.outputs)):
                model.outputs.extend(names)
            elif len(names) != 1:
                _LOGGER.debug('Piper alignment output could not be autodetected')
        return PiperVoice(model, config, Path(espeak_data_dir), Path.cwd() if download_dir is None else Path(download_dir))

    def phonemize(self, text):
        if self.config.phoneme_type == PhonemeType.TEXT:
            return [list(unicodedata.normalize('NFD', text))]
        backend = _CACHED_PHONEMIZERS.get(self.config.phoneme_type)
        if backend:
            module_name, class_name, attribute, resource = backend
            phonemizer = getattr(self, attribute, None)
            if phonemizer is None:
                arguments = () if resource is None else (self.download_dir / resource,)
                phonemizer = getattr(importlib.import_module(module_name), class_name)(*arguments)
                setattr(self, attribute, phonemizer)
            return phonemizer.phonemize(text)
        if self.config.phoneme_type != PhonemeType.ESPEAK:
            raise NotImplementedError('Qualified Piper phonemizer is not available for ' + str(self.config.phoneme_type))
        return self._phonemize_espeak(text)

    @staticmethod
    def _append_raw_phonemes(sentences, parts, index, part):
        if not sentences:
            sentences.append([])
        if index > 0 and parts[index - 1].endswith(' '):
            sentences[-1].append(' ')
        sentences[-1].extend(part[2:-2].strip())
        if index < len(parts) - 1 and parts[index + 1].startswith(' '):
            sentences[-1].append(' ')

    def _append_espeak_text_part(self, sentences, part, previous_raw):
        global _PHONEMIZER
        if self.config.espeak_voice == 'ar' and self.use_tashkeel:
            if self.tashkeel_diacritizier is None:
                from standalone.local_piper_arabic import TashkeelDiacritizer
                self.tashkeel_diacritizier = TashkeelDiacritizer()
            part = self.tashkeel_diacritizier(part, taskeen_threshold=self.taskeen_threshold)
        with _PHONEMIZER_LOCK:
            if _PHONEMIZER is None:
                from standalone.local_piper_phoneme import EspeakPhonemizer
                _PHONEMIZER = EspeakPhonemizer(self.espeak_data_dir)
            phonemes = _PHONEMIZER.phonemize(self.config.espeak_voice, part, vowel_clusters=self.config.vowel_clusters)
            if previous_raw and phonemes:
                sentences[-1].extend(phonemes[0])
                phonemes = phonemes[1:]
            sentences.extend(phonemes)

    def _phonemize_espeak(self, text):
        sentences = []
        parts = _PHONEME_BLOCK_PATTERN.split(text)
        previous_raw = False
        for index, part in enumerate(parts):
            if part.startswith('[['):
                previous_raw = True
                self._append_raw_phonemes(sentences, parts, index, part)
                continue
            self._append_espeak_text_part(sentences, part, previous_raw)
            previous_raw = False
        if sentences and (not sentences[-1]):
            sentences.pop()
        return sentences

    def phonemes_to_ids(self, phonemes):
        if self.config.phoneme_type == PhonemeType.PINYIN:
            from .local_piper_chinese import phonemes_to_ids as chinese_ids
            return chinese_ids(phonemes, self.config.phoneme_id_map)
        return phonemes_to_ids(phonemes, self.config.phoneme_id_map)

    def phoneme_ids_to_audio(self, phoneme_ids, syn_config=None, include_alignments=False):
        syn_config = _DEFAULT_SYNTHESIS_CONFIG if syn_config is None else syn_config
        length = self.config.length_scale if syn_config.length_scale is None else syn_config.length_scale
        noise = self.config.noise_scale if syn_config.noise_scale is None else syn_config.noise_scale
        noise_w = self.config.noise_w_scale if syn_config.noise_w_scale is None else syn_config.noise_w_scale
        args = {'input': Tensor.make((1, len(phoneme_ids)), phoneme_ids, 7), 'input_lengths': Tensor.make((1,), [len(phoneme_ids)], 7), 'scales': Tensor.make((3,), [noise, length, noise_w])}
        speaker = syn_config.speaker_id
        if self.config.num_speakers <= 1:
            speaker = None
        elif speaker is None:
            speaker = self.config.default_speaker_id
        if speaker is not None:
            args['sid'] = Tensor.make((1,), [speaker], 7)
        result = self.session.run(args)
        audio = array('f', result[0].values)
        if not include_alignments:
            return audio
        if len(result) == 1:
            return (audio, None)
        return (audio, [int(f32(v * self.config.hop_length)) for v in result[1].values])

    def synthesize(self, text, syn_config=None, include_alignments=False):
        syn_config = _DEFAULT_SYNTHESIS_CONFIG if syn_config is None else syn_config
        for phonemes in self.phonemize(text):
            if not phonemes:
                continue
            ids = self.phonemes_to_ids(phonemes)
            result = self.phoneme_ids_to_audio(ids, syn_config, include_alignments)
            audio, samples = result if isinstance(result, tuple) else (result, None)
            if syn_config.normalize_audio:
                maximum = max(map(abs, audio))
                audio = array('f', (0.0 for _ in audio)) if maximum < 1e-08 else array('f', (v / maximum for v in audio))
            if syn_config.volume != 1.0:
                audio = array('f', (v * syn_config.volume for v in audio))
            audio = array('f', (max(-1.0, min(1.0, v)) for v in audio))
            alignments = None
            if samples is not None and len(samples) == len(ids):
                alignments = self._align_phonemes(phonemes, ids, samples)
            yield AudioChunk(self.config.sample_rate, 2, 1, audio, phonemes, ids, samples, alignments)

    def _align_phonemes(self, phonemes, ids, samples):
        pad = self.config.phoneme_id_map.get(PAD, [])
        position = 0
        alignments = []
        failed = False
        for phoneme in itertools.chain([BOS], phonemes, [EOS]):
            expected = list(self.config.phoneme_id_map.get(phoneme, []))
            if phoneme != EOS:
                expected.extend(pad)
            start = position
            for number in expected:
                if position >= len(ids) or number != ids[position]:
                    failed = True
                    break
                position += 1
            if failed:
                break
            alignments.append(PhonemeAlignment(phoneme, expected, sum(samples[start:position])))
        return None if failed else alignments

    def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True, include_alignments=False):
        alignments = []
        first = True
        for chunk in self.synthesize(text, syn_config, include_alignments):
            if first:
                if set_wav_format:
                    wav_file.setframerate(chunk.sample_rate)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setnchannels(chunk.sample_channels)
                first = False
            wav_file.writeframes(chunk.audio_int16_bytes)
            if include_alignments and chunk.phoneme_alignments:
                alignments.extend(chunk.phoneme_alignments)
        return alignments if include_alignments else None
