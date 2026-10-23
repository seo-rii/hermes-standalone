"""Qualified stdlib Whisper inference engine reading original CTranslate2 model data."""
from dataclasses import replace
import json
from pathlib import Path
import weakref

from .local_whisper_decoder import WhisperDecoder
from .local_whisper_encoder import EncodedAudio, WhisperEncoder
from .local_whisper_float import softmax
from .local_whisper_search import GenerationResult, SearchOptions, WhisperSearch
from .local_whisper_weights import ModelWeights


def _model_input(path, files):
    memory = None
    if files is not None:
        if not isinstance(files, dict):
            raise TypeError('Whisper files must map file names to bytes or file-like objects')
        memory = {}
        for name, content in files.items():
            if not isinstance(name, str):
                raise TypeError('Whisper in-memory file names must be strings')
            reader = getattr(content, 'read', None)
            content = reader() if reader is not None else content
            if not isinstance(content, bytes):
                raise TypeError('Whisper file content must be file-like or bytes')
            memory[name] = content

    def read(name):
        if memory is not None:
            return memory.get(name)
        try:
            return (path / name).read_bytes()
        except OSError:
            return None

    if memory is not None:
        model = memory.get('model.bin')
        if model is None:
            raise RuntimeError("Unable to open file 'model.bin' in model '" + str(path) + "'")
    else:
        model = path / 'model.bin'
        if not model.is_file():
            raise RuntimeError("Unable to open file 'model.bin' in model '" + str(path) + "'")
    raw = read('config.json')
    config = json.loads(raw) if raw is not None else {}
    raw = read('vocabulary.json')
    if raw is not None:
        vocabulary = tuple(json.loads(raw))
    else:
        raw = read('vocabulary.txt')
        if raw is None:
            raise RuntimeError('Cannot load the vocabulary from the model directory')
        vocabulary = tuple(raw.decode().splitlines())
    return config, vocabulary, model


class WhisperEngine:
    def __init__(self, model_path, device='auto', device_index=0, compute_type='auto',
                 cpu_threads=0, num_workers=1, files=None, **kwargs):
        if kwargs:
            raise TypeError('Unexpected Whisper engine options: ' + ', '.join(sorted(kwargs)))
        if device not in ('cpu','auto'):
            raise ValueError('The stdlib Whisper engine executes the configured model on the CPU')
        if isinstance(compute_type, dict):
            compute_type = compute_type.get(device, 'default')
        if compute_type not in ('auto', 'default', 'float32', 'int8', 'int8_float32'):
            raise ValueError('Whisper CPU compute type must be auto, default, float32, int8, or int8_float32')
        self.device = 'cpu'
        self.device_index = list(device_index) if isinstance(device_index, (list, tuple)) else [device_index]
        self.path = Path(model_path)
        self.config, self.vocabulary, model = _model_input(self.path, files)
        try:
            self.weights = ModelWeights(model)
        except OSError as exc:
            raise RuntimeError("Unable to open file 'model.bin' in model '" + str(self.path) + "'") from exc
        self._finalizer = weakref.finalize(self, self.weights.close)
        selected = ('float32' if self.weights['decoder/embeddings/weight'].dtype in ('float32', 'float16', 'bfloat16')
                    else 'int8_float32') if compute_type == 'default' else ('int8_float32' if compute_type in ('auto', 'int8') else compute_type)
        self.compute_type = selected
        self.encoder = WhisperEncoder(self.weights,self.config,selected,int8_saturating_pairs=selected=='int8_float32')
        self.decoder = WhisperDecoder(self.weights,self.config,selected,int8_saturating_pairs=selected=='int8_float32')
        if len(self.vocabulary) != self.decoder.vocabulary:
            self.close()
            raise ValueError('Whisper vocabulary does not match the model embeddings')
        self.search = WhisperSearch(self.decoder,self.vocabulary,self.config)
        self.is_multilingual = '' in self.vocabulary
        self.n_mels = self.encoder.n_mels
        self.num_languages = self.search.no_speech - self.search.sot - 5

    def encode(self, features, to_cpu=False):
        if isinstance(features, EncodedAudio):
            return features
        return self.encoder.encode(features)

    def generate(self, features, prompts, **kwargs):
        audio = self.encode(features)
        prompts = tuple(prompts)
        if len(prompts) != 1:
            raise ValueError('An EncodedAudio instance describes one Whisper batch item')
        options = SearchOptions(**kwargs)
        if options.beam_size == 1 and options.num_hypotheses > 1:
            single = replace(options,num_hypotheses=1,return_scores=True)
            results = [self.search.generate(audio,prompts[0],single) for _ in range(options.num_hypotheses)]
            results.sort(key=lambda value:value.scores[0],reverse=True)
            return [GenerationResult([value.sequences_ids[0] for value in results],
                                     [value.scores[0] for value in results] if options.return_scores else [],
                                     results[0].no_speech_prob,[value.sequences[0] for value in results])]
        return [self.search.generate(audio,prompts[0],options)]

    def detect_language(self, features):
        if not self.is_multilingual:
            raise RuntimeError('detect_language can only be called on multilingual models')
        audio=self.encode(features)
        output=self.decoder.forward((self.search.sot,),audio)
        logits=self.decoder.logits(output)[0]
        ids=self.config['lang_ids']
        probabilities=softmax(tuple(logits[index] for index in ids))
        return [sorted(((self.vocabulary[index],probability) for index,probability in zip(ids,probabilities,strict=True)),
                       key=lambda item:item[1],reverse=True)]

    def align(self, features, start_sequence, text_tokens, num_frames, median_filter_width=7):
        from .local_whisper_align import align
        if isinstance(num_frames, (tuple, list)):
            if len(num_frames) != 1:
                raise ValueError('Whisper alignment frame counts need one value per audio item')
            num_frames = num_frames[0]
        return align(self.decoder, self.encode(features), start_sequence, text_tokens, num_frames,
                     self.config, self.search.no_timestamps, self.search.eot, median_filter_width)

    def close(self):
        self.encoder.close()
        self.decoder.close()
        self._finalizer()
