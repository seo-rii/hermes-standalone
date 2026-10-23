"""CTranslate2 Whisper beam search, sampling, forced prefix and timestamp rules."""
from dataclasses import dataclass
import heapq
import math
import random

from . import local_whisper_float
from .local_whisper_tensor import f32

_DISABLED = -3.4028234663852886e38


@dataclass(frozen=True)
class GenerationResult:
    sequences_ids: list[list[int]]
    scores: list[float]
    no_speech_prob: float = 0.0
    sequences: list[list[str]] | None = None


@dataclass(frozen=True)
class SearchOptions:
    beam_size: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    max_length: int = 448
    sampling_topk: int = 1
    sampling_temperature: float = 1.0
    num_hypotheses: int = 1
    suppress_blank: bool = True
    suppress_tokens: tuple = (-1,)
    max_initial_timestamp_index: int = 50
    return_scores: bool = False
    return_no_speech_prob: bool = False

    def __post_init__(self):
        for name in ("patience", "length_penalty", "repetition_penalty", "sampling_temperature"):
            object.__setattr__(self, name, f32(getattr(self, name)))
        for value, lower, upper in ((self.beam_size, 1, 100), (self.num_hypotheses, 1, 100),
                                    (self.no_repeat_ngram_size, 0, 448), (self.max_length, 1, 448),
                                    (self.sampling_topk, 0, 1000000), (self.max_initial_timestamp_index, 0, 1000000)):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError("Invalid Whisper generation integer option")
        for value in (self.patience, self.repetition_penalty, self.sampling_temperature):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Whisper patience, repetition penalty and sampling temperature must be positive")
        if not math.isfinite(self.length_penalty):
            raise ValueError("Whisper length penalty must be finite")
        if (self.num_hypotheses > math.floor(f32(self.beam_size * self.patience) + 0.5)
                and not (self.beam_size == 1 and self.sampling_topk != 1)):
            raise ValueError("Whisper hypotheses cannot exceed beam_size times patience")


def logsumexp(values):
    maximum = max(values)
    count = len(values) - len(values) % 8 if len(values) > 8 else 0
    exps = tuple(local_whisper_float.exp(f32(value - maximum)) if i < count
                 else f32(math.exp(f32(value - maximum))) for i, value in enumerate(values))
    return f32(f32(math.log(local_whisper_float.total(exps))) + maximum)


def log_softmax(values):
    maximum = max(values)
    count = len(values) - len(values) % 8 if len(values) > 8 else 0
    exps = tuple(local_whisper_float.exp(f32(value - maximum)) if i < count
                 else f32(math.exp(f32(value - maximum))) for i, value in enumerate(values))
    offset = f32(-maximum - f32(math.log(local_whisper_float.total(exps))))
    return tuple(f32(value + offset) for value in values)


def penalize(logits, sequence, options):
    values = list(logits)
    if options.repetition_penalty != 1:
        penalty = f32(options.repetition_penalty)
        for token in set(sequence):
            values[token] = f32(values[token] * penalty if values[token] < 0 else values[token] / penalty)
    size = options.no_repeat_ngram_size
    if size and len(sequence) >= size:
        suffix = sequence[len(sequence) - size + 1:]
        for index in range(len(sequence) - size + 1):
            if sequence[index:index + size - 1] == suffix:
                values[sequence[index + size - 1]] = _DISABLED
    return values


class WhisperSearch:
    def __init__(self, decoder, vocabulary, config):
        self.decoder, self.vocabulary, self.config = decoder, tuple(vocabulary), config
        tokens = {value: index for index, value in enumerate(vocabulary)}
        self.sot, self.eot = tokens['<|startoftranscript|>'], tokens['<|endoftext|>']
        self.no_timestamps = tokens['<|notimestamps|>']
        self.no_speech = tokens.get('<|nospeech|>', tokens.get('<|nocaptions|>', self.eot))
        self.random = random.Random()

    def timestamp_rules(self, values, sequence, step, begin, options):
        timestamp = self.no_timestamps + 1
        values[self.no_timestamps] = _DISABLED
        if step == begin == 0:
            values[:timestamp] = [_DISABLED] * timestamp
            upper = min(len(values), timestamp + options.max_initial_timestamp_index + 1)
            values[upper:] = [_DISABLED] * (len(values) - upper)
        elif step > begin:
            previous = sequence[-1]
            if previous >= timestamp:
                penultimate = sequence[-2] if step - 1 > begin else previous
                if penultimate >= timestamp:
                    values[timestamp:] = [_DISABLED] * (len(values) - timestamp)
                    return
                values[:self.eot] = [_DISABLED] * self.eot
                values[timestamp:previous] = [_DISABLED] * (previous - timestamp)
            else:
                for previous in reversed(sequence[begin:]):
                    if previous >= timestamp:
                        values[timestamp:previous + 1] = [_DISABLED] * (previous - timestamp + 1)
                        break
            probabilities = log_softmax(values)
            if logsumexp(probabilities[timestamp:]) > max(probabilities[:timestamp]):
                values[:timestamp] = [_DISABLED] * timestamp

    def process(self, logits, sequence, step, prefix, options, timestamps):
        values = penalize(logits, sequence, options)
        suppress = set()
        for token in options.suppress_tokens:
            if token == -1:
                suppress.update(self.config.get('suppress_ids', ()))
            elif token >= 0:
                suppress.add(token)
        if options.suppress_blank and step == len(prefix):
            suppress.update(self.config.get('suppress_ids_begin', ()))
        for token in suppress:
            if 0 <= token < len(values):
                values[token] = _DISABLED
        if timestamps:
            self.timestamp_rules(values, sequence, step, len(prefix), options)
        return log_softmax(values)

    def sample(self, candidates, count, options):
        if options.sampling_topk == 1:
            return heapq.nlargest(count, candidates, key=lambda item: (item[0], -item[1], -item[2]))
        pool = heapq.nlargest(options.sampling_topk, candidates) if options.sampling_topk else list(candidates)
        if count == 1:
            inverse = f32(1.0 / options.sampling_temperature)
            weights = local_whisper_float.softmax(tuple(f32(value[0] * inverse) for value in pool))
            return self.random.choices(pool, weights=weights, k=1)
        return heapq.nlargest(count, pool, key=lambda item: item[0] / options.sampling_temperature
                              - math.log(-math.log(max(self.random.random(), 2.0 ** -53))))

    def prompt_state(self, audio, prompt, options):
        try:
            sot = prompt.index(self.sot)
        except ValueError:
            raise ValueError("<|startoftranscript|> token was not found in the prompt") from None
        end = sot
        while end < len(prompt) and self.sot <= prompt[end] <= self.no_timestamps:
            end += 1
        state, no_speech = None, 0.0
        if end > 1:
            initial = self.decoder.forward(prompt[:end - 1], audio)
            state = initial.state
            if options.return_no_speech_prob and sot != end - 1:
                logits = self.decoder.logits(initial, (sot,))[0]
                no_speech = local_whisper_float.softmax(logits)[self.no_speech]
        return state, prompt[end - 1:], no_speech, sot == end - 1

    def generate(self, audio, prompt, options):
        prompt = tuple(prompt)
        if len(prompt) > self.decoder.max_positions or not prompt:
            raise ValueError("Whisper prompt exceeds its learned context")
        state, start, no_speech, speech_at_start = self.prompt_state(audio, prompt, options)
        start_step = state.position if state is not None else 0
        maximum = min(options.max_length // 2, options.max_length - start_step)
        if maximum <= 0:
            raise ValueError("Whisper prompt exceeds the generation maximum length")
        prefix = start[1:]
        timestamps = start[0] != self.no_timestamps
        beams = [((), 0.0, start[0], state)]
        results, top_finished = [], False
        for step in range(maximum):
            candidates, next_states = [], []
            for index, (sequence, score, token, state) in enumerate(beams):
                output = self.decoder.forward((token,), audio, state)
                logits = self.decoder.logits(output)[0]
                next_states.append(output.state)
                if options.return_no_speech_prob and speech_at_start and step == 0:
                    no_speech = local_whisper_float.softmax(logits)[self.no_speech]
                values = self.process(logits, sequence, step, prefix, options, timestamps)
                candidates.extend((f32(score + value), index, word) for word, value in enumerate(values))
            count = options.beam_size * 2 if options.beam_size > 1 else 1
            best = self.sample(candidates, count, options)
            if step < len(prefix):
                best = [(0.0 if rank == 0 else -1e10, 0, prefix[step]) for rank in range(count)]
            elif prefix and step == len(prefix):
                best = [(score, index, word) if rank == 0 or word != self.eot else (-1e10, 0, 0)
                        for rank, (score, index, word) in enumerate(best)]
            candidates = [(beams[index][0] + (word,), score, word, next_states[index]) for score, index, word in best]
            beams, top_finished = self.finish_step(candidates, results, step, maximum, len(prefix), options, top_finished)
            if (step + 1 == maximum or (options.length_penalty == 0 and top_finished and len(results) >= options.num_hypotheses)
                    or (options.length_penalty != 0 and len(results) >= math.floor(f32(options.beam_size * options.patience) + 0.5))):
                break
        finalized = []
        for sequence, score in results:
            denominator = f32(len(sequence) ** options.length_penalty)
            score = f32(score / denominator) if denominator else (float('-inf') if score < 0 else float('nan'))
            finalized.append((score, list(sequence)))
        finalized.sort(key=lambda value: value[0], reverse=True)
        finalized = finalized[:options.num_hypotheses]
        sequences = [value[1] for value in finalized]
        return GenerationResult(sequences, [value[0] for value in finalized] if options.return_scores else [], no_speech,
                                [[self.vocabulary[token] for token in sequence] for sequence in sequences])

    def finish_step(self, candidates, results, step, maximum, prefix_length, options, top_finished):
        active, secondary = [], options.beam_size
        for rank in range(min(options.beam_size, len(candidates))):
            sequence, score, word, state = candidates[rank]
            finished = word == self.eot and step >= prefix_length or step + 1 == maximum
            chosen = candidates[rank]
            if finished:
                if rank == 0:
                    top_finished = True
                results.append((sequence[:-1] if word == self.eot else sequence, score))
                for next_rank in range(secondary, len(candidates)):
                    if candidates[next_rank][2] != self.eot:
                        chosen, secondary = candidates[next_rank], next_rank + 1
                        break
            active.append(chosen)
        return active, top_finished
