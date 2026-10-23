"""Stdlib Whisper autoregressive decoder with immutable KV prefixes and cross-attention."""
from dataclasses import dataclass
import itertools
import math

from .local_whisper_tensor import EncodedAudio, WhisperMath, f32, row


@dataclass(frozen=True)
class DecoderState:
    decoder: object
    audio: EncodedAudio
    position: int
    layers: tuple


@dataclass(frozen=True)
class DecoderStep:
    hidden: tuple[tuple[float, ...], ...]
    state: DecoderState
    cross_attention: tuple = ()


class WhisperDecoder:
    def __init__(self, weights, config=None, compute_type="float32", *, int8_saturating_pairs=False):
        self.math = WhisperMath(weights, config, compute_type, int8_saturating_pairs=int8_saturating_pairs)
        self.weights = weights
        self.layers = self.math.layer_count("decoder")
        self.heads = int(self.math.attribute("decoder/num_heads", 8))
        embeddings = weights["decoder/embeddings/weight"]
        positions = weights["decoder/position_encodings/encodings"]
        if len(embeddings.shape) != 2 or len(positions.shape) != 2 or embeddings.shape[1] != positions.shape[1]:
            raise ValueError("Invalid Whisper decoder embedding/position shapes")
        self.vocabulary, self.width, self.max_positions = embeddings.shape[0], embeddings.shape[1], positions.shape[0]
        self.scale = f32(math.sqrt(self.width)) if self.math.attribute("decoder/scale_embeddings", True) else 1.0

    def forward(self, tokens, audio, state=None, *, return_attention=False):
        tokens = tuple(itertools.islice(tokens, self.max_positions + 1))
        if not tokens or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.vocabulary for value in tokens):
            raise ValueError("Whisper decoder needs a bounded nonempty token sequence")
        if (not isinstance(audio, EncodedAudio) or not audio.rows
                or any(len(value) != self.width for value in audio.rows)):
            raise ValueError("Whisper decoder audio has the wrong encoded channel count")
        if state is not None and (state.decoder is not self or state.audio is not audio):
            raise ValueError("Whisper decoder state belongs to another model or audio")
        position = state.position if state is not None else 0
        if position + len(tokens) > self.max_positions:
            raise ValueError("Whisper decoder prompt exceeds its learned position table")
        if state is not None and (len(state.layers) != self.layers
                                  or any(len(layer["self"].get("keys", ())) != position for layer in state.layers)):
            raise ValueError("Whisper decoder state has an inconsistent KV position")
        layers = (tuple({key: dict(value) for key, value in layer.items()} for layer in state.layers)
                  if state is not None else tuple({"self": {}, "cross": {}} for _ in range(self.layers)))
        embedded = self.math.embeddings(tokens, "decoder/embeddings")
        positions = self.weights["decoder/position_encodings/encodings"]
        hidden = tuple(row(f32(value * self.scale) + offset for value, offset in
                           zip(source, positions.row(position + index), strict=True)) for index, source in enumerate(embedded))
        attentions = []
        for index in range(self.layers):
            scope = "decoder/layer_" + str(index)
            hidden, _ = self.math.attention(hidden, scope + "/self_attention", self.heads, cache=layers[index]["self"], causal=True)
            hidden, attention = self.math.attention(hidden, scope + "/attention", self.heads, memory=audio.rows, cache=layers[index]["cross"], return_scores=return_attention)
            if return_attention:
                attentions.append(attention)
            hidden = self.math.feedforward(hidden, scope + "/ffn")
        hidden = self.math.normalize(hidden, "decoder/layer_norm")
        return DecoderStep(hidden, DecoderState(self, audio, position + len(tokens), layers), tuple(attentions))

    def logits(self, step, positions=None):
        if not isinstance(step, DecoderStep) or step.state.decoder is not self:
            raise ValueError("Whisper decoder logits belong to another model")
        if positions is None:
            positions = (len(step.hidden) - 1,)
        positions = tuple(itertools.islice(positions, self.max_positions + 1))
        if len(positions) > self.max_positions or any(isinstance(value, bool) or not isinstance(value, int)
                                                     or not 0 <= value < len(step.hidden) for value in positions):
            raise ValueError("Invalid Whisper decoder logit positions")
        return self.math.dense(tuple(step.hidden[index] for index in positions), "decoder/projection")

    def close(self):
        self.math.close()
