"""CTranslate2 Whisper selected-head median and dynamic-time-warping word alignment."""
from dataclasses import dataclass
import math

from .local_whisper_float import f32, fma32, softmax


@dataclass(frozen=True)
class AlignmentResult:
    text_token_probs: list[float]
    alignments: list[tuple[int, int]]


def normalized_columns(rows):
    columns = tuple(zip(*rows, strict=True))
    output = [[] for _ in rows]
    inverse_size = f32(1.0 / len(rows))
    for column in columns:
        total, squares = 0.0, 0.0
        for value in column:
            total = f32(total + value)
            squares = fma32(value, value, squares)
        mean = f32(total * inverse_size)
        variance = max(fma32(squares, inverse_size, -f32(mean * mean)), 0.0)
        inverse = f32(1.0 / f32(math.sqrt(variance))) if variance else math.inf
        for row, value in zip(output, column, strict=True):
            row.append(f32(f32(value - mean) * inverse))
    return tuple(tuple(row) for row in output)


def median_filter(row, width):
    radius, length = width // 2, len(row)
    if length <= radius:
        return row
    output = []
    for position in range(length):
        window = []
        for offset in range(-radius, radius + 1):
            index = abs(position + offset)
            if index >= length:
                index = length - (index - length) - 2
            window.append(row[index])
        output.append(sorted(window)[radius])
    return tuple(output)


def negative_dtw(rows):
    height, width = len(rows), len(rows[0]) if rows else 0
    costs = [[math.inf] * (width + 1) for _ in range(height + 1)]
    traces = [[-1] * (width + 1) for _ in range(height + 1)]
    costs[0][0] = 0.0
    for column in range(1, width + 1):
        for row in range(1, height + 1):
            diagonal, up, left = costs[row - 1][column - 1], costs[row - 1][column], costs[row][column - 1]
            if diagonal < up and diagonal < left:
                previous, trace = diagonal, 0
            elif up < diagonal and up < left:
                previous, trace = up, 1
            else:
                previous, trace = left, 2
            costs[row][column] = f32(-rows[row - 1][column - 1] + previous)
            traces[row][column] = trace
    traces[0] = [2] * (width + 1)
    for row in traces:
        row[0] = 1
    row, column, path = height, width, []
    while row > 0 or column > 0:
        path.append((row - 1, column - 1))
        trace = traces[row][column]
        if trace == 0:
            row, column = row - 1, column - 1
        elif trace == 1:
            row -= 1
        else:
            column -= 1
    return list(reversed(path))


def align(decoder, audio, start_sequence, text_tokens, num_frames, config, no_timestamps, eot, median_filter_width=7):
    if median_filter_width < 1 or median_filter_width % 2 == 0:
        raise ValueError('Whisper alignment median width must be a positive odd number')
    heads = config.get('alignment_heads')
    if not heads:
        raise RuntimeError("The model configuration does not contain alignment_heads")
    if len(text_tokens) != 1:
        raise ValueError('An EncodedAudio instance describes one Whisper batch item')
    tokens = tuple(text_tokens[0])
    sequence = tuple(start_sequence) + (no_timestamps,) + tokens + (eot,)
    output = decoder.forward(sequence, audio, return_attention=True)
    offset = len(start_sequence)
    logits = decoder.logits(output, range(offset, offset + len(tokens)))
    probabilities = [softmax(values[:eot])[token] if 0 <= token < eot else 0.0 for token, values in zip(tokens, logits, strict=True)]
    frames = min(len(audio.rows), num_frames // 2)
    if not frames:
        return [AlignmentResult(probabilities, [])]
    selected = []
    for layer, head in sorted(heads, key=lambda value: value[0]):
        raw = tuple(step[head][:frames] for step in output.cross_attention[layer])
        selected.append(normalized_columns(tuple(softmax(values) for values in raw)))
    filtered = [tuple(median_filter(row, median_filter_width) for row in matrix) for matrix in selected]
    averaged = []
    for position in range(len(sequence)):
        row = []
        for frame in range(frames):
            total = 0.0
            for matrix in filtered:
                total = f32(total + matrix[position][frame])
            row.append(f32(total / len(filtered)))
        averaged.append(tuple(row))
    path = negative_dtw(averaged[offset:offset + len(tokens) + 1])
    return [AlignmentResult(probabilities, path)]
