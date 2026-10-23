"""Explicit PCM operations used by the original wake capture/feed loop."""

from standalone.audio_samples import PCM16, as_pcm, from_samples, silence


def frames(data, frame_length):
    if type(frame_length) is not int or not 1 <= frame_length <= 65536:
        raise ValueError("Wake frame length must be between one and 65536 samples")
    block = as_pcm(data)
    raw = block.tobytes()
    step = frame_length * 2
    for offset in range(0, len(raw), step):
        part = raw[offset:offset + step]
        yield PCM16(part + b"\0" * (step - len(part)))


def resample_frame(data, output_length):
    """Original per-frame averaging/interpolation and ties-to-even PCM rounding."""
    if type(output_length) is not int or not 1 <= output_length <= 65536:
        raise ValueError("Wake output frame length must be between one and 65536 samples")
    samples = tuple(as_pcm(data).values())
    if len(samples) == output_length:
        return from_samples(samples)
    if not samples:
        return silence(output_length)
    if len(samples) > output_length:
        edges = [index * len(samples) // output_length for index in range(output_length + 1)]
        values = [sum(samples[start:end]) / (end - start) for start, end in zip(edges, edges[1:])]
    else:
        values = []
        for index in range(output_length):
            position = index * (len(samples) - 1) / (output_length - 1) if output_length > 1 else 0
            left = int(position)
            fraction = position - left
            values.append(samples[left] + fraction * (samples[min(left + 1, len(samples) - 1)] - samples[left]))
    return from_samples(round(value) for value in values)


def peak(data):
    return max((abs(value) for value in as_pcm(data).values()), default=0)
