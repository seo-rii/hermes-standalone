"""Discord's PCM16 volume operation, without the removed audioop extension.

Samples retain audioop's native byte order, saturation and rounding toward
negative infinity. This is the one operation used by discord.py, not a general
audioop replacement or an audio codec.
"""

import math
import struct


def mul(fragment: bytes, width: int, factor: float) -> bytes:
    if width != 2:
        raise ValueError("Discord PCM volume requires 16-bit samples")
    if len(fragment) % 2:
        raise ValueError("PCM fragment does not contain complete samples")
    factor = float(factor)
    if not math.isfinite(factor):
        raise ValueError("PCM volume must be finite")
    output = bytearray(len(fragment))
    for index, (sample,) in enumerate(struct.iter_unpack("=h", fragment)):
        scaled = max(-32768, min(32767, sample * factor))
        struct.pack_into("=h", output, index * 2, math.floor(scaled))
    return bytes(output)
