"""HE-AAC frequency bands and patches from pinned FFmpeg (LGPL-2.1+)."""
from dataclasses import dataclass
import math
from standalone.local_whisper_audio_aac_bits import AacError
from standalone.local_whisper_audio_aac_dsp import f32
from .local_whisper_audio_aac_sbr_tables import SBR_OFFSET


@dataclass(frozen=True)
class Parameters:
    start: int
    stop: int
    crossover: int = 0
    scale: int = 2
    alter: int = 1
    noise: int = 2


def make_bands(start, stop, count):
    if start <= 0 or stop < start or count <= 0:
        raise AacError('Invalid SBR logarithmic band range')
    base = f32(f32(stop / start) ** f32(1 / count))
    product, previous, result = float(start), start, []
    for _ in range(count - 1):
        product = f32(product * base)
        present = round(product)
        result.append(present - previous)
        previous = present
    result.append(stop - previous)
    return result


def _cumulative(start, widths):
    result = [start]
    for width in widths:
        if width <= 0:
            raise AacError('SBR frequency band has no subbands')
        result.append(result[-1] + width)
    return result


class Frequency:
    def __init__(self, sample_rate, parameters, limiter=2):
        self.rate, self.parameters, self.limiter = sample_rate, parameters, limiter
        if not (0 <= parameters.start <= 15 and 0 <= parameters.stop <= 15 and
                0 <= parameters.crossover <= 7 and 0 <= parameters.scale <= 3 and
                0 <= parameters.alter <= 1 and 0 <= parameters.noise <= 3 and 0 <= limiter <= 3):
            raise AacError('Invalid SBR frequency header')
        rows = {16000: 0, 22050: 1, 24000: 2, 32000: 3, 44100: 4, 48000: 4,
                64000: 4, 88200: 5, 96000: 5, 128000: 5, 176400: 5, 192000: 5}
        if sample_rate not in rows:
            raise AacError('Unsupported SBR sample rate')
        base = 3000 if sample_rate < 32000 else 4000 if sample_rate < 64000 else 5000
        start_min = ((base << 7) + (sample_rate >> 1)) // sample_rate
        stop_min = ((base << 8) + (sample_rate >> 1)) // sample_rate
        self.k0 = start_min + SBR_OFFSET[rows[sample_rate]][parameters.start]
        stop = parameters.stop
        self.k2 = stop_min + sum(sorted(make_bands(stop_min, 64, 13))[:stop]) if stop < 14 else self.k0 * (stop - 12)
        self.k2 = min(64, self.k2)
        maximum = 48 if sample_rate <= 32000 else 35 if sample_rate == 44100 else 32
        if self.k2 - self.k0 > maximum:
            raise AacError('Too many SBR QMF subbands')
        self.master = self._master()
        count = len(self.master) - 1
        if not 0 < count <= 48 or parameters.crossover >= count:
            raise AacError('Invalid SBR master band count or crossover')
        self.high = self.master[parameters.crossover:]
        self.kx, self.m = self.high[0], self.high[-1] - self.high[0]
        if self.kx > 32 or self.kx + self.m > 64:
            raise AacError('SBR frequency borders exceed QMF bank')
        parity = (len(self.high) - 1) & 1
        self.low = [self.high[0]] + [self.high[2 * i - parity] for i in range(1, (len(self.high) >> 1) + 1)]
        n_q = max(1, round(f32(parameters.noise * f32(math.log2(f32(self.k2 / self.kx))))))
        if n_q > 5:
            raise AacError('Too many SBR noise bands')
        self.noise, position = [self.low[0]], 0
        for i in range(1, n_q + 1):
            position += (len(self.low) - 1 - position) // (n_q + 1 - i)
            self.noise.append(self.low[position])
        self.patches = self._patches()
        self.limits = self._limits(limiter)

    def _master(self):
        p = self.parameters
        if not p.scale:
            dk = p.alter + 1
            count = ((self.k2 - self.k0 + (dk & 2)) >> dk) << 1
            if not 0 < count <= 48 or p.crossover >= count:
                raise AacError('Invalid linear SBR band count')
            widths = [dk] * count
            difference = self.k2 - self.k0 - count * dk
            if difference < 0:
                widths[0] -= 1
                widths[1] -= difference < -1
            elif difference:
                widths[-1] += 1
            return _cumulative(self.k0, widths)
        half = 7 - p.scale
        split = self.k0 * 2 if 49 * self.k2 > 110 * self.k0 else self.k2
        count = round(f32(half * f32(math.log2(f32(split / self.k0))))) * 2
        first = sorted(make_bands(self.k0, split, count))
        master = _cumulative(self.k0, first)
        if split == self.k2:
            return master
        warp = f32(.76923076923076923077) if p.alter else 1.
        count = round(f32(f32(half * warp) * f32(math.log2(f32(self.k2 / split))))) * 2
        second = make_bands(split, self.k2, count)
        if min(second) < first[-1]:
            second.sort()
            change = min(first[-1] - second[0], (second[-1] - second[0]) >> 1)
            second[0] += change
            second[-1] -= change
        return master + _cumulative(split, sorted(second))[1:]

    def _patches(self):
        msb, usb, patches, last = self.k0, self.kx, [], None
        goal = ((1000 << 11) + (self.rate >> 1)) // self.rate
        k = next((i for i, value in enumerate(self.master) if value >= goal), len(self.master) - 1) if goal < self.kx + self.m else len(self.master) - 1
        while True:
            if last == (k, msb):
                raise AacError('SBR patch construction does not advance')
            last, i = (k, msb), k
            while True:
                if i < 0:
                    raise AacError('Invalid SBR patch source band')
                sb = self.master[i]
                odd = (sb + self.k0) & 1
                i -= 1
                if sb <= self.k0 - 1 + msb - odd:
                    break
            if len(patches) > 5:
                raise AacError('Too many SBR patches')
            size = max(sb - usb, 0)
            if size:
                patches.append((self.k0 - odd - size, size))
                usb = msb = sb
            else:
                msb = self.kx
            if self.master[k] - sb < 3:
                k = len(self.master) - 1
            if sb == self.kx + self.m:
                break
        if len(patches) > 1 and patches[-1][1] < 3:
            patches.pop()
        return patches

    def _limits(self, limiter):
        if not limiter:
            return [self.low[0], self.low[-1]]
        warped = tuple(map(f32, (1.32715174233856803909, 1.18509277094158210129, 1.11987160404675912501)))[limiter - 1]
        borders = [self.kx]
        for _, size in self.patches:
            borders.append(borders[-1] + size)
        values = sorted(self.low + borders[1:-1])
        result = [values[0]]
        for value in values[1:]:
            if value >= f32(result[-1] * warped):
                result.append(value)
            elif value == result[-1] or value not in borders:
                continue
            elif result[-1] not in borders:
                result[-1] = value
            else:
                result.append(value)
        return result
