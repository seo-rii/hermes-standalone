"""Scalar HE-AAC analysis/synthesis QMF from pinned FFmpeg (LGPL-2.1+)."""
from standalone.local_whisper_audio_aac_dsp import f32
from .local_whisper_audio_aac_sbr_transform import imdct
from .local_whisper_audio_aac_sbr_tables import SBR_QMF_WINDOW_DS, SBR_QMF_WINDOW_US


class Analysis:
    def __init__(self):
        self.samples = [0.] * 1312

    def run(self, pcm):
        if len(pcm) != 1024:
            raise ValueError('HE-AAC QMF analysis needs 1024 core samples')
        x = self.samples
        x[:288], x[288:] = x[1024:], list(pcm)
        result = []
        for slot in range(32):
            start = slot * 32
            z = [f32(SBR_QMF_WINDOW_DS[i] * x[start + 319 - i]) for i in range(320)]
            for k in range(64):
                value = z[k]
                for block in range(1, 5):
                    value = f32(value + z[k + 64 * block])
                z[k] = value
            source = [z[0], z[1]]
            for k in range(1, 32):
                source.extend((-z[64 - k], z[k + 1]))
            transformed = imdct(source, scale=-65536.)
            result.append([complex(-transformed[63 - k], transformed[k]) for k in range(32)])
        return result


class Synthesis:
    def __init__(self):
        self.samples = [0.] * 2304
        self.offset = 1152

    def run(self, rows, *, downsample=False):
        if len(rows) < 32 or any(len(row) != 64 for row in rows[:32]):
            raise ValueError('HE-AAC QMF synthesis needs 32 rows of 64 complex bands')
        div = int(bool(downsample))
        width, step = 64 >> div, 128 >> div
        window = SBR_QMF_WINDOW_DS if div else SBR_QMF_WINDOW_US
        output, v = [], self.samples
        for row in rows[:32]:
            if self.offset < step:
                saved = 1152 >> div
                v[2304 - saved:] = v[:saved]
                self.offset = 2304 - saved - step
            else:
                self.offset -= step
            start = self.offset
            if div:
                source = [-value.real for value in row[:32]] + [value.imag for value in row[:32][::-1]]
                z = imdct(source, scale=1 / (64 * 32768))
                for i in range(32):
                    v[start + i] = z[63 - 2 * i]
                    v[start + 63 - i] = -z[62 - 2 * i]
            else:
                real = imdct([value.real for value in row], scale=1 / (64 * 32768))
                imaginary = imdct([(-value.imag if i & 1 else value.imag) for i, value in enumerate(row)], scale=1 / (64 * 32768))
                for i in range(64):
                    v[start + i] = f32(imaginary[i] - real[63 - i])
                    v[start + 127 - i] = f32(imaginary[i] + real[63 - i])
            for i in range(width):
                value = f32(v[start + i] * window[i])
                for band, offset in enumerate((192, 256, 448, 512, 704, 768, 960, 1024, 1216), 1):
                    value = f32(value + f32(v[start + (offset >> div) + i] * window[band * width + i]))
                output.append(value)
        return output
