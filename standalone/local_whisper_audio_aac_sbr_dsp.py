"""HE-AAC high-frequency reconstruction from pinned FFmpeg (LGPL-2.1+)."""
import math
from standalone.local_whisper_audio_aac_bits import AacError
from standalone.local_whisper_audio_aac_dsp import f32 as f
from .local_whisper_audio_aac_sbr_qmf import Analysis, Synthesis
from .local_whisper_audio_aac_sbr_tables import NOISE


def _square(values, start, stop):
    real = imaginary = 0.
    for value in values[start:stop]:
        real = f(real + f(value.real * value.real))
        imaginary = f(imaginary + f(value.imag * value.imag))
    return f(real + imaginary)


def _correlate(x):
    def real(a, b):
        return f(f(a.real * b.real) + f(a.imag * b.imag))
    def imag(a, b):
        return f(f(a.real * b.imag) - f(a.imag * b.real))
    r2, i2 = real(x[0], x[2]), imag(x[0], x[2])
    r1 = i1 = r0 = 0.
    for i in range(1, 38):
        r0 = f(r0 + real(x[i], x[i]))
        r1 = f(r1 + real(x[i], x[i + 1]))
        i1 = f(i1 + imag(x[i], x[i + 1]))
        r2 = f(r2 + real(x[i], x[i + 2]))
        i2 = f(i2 + imag(x[i], x[i + 2]))
    def append_real(total, a, b):
        return f(f(total + f(a.real * b.real)) + f(a.imag * b.imag))
    def append_imag(total, a, b):
        return f(f(total + f(a.real * b.imag)) - f(a.imag * b.real))
    return (complex(append_real(r1, x[38], x[39]), append_imag(i1, x[38], x[39])),
            complex(r2, i2), append_real(r0, x[38], x[38]),
            complex(append_real(r1, x[0], x[1]), append_imag(i1, x[0], x[1])),
            append_real(r0, x[0], x[0]))


def inverse_filter(low, count):
    output = []
    for k in range(count):
        p00, p01, p10, p11, p21 = _correlate(low[k])
        dk = f(f(p21 * p10) - f(f(f(p11.real * p11.real) + f(p11.imag * p11.imag)) / f(1.000001)))
        if dk:
            real = f(f(f(p00.real * p11.real) - f(p00.imag * p11.imag)) - f(p01.real * p10))
            imaginary = f(f(f(p00.real * p11.imag) + f(p00.imag * p11.real)) - f(p01.imag * p10))
            a1 = complex(f(real / dk), f(imaginary / dk))
        else:
            a1 = 0j
        if p10:
            real = f(f(p00.real + f(a1.real * p11.real)) + f(a1.imag * p11.imag))
            imaginary = f(f(p00.imag + f(a1.imag * p11.real)) - f(a1.real * p11.imag))
            a0 = complex(f(-real / p10), f(-imaginary / p10))
        else:
            a0 = 0j
        if f(f(a1.real * a1.real) + f(a1.imag * a1.imag)) >= 16 or f(f(a0.real * a0.real) + f(a0.imag * a0.imag)) >= 16:
            a0 = a1 = 0j
        output.append((a0, a1))
    return output


def _exp2(value):
    return math.inf if value >= 128 else 0. if value < -149 else 2. ** value


def dequant(sbr, kind):
    envelopes = [[[0.] * 48 for _ in range(9)] for _ in range(2)]
    noises = [[[0.] * 5 for _ in range(3)] for _ in range(2)]
    freq, channels = sbr.frequency, sbr.channels
    if kind == 1 and sbr.coupling:
        a, b = channels
        offset = 12 if a.amp_res else 24
        for e in range(1, a.count + 1):
            n = len(freq.high if a.freq_res[e] else freq.low) - 1
            for k in range(n):
                exponent = a.env_q[e][k]
                pan = offset - b.env_q[e][k]
                temp1 = _exp2(exponent + 7) if a.amp_res else f(_exp2((exponent >> 1) + 7) * (math.sqrt(2) if exponent & 1 else 1))
                temp2 = _exp2(pan) if a.amp_res else f(_exp2(pan >> 1) * (math.sqrt(2) if pan & 1 else 1))
                if temp1 > 1e20:
                    temp1 = 1.
                factor = f(temp1 / f(1 + temp2))
                envelopes[0][e][k], envelopes[1][e][k] = factor, f(factor * temp2)
        for e in range(1, a.noise_count + 1):
            for k in range(len(freq.noise) - 1):
                temp1, temp2 = _exp2(7 - a.noise_q[e][k]), _exp2(12 - b.noise_q[e][k])
                factor = f(temp1 / f(1 + temp2))
                noises[0][e][k], noises[1][e][k] = factor, f(factor * temp2)
    else:
        for ch in range(2 if kind == 1 else 1):
            data = channels[ch]
            for e in range(1, data.count + 1):
                for k in range(len(freq.high if data.freq_res[e] else freq.low) - 1):
                    exponent = data.env_q[e][k]
                    value = _exp2(exponent + 6) if data.amp_res else f(_exp2((exponent >> 1) + 6) * (math.sqrt(2) if exponent & 1 else 1))
                    envelopes[ch][e][k] = 1. if value > 1e20 else value
            for e in range(1, data.noise_count + 1):
                for k in range(len(freq.noise) - 1):
                    noises[ch][e][k] = _exp2(6 - data.noise_q[e][k])
    return envelopes, noises


def _chirp(data, state, count):
    table = tuple(map(f, (0., .75, .9, .98)))
    for i in range(count):
        value = f(.6) if data.invf[0][i] + data.invf[1][i] == 1 else table[data.invf[0][i]]
        value = f(f(.75 * value) + f(.25 * state[i])) if value < state[i] else f(f(.90625 * value) + f(.09375 * state[i]))
        state[i] = 0. if value < .015625 else value


def _generate(low, alphas, bandwidth, data, freq):
    high = [[0j] * 40 for _ in range(64)]
    k, g = freq.kx, 0
    for start, size in freq.patches:
        for offset in range(size):
            p = start + offset
            while g <= len(freq.noise) - 1 and k >= freq.noise[g]:
                g += 1
            g -= 1
            if g < 0:
                raise AacError('Missing SBR noise band')
            a0, a1 = alphas[p]
            bw = bandwidth[g]
            a = (f(f(a1.real * bw) * bw), f(f(a1.imag * bw) * bw), f(a0.real * bw), f(a0.imag * bw))
            for i in range(data.time[0] * 2 + 2, data.time[data.count] * 2 + 2):
                x, y, z = low[p][i - 2], low[p][i - 1], low[p][i]
                real = f(f(f(f(f(x.real * a[0]) - f(x.imag * a[1])) + f(y.real * a[2])) - f(y.imag * a[3])) + z.real)
                imaginary = f(f(f(f(f(x.imag * a[0]) + f(x.real * a[1])) + f(y.imag * a[2])) + f(y.real * a[3])) + z.imag)
                high[k][i] = complex(real, imaginary)
            k += 1
    return high


def _mapping(data, state, freq, env, noise):
    original, floors, sinusoids = [], [], []
    state.s_index[1:8] = [[0] * 48 for _ in range(7)]
    for e in range(data.count):
        table = freq.high if data.freq_res[e + 1] else freq.low
        energy, floor, sinusoid = [0.] * freq.m, [0.] * freq.m, [0] * freq.m
        for i in range(len(table) - 1):
            for m in range(table[i] - freq.kx, table[i + 1] - freq.kx):
                energy[m] = env[e + 1][i]
        index = int(data.noise_count > 1 and data.time[e] >= data.noise_time[1])
        for i in range(len(freq.noise) - 1):
            for m in range(freq.noise[i] - freq.kx, freq.noise[i + 1] - freq.kx):
                floor[m] = noise[index + 1][i]
        if data.harmonic_flag:
            for i in range(len(freq.high) - 1):
                midpoint = ((freq.high[i] + freq.high[i + 1]) >> 1) - freq.kx
                state.s_index[e + 1][midpoint] = data.harmonic[i] * int(e >= data.e_a[1] or state.s_index[0][midpoint] == 1)
        for i in range(len(table) - 1):
            start, stop = table[i] - freq.kx, table[i + 1] - freq.kx
            value = int(any(state.s_index[e + 1][start:stop]))
            sinusoid[start:stop] = [value] * (stop - start)
        original.append(energy)
        floors.append(floor)
        sinusoids.append(sinusoid)
    state.s_index[0][:] = state.s_index[data.count]
    return original, floors, sinusoids


def _estimate(high, data, sbr):
    freq, estimated = sbr.frequency, []
    for e in range(data.count):
        start, stop = data.time[e] * 2 + 2, data.time[e + 1] * 2 + 2
        values = [0.] * freq.m
        if start >= 40:
            raise AacError('SBR envelope starts outside QMF data')
        if sbr.interpolate:
            reciprocal = f(.5 / (data.time[e + 1] - data.time[e]))
            for m in range(freq.m):
                values[m] = f(_square(high[m + freq.kx], start, stop) * reciprocal)
        else:
            table = freq.high if data.freq_res[e + 1] else freq.low
            for p in range(len(table) - 1):
                total = 0.
                for k in range(table[p], table[p + 1]):
                    total = f(total + _square(high[k], start, stop))
                total = f(total / ((stop - start) * (table[p + 1] - table[p])))
                values[table[p] - freq.kx:table[p + 1] - freq.kx] = [total] * (table[p + 1] - table[p])
        estimated.append(values)
    return estimated


def _gain(data, state, sbr, original, floors, sinusoids, estimated):
    epsilon, tiny = 2. ** -23, 2. ** -126
    gains, noises, harmonics = [], [], []
    freq = sbr.frequency
    limit = tuple(map(f, (.70795, 1., 1.41254, 10000000000.)))[sbr.limiter_gain]
    for e in range(data.count):
        delta = int(e not in data.e_a)
        gain, noise, harmonic = [0.] * freq.m, [0.] * freq.m, [0.] * freq.m
        for k in range(len(freq.limits) - 1):
            start, stop = freq.limits[k] - freq.kx, freq.limits[k + 1] - freq.kx
            for m in range(start, stop):
                energy, floor, current = original[e][m], floors[e][m], estimated[e][m]
                temp = f(energy / f(1 + floor))
                noise[m] = f(math.sqrt(f(temp * floor)))
                harmonic[m] = f(math.sqrt(f(temp * state.s_index[e + 1][m])))
                if not sinusoids[e][m]:
                    denominator = f(f(1 + current) * f(1 + f(floor * delta)))
                    value = f(energy / denominator)
                else:
                    denominator = f(f(1 + current) * f(1 + floor))
                    value = f(f(energy * floor) / denominator)
                gain[m] = f(f(math.sqrt(value)) + tiny)
            total0 = total1 = 0.
            for m in range(start, stop):
                total0, total1 = f(total0 + original[e][m]), f(total1 + estimated[e][m])
            maximum = min(100000., f(limit * f(math.sqrt(f(f(epsilon + total0) / f(epsilon + total1))))))
            for m in range(start, stop):
                noise[m] = min(noise[m], f(f(noise[m] * maximum) / gain[m]))
                gain[m] = min(gain[m], maximum)
            total0 = total1 = 0.
            for m in range(start, stop):
                total0 = f(total0 + original[e][m])
                energy = f(f(estimated[e][m] * gain[m]) * gain[m])
                energy = f(energy + f(harmonic[m] * harmonic[m]))
                energy = f(energy + f(f(int(delta and not harmonic[m]) * noise[m]) * noise[m]))
                total1 = f(total1 + energy)
            boost = min(f(1.584893192), f(math.sqrt(f(f(epsilon + total0) / f(epsilon + total1)))))
            for m in range(start, stop):
                gain[m], noise[m], harmonic[m] = f(gain[m] * boost), f(noise[m] * boost), f(harmonic[m] * boost)
        gains.append(gain)
        noises.append(noise)
        harmonics.append(harmonic)
    return gains, noises, harmonics


def _assemble(high, data, state, sbr, gain, noise, harmonic, output):
    smooth = 4 * (not sbr.smooth)
    freq = sbr.frequency
    weights = tuple(map(f, (.33333333333333, .30150283239582, .21816949906249, .11516383427084, .03183050093751)))
    if sbr.reset:
        for i in range(smooth):
            state.g_temp[i + data.time[0] * 2][:freq.m] = gain[0]
            state.q_temp[i + data.time[0] * 2][:freq.m] = noise[0]
    elif smooth:
        for i in range(4):
            state.g_temp[i + data.time[0] * 2][:] = state.g_temp[i + data.old_end * 2]
            state.q_temp[i + data.time[0] * 2][:] = state.q_temp[i + data.old_end * 2]
    for e in range(data.count):
        for i in range(data.time[e] * 2, data.time[e + 1] * 2):
            state.g_temp[smooth + i][:freq.m] = gain[e]
            state.q_temp[smooth + i][:freq.m] = noise[e]
    for e in range(data.count):
        attack = e in data.e_a
        for i in range(data.time[e] * 2, data.time[e + 1] * 2):
            if smooth and not attack:
                g_filt, q_filt = [0.] * freq.m, [0.] * freq.m
                for m in range(freq.m):
                    for j, weight in enumerate(weights):
                        g_filt[m] = f(g_filt[m] + f(state.g_temp[i + smooth - j][m] * weight))
                        q_filt[m] = f(q_filt[m] + f(state.q_temp[i + smooth - j][m] * weight))
            else:
                g_filt, q_filt = state.g_temp[i + smooth], state.q_temp[i]
            sine, index = data.sine_index, data.noise_index
            if attack:
                axis = sine & 1
                a = 1 - ((sine + (freq.kx & 1)) & 2)
                b = (a ^ -axis) + axis
            phi0 = (1., 0., -1., 0.)[sine]
            phi1 = (0., 1 - 2 * (freq.kx & 1), 0., -1 + 2 * (freq.kx & 1))[sine]
            for m in range(freq.m):
                value = high[freq.kx + m][i + 2]
                real, imaginary = f(value.real * g_filt[m]), f(value.imag * g_filt[m])
                index = (index + 1) & 511
                if attack:
                    value = f(harmonic[e][m] * (b if m & 1 else a))
                    if axis:
                        imaginary = f(imaginary + value)
                    else:
                        real = f(real + value)
                elif harmonic[e][m]:
                    real = f(real + f(harmonic[e][m] * phi0))
                    imaginary = f(imaginary + f(harmonic[e][m] * phi1))
                else:
                    real = f(real + f(q_filt[m] * NOISE[index][0]))
                    imaginary = f(imaginary + f(q_filt[m] * NOISE[index][1]))
                output[i][freq.kx + m] = complex(real, imaginary)
                phi1 = -phi1
            data.noise_index, data.sine_index = index, (sine + 1) & 3


class _DspChannel:
    def __init__(self):
        self.analysis, self.synthesis = Analysis(), Synthesis()
        self.previous_w = [[0j] * 32 for _ in range(32)]
        self.y_pos = 0
        self.y = [[[0j] * 64 for _ in range(38)] for _ in range(2)]
        self.bandwidth = [0.] * 5
        self.s_index = [[0] * 48 for _ in range(9)]
        self.g_temp = [[0.] * 48 for _ in range(42)]
        self.q_temp = [[0.] * 48 for _ in range(42)]


class Engine:
    def __init__(self, sbr):
        self.sbr = sbr
        self.channels = [_DspChannel(), _DspChannel()]

    def run(self, pcm, *, kind=0, downsample=False, ps_enabled=False):
        sbr = self.sbr
        if sbr.started and not sbr.ready:
            sbr.turnoff()
        if not sbr.pushed:
            sbr.previous_kx, sbr.previous_m = sbr.kx, sbr.m
        sbr.pushed = False
        if sbr.started:
            envelopes, noises = dequant(sbr, kind)
            sbr.ready = False
        all_rows = []
        for ch, samples in enumerate(pcm):
            state, data = self.channels[ch], sbr.channels[ch]
            w = state.analysis.run(samples)
            low = [[0j] * 40 for _ in range(32)]
            for k in range(sbr.kx):
                for i in range(32):
                    low[k][i + 8] = w[i][k]
            for k in range(sbr.previous_kx):
                for i in range(8):
                    low[k][i] = state.previous_w[i + 24][k]
            state.previous_w = w
            old_y = state.y[state.y_pos]
            state.y_pos ^= 1
            new_y = state.y[state.y_pos]
            if sbr.started:
                freq = sbr.frequency
                alphas = inverse_filter(low, freq.k0)
                _chirp(data, state.bandwidth, len(freq.noise) - 1)
                high = _generate(low, alphas, state.bandwidth, data, freq)
                original, floors, sinusoids = _mapping(data, state, freq, envelopes[ch], noises[ch])
                estimated = _estimate(high, data, sbr)
                gain, noise, harmonic = _gain(data, state, sbr, original, floors, sinusoids, estimated)
                _assemble(high, data, state, sbr, gain, noise, harmonic, new_y)
            rows = [[0j] * 64 for _ in range(38)]
            temporary = max(2 * data.old_end - 32, 0)
            for k in range(sbr.previous_kx):
                for i in range(temporary):
                    rows[i][k] = low[k][i + 2]
            for k in range(sbr.previous_kx, sbr.previous_kx + sbr.previous_m):
                for i in range(temporary):
                    rows[i][k] = old_y[i + 32][k]
            for k in range(sbr.kx):
                for i in range(temporary, 38):
                    rows[i][k] = low[k][i + 2]
            for k in range(sbr.kx, sbr.kx + sbr.m):
                for i in range(temporary, 32):
                    rows[i][k] = new_y[i][k]
            all_rows.append(rows)
        if ps_enabled:
            if sbr.ps is not None and sbr.ps.started:
                all_rows = sbr.ps.apply(all_rows[0], sbr.kx + sbr.m)
            else:
                all_rows = [all_rows[0], [row[:] for row in all_rows[0]]]
        return [state.synthesis.run(rows, downsample=downsample) for state, rows in zip(self.channels, all_rows)]
