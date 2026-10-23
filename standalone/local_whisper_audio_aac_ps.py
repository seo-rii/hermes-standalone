"""Stateful MPEG-4 Parametric Stereo DSP from pinned FFmpeg 8.1.2 (LGPL)."""
from standalone.local_whisper_audio_aac_dsp import f32 as f
from .local_whisper_audio_aac_ps_bits import PsBits
from .local_whisper_audio_aac_ps_tables import (
    F20_0_8, F34_0_12, F34_1_8, F34_2_4, K_TO_I_20, K_TO_I_34,
    HA, HB, PD_RE_SMOOTH, PD_IM_SMOOTH, PHI_FRACT, Q_FRACT_ALLPASS,
)

_G1 = tuple(map(f, (0., .01899487526049, 0., -.07293139167538, 0., .30596630545168, .5)))
_A = tuple(map(f, (.65143905753106, .56471812200776, .48954165955695)))
_PEAK = f(.76592833836465)


def _map20_to34(values, full, floating=False):
    result = [0] * 34
    index = (0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10,
             11, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 18, 18, 19, 19)
    end = 34 if full else 17
    for k in range(end):
        result[k] = values[index[k]]
    for k, a, b in ((1, 0, 1), (4, 2, 3)):
        total = f(values[a] + values[b]) if floating else values[a] + values[b]
        result[k] = f(total * .5) if floating else int(total / 2)
    return result


def _map34_to20(values, full, floating=False):
    groups = (((0, 2), (1, 1)), ((1, 1), (2, 2)), ((3, 2), (4, 1)), ((4, 1), (5, 2)),
              ((6, 1), (7, 1)), ((8, 1), (9, 1)), ((10, 1),), ((11, 1),),
              ((12, 1), (13, 1)), ((14, 1), (15, 1)), ((16, 1),), ((17, 1),),
              ((18, 1),), ((19, 1),), ((20, 1), (21, 1)), ((22, 1), (23, 1)),
              ((24, 1), (25, 1)), ((26, 1), (27, 1)), ((28, 1), (29, 1), (30, 1), (31, 1)),
              ((32, 1), (33, 1)))
    result = list(values) if floating else [0] * 34
    for k, group in enumerate(groups[:20 if full else 11]):
        if floating:
            total = f(values[group[0][0]] * group[0][1])
            for index, weight in group[1:]:
                total = f(total + f(values[index] * weight))
            result[k] = f(total * (f(.33333333) if k < 4 else 1 / sum(weight for _, weight in group)))
        else:
            result[k] = int(sum(values[index] * weight for index, weight in group) / sum(weight for _, weight in group))
    return result


def _remap(values, count, is34, full):
    if is34:
        if count in (20, 11):
            return _map20_to34(values, full)
        if count in (10, 5):
            indices = (0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5,
                       6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9)
            result = [values[index] for index in indices[:34 if full else 16]]
            return result + [0] * (34 - len(result))
    else:
        if count in (34, 17):
            return _map34_to20(values, full)
        if count in (10, 5):
            result = [values[k // 2] for k in range(20 if full else 10)]
            return result + [0] * (34 - len(result))
    return values


def _complex_filter(samples, filters):
    pairs = []
    for j in range(6):
        a, b = samples[j], samples[12 - j]
        pairs.append((f(a.real + b.real), f(a.imag - b.imag), f(a.imag + b.imag), f(a.real - b.real)))
    result = []
    for row in filters:
        real, imag = f(row[6][0] * samples[6].real), f(row[6][0] * samples[6].imag)
        for j, (re0, re1, im0, im1) in enumerate(pairs):
            re, im = row[j]
            real = f(real + f(f(re * re0) - f(im * re1)))
            imag = f(imag + f(f(re * im0) + f(im * im1)))
        result.append(complex(real, imag))
    return result


def _sum_complex(values):
    real = imag = 0.
    for value in values:
        real, imag = f(real + value.real), f(imag + value.imag)
    return complex(real, imag)


class Ps(PsBits):
    def __init__(self):
        super().__init__()
        self.in_buf = [[0j] * 44 for _ in range(5)]
        self.delay = [[0j] * 46 for _ in range(91)]
        self.ap_delay = [[[0j] * 37 for _ in range(3)] for _ in range(50)]
        self.peak_decay_nrg = [0.] * 34
        self.power_smooth = [0.] * 34
        self.peak_decay_diff_smooth = [0.] * 34
        self.h = [[[[0.] * 34 for _ in range(6)] for _ in range(2)] for _ in range(4)]
        self.ipd_hist, self.opd_hist = [0] * 17, [0] * 17

    def _analysis(self, rows):
        out = [[0j] * 32 for _ in range(91)]
        for k in range(5):
            self.in_buf[k][6:] = [row[k] for row in rows]
        if self.is34bands:
            offset = 0
            for k, filters in enumerate((F34_0_12, F34_1_8, F34_2_4, F34_2_4, F34_2_4)):
                for n in range(32):
                    values = _complex_filter(self.in_buf[k][n:n + 13], filters)
                    for j, value in enumerate(values):
                        out[offset + j][n] = value
                offset += len(filters)
            begin, offset = 5, 27
        else:
            for n in range(32):
                z = _complex_filter(self.in_buf[0][n:n + 13], F20_0_8)
                values = (z[6], z[7], z[0], z[1], _sum_complex((z[2], z[5])), _sum_complex((z[3], z[4])))
                for k, value in enumerate(values):
                    out[k][n] = value
            for band, offset, reverse in ((1, 6, 1), (2, 8, 0)):
                for n in range(32):
                    data = self.in_buf[band][n:n + 13]
                    in_re, in_im = f(_G1[6] * data[6].real), f(_G1[6] * data[6].imag)
                    op_re = op_im = 0.
                    for j in range(0, 6, 2):
                        op_re = f(op_re + f(_G1[j + 1] * f(data[j + 1].real + data[11 - j].real)))
                        op_im = f(op_im + f(_G1[j + 1] * f(data[j + 1].imag + data[11 - j].imag)))
                    out[offset + reverse][n] = complex(f(in_re + op_re), f(in_im + op_im))
                    out[offset + (not reverse)][n] = complex(f(in_re - op_re), f(in_im - op_im))
            begin, offset = 3, 7
        for band in range(begin, 64):
            out[band + offset][:] = [row[band] for row in rows[:32]]
        for data in self.in_buf:
            data[:6] = data[32:38]
        return out

    def _decorrelate(self, values):
        is34 = self.is34bands
        count, cutoff, allpass, short = ((71, 10, 30, 42), (91, 32, 50, 62))[is34]
        index = K_TO_I_34 if is34 else K_TO_I_20
        if is34 != self.is34bands_old:
            self.peak_decay_nrg[:] = self.power_smooth[:] = self.peak_decay_diff_smooth[:] = [0.] * 34
            self.delay = [[0j] * 46 for _ in range(91)]
            self.ap_delay = [[[0j] * 37 for _ in range(3)] for _ in range(50)]
        power = [[0.] * 32 for _ in range(34)]
        for k in range(count):
            row = power[index[k]]
            for n, z in enumerate(values[k]):
                row[n] = f(row[n] + f(f(z.real * z.real) + f(z.imag * z.imag)))
        gains = [[0.] * 32 for _ in range(34)]
        for k in range(34 if is34 else 20):
            peak, smooth, difference = self.peak_decay_nrg[k], self.power_smooth[k], self.peak_decay_diff_smooth[k]
            for n, energy in enumerate(power[k]):
                peak = max(f(_PEAK * peak), energy)
                smooth = f(smooth + f(.25 * f(energy - smooth)))
                difference = f(difference + f(.25 * f(f(peak - energy) - difference)))
                denominator = f(1.5 * difference)
                gains[k][n] = f(smooth / denominator) if denominator > smooth else 1.
            self.peak_decay_nrg[k], self.power_smooth[k], self.peak_decay_diff_smooth[k] = peak, smooth, difference
        out = [[0j] * 32 for _ in range(91)]
        for k in range(count):
            delay = self.delay[k]
            delay[:14], delay[14:] = delay[32:46], values[k][:]
            gain = gains[index[k]]
            if k >= allpass:
                distance = 14 if k < short else 1
                for n in range(32):
                    z = delay[n + 14 - distance]
                    out[k][n] = complex(f(gain[n] * z.real), f(gain[n] * z.imag))
                continue
            slope = min(max(f(1. - f(f(.05) * (k - cutoff))), 0.), 1.)
            ag = [f(a * slope) for a in _A]
            for row in self.ap_delay[k]:
                row[:5] = row[32:37]
            phi_r, phi_i = PHI_FRACT[is34][k]
            for n in range(32):
                z = delay[n + 12]
                real = f(f(z.real * phi_r) - f(z.imag * phi_i))
                imag = f(f(z.real * phi_i) + f(z.imag * phi_r))
                for m in range(3):
                    row = self.ap_delay[k][m]
                    z = row[n + 2 - m]
                    q_r, q_i = Q_FRACT_ALLPASS[is34][k][m]
                    old_r, old_i = real, imag
                    real = f(f(f(z.real * q_r) - f(z.imag * q_i)) - f(ag[m] * old_r))
                    imag = f(f(f(z.real * q_i) + f(z.imag * q_r)) - f(ag[m] * old_i))
                    row[n + 5] = complex(f(old_r + f(ag[m] * real)), f(old_i + f(ag[m] * imag)))
                out[k][n] = complex(f(gain[n] * real), f(gain[n] * imag))
        return out

    def _mix(self, left, right):
        is34 = self.is34bands
        nr, bands, phases = (34, 91, 17) if is34 else (20, 71, 11)
        index = K_TO_I_34 if is34 else K_TO_I_20
        if self.num_env_old:
            for component in self.h:
                for row in component:
                    row[0][:] = row[self.num_env_old]
        iid = [_remap(row, self.nr_iid_par, is34, True) for row in self.iid_par[:self.num_env]]
        icc = [_remap(row, self.nr_icc_par, is34, True) for row in self.icc_par[:self.num_env]]
        if self.enable_ipdopd:
            ipd = [_remap(row, self.nr_ipdopd_par, is34, False) for row in self.ipd_par[:self.num_env]]
            opd = [_remap(row, self.nr_ipdopd_par, is34, False) for row in self.opd_par[:self.num_env]]
        if is34 != self.is34bands_old:
            for component in self.h:
                for row in component:
                    row[0][:] = _map20_to34(row[0], True, True) if is34 else _map34_to20(row[0], True, True)
            self.ipd_hist[:] = self.opd_hist[:] = [0] * 17
        table = HA if self.icc_mode < 3 else HB
        for e in range(self.num_env):
            for b in range(nr):
                h = list(table[iid[e][b] + 7 + 23 * self.iid_quant][icc[e][b]])
                if self.enable_ipdopd and b < phases:
                    oi, ii = self.opd_hist[b] * 8 + opd[e][b], self.ipd_hist[b] * 8 + ipd[e][b]
                    or_, oim, ir, iim = PD_RE_SMOOTH[oi], PD_IM_SMOOTH[oi], PD_RE_SMOOTH[ii], PD_IM_SMOOTH[ii]
                    self.opd_hist[b], self.ipd_hist[b] = oi & 63, ii & 63
                    adj_r, adj_i = f(f(or_ * ir) + f(oim * iim)), f(f(oim * ir) - f(or_ * iim))
                    for k in range(4):
                        self.h[k][1][e + 1][b] = f(h[k] * (oim if k in (0, 2) else adj_i))
                        h[k] = f(h[k] * (or_ if k in (0, 2) else adj_r))
                for k in range(4):
                    self.h[k][0][e + 1][b] = h[k]
            start, stop = self.border_position[e:e + 2]
            width = f(1. / (stop - start or 1))
            for k in range(bands):
                b = index[k]
                real = [component[0][e][b] for component in self.h]
                step_r = [f(f(component[0][e + 1][b] - real[j]) * width) for j, component in enumerate(self.h)]
                if self.enable_ipdopd:
                    sign = -1 if (is34 and 9 <= k <= 13) or (not is34 and k <= 1) else 1
                    imag = [sign * component[1][e][b] for component in self.h]
                    step_i = [f(f(component[1][e + 1][b] - imag[j]) * width) for j, component in enumerate(self.h)]
                for n in range(start + 1, stop + 1):
                    a, z = left[k][n], right[k][n]
                    real = [f(v + d) for v, d in zip(real, step_r)]
                    lr = f(f(real[0] * a.real) + f(real[2] * z.real))
                    li = f(f(real[0] * a.imag) + f(real[2] * z.imag))
                    rr = f(f(real[1] * a.real) + f(real[3] * z.real))
                    ri = f(f(real[1] * a.imag) + f(real[3] * z.imag))
                    if self.enable_ipdopd:
                        imag = [f(v + d) for v, d in zip(imag, step_i)]
                        lr = f(f(lr - f(imag[0] * a.imag)) - f(imag[2] * z.imag))
                        li = f(f(li + f(imag[0] * a.real)) + f(imag[2] * z.real))
                        rr = f(f(rr - f(imag[1] * a.imag)) - f(imag[3] * z.imag))
                        ri = f(f(ri + f(imag[1] * a.real)) + f(imag[3] * z.real))
                    left[k][n], right[k][n] = complex(lr, li), complex(rr, ri)

    def _synthesis(self, values, rows):
        is34 = self.is34bands
        groups = ((0, 12), (12, 20), (20, 24), (24, 28), (28, 32)) if is34 else ((0, 6), (6, 8), (8, 10))
        for n in range(32):
            for band, (start, stop) in enumerate(groups):
                rows[n][band] = _sum_complex(values[k][n] for k in range(start, stop))
            for band in range(len(groups), 64):
                rows[n][band] = values[band + (27 if is34 else 7)][n]
        return rows

    def apply(self, rows, top):
        is34 = self.is34bands
        count, allpass = (91, 50) if is34 else (71, 30)
        top += count - 64
        for k in range(top, count):
            self.delay[k][:] = [0j] * 46
        for k in range(top, allpass):
            self.ap_delay[k] = [[0j] * 37 for _ in range(3)]
        left = self._analysis(rows)
        right = self._decorrelate(left)
        self._mix(left, right)
        other = [[0j] * 64 for _ in range(38)]
        return [self._synthesis(left, rows), self._synthesis(right, other)]
