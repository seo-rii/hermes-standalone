"""Layer III hybrid transform and polyphase synthesis from pinned minimp3 (CC0)."""

import struct
from .local_whisper_audio_mp3_tables import AA, TWID9, TWID3, MDCT_WINDOW, SEC, WIN


def _dct9(y):
    s0, s2, s4, s6, s8 = y[0::2]
    t0 = s0 + s6 * .5
    s0 -= s6
    t4 = (s4 + s2) * .9396926164627075
    t2 = (s8 + s2) * .7660444378852844
    s6 = (s4 - s8) * .1736481785774231
    s4 += s8 - s2
    s2 = s0 - s4 * .5
    y[4] = s4 + s0
    s8 = t0 - t2 + s6
    s0 = t0 - t4 + t2
    s4 = t0 + t4 - s6
    s1, s3, s5, s7 = y[1::2]
    s3 *= .8660253882408142
    t0 = (s5 + s1) * .9848077297210693
    t4 = (s5 - s7) * .3420201539993286
    t2 = (s1 + s7) * .6427876353263855
    s1 = (s1 - s5 - s7) * .8660253882408142
    s5 = t0 - s3 - t2
    s7 = t4 - s3 - t0
    s3 = t4 + s3 - t2
    y[0], y[1], y[2], y[3] = s4 - s7, s2 + s1, s0 - s3, s8 + s5
    y[5], y[6], y[7], y[8] = s8 - s5, s0 + s3, s2 - s1, s4 + s7


def _imdct36(values, overlap, window, first, count):
    for band in range(first, first + count):
        offset, previous = 18 * band, 9 * band
        co, si = [0.] * 9, [0.] * 9
        co[0], si[0] = -values[offset], values[offset + 17]
        for i in range(4):
            si[8 - 2 * i] = values[offset + 4 * i + 1] - values[offset + 4 * i + 2]
            co[1 + 2 * i] = values[offset + 4 * i + 1] + values[offset + 4 * i + 2]
            si[7 - 2 * i] = values[offset + 4 * i + 4] - values[offset + 4 * i + 3]
            co[2 + 2 * i] = -(values[offset + 4 * i + 3] + values[offset + 4 * i + 4])
        _dct9(co)
        _dct9(si)
        for i in range(1, 8, 2):
            si[i] = -si[i]
        for i in range(9):
            old = overlap[previous + i]
            total = co[i] * TWID9[9 + i] + si[i] * TWID9[i]
            overlap[previous + i] = co[i] * TWID9[i] - si[i] * TWID9[9 + i]
            values[offset + i] = old * window[i] - total * window[9 + i]
            values[offset + 17 - i] = old * window[9 + i] + total * window[i]


def _idct3(x0, x1, x2):
    m1 = x1 * .8660253882408142
    a1 = x0 - x2 * .5
    return [a1 + m1, x0 + x2, a1 - m1]


def _imdct12(values, first, previous):
    co = _idct3(-values[first], values[first + 6] + values[first + 3], values[first + 12] + values[first + 9])
    si = _idct3(values[first + 15], values[first + 12] - values[first + 9], values[first + 6] - values[first + 3])
    si[1] = -si[1]
    output, new = [0.] * 6, [0.] * 3
    for i in range(3):
        total = co[i] * TWID3[3 + i] + si[i] * TWID3[i]
        new[i] = co[i] * TWID3[i] - si[i] * TWID3[3 + i]
        output[i] = previous[i] * TWID3[2 - i] - total * TWID3[5 - i]
        output[5 - i] = previous[i] * TWID3[5 - i] + total * TWID3[2 - i]
    return output, new


def _imdct(values, overlap, block_type, long_bands):
    if long_bands:
        _imdct36(values, overlap, MDCT_WINDOW[0], 0, long_bands)
    if block_type != 2:
        _imdct36(values, overlap, MDCT_WINDOW[block_type == 3], long_bands, 32 - long_bands)
        return
    for band in range(long_bands, 32):
        offset, previous = 18 * band, 9 * band
        old = values[offset:offset + 18]
        values[offset:offset + 6] = overlap[previous:previous + 6]
        p1, remain = _imdct12(old, 0, overlap[previous + 6:previous + 9])
        p2, remain = _imdct12(old, 1, remain)
        p3, remain = _imdct12(old, 2, remain)
        values[offset + 6:offset + 18] = p1 + p2
        overlap[previous:previous + 9] = p3 + remain


def _dct32(values):
    for k in range(18):
        t = [[0.] * 8 for _ in range(4)]
        for i in range(8):
            x0, x1 = values[k + i * 18], values[k + (15 - i) * 18]
            x2, x3 = values[k + (16 + i) * 18], values[k + (31 - i) * 18]
            t0, t1 = x0 + x3, x1 + x2
            t2, t3 = (x1 - x2) * SEC[3 * i], (x0 - x3) * SEC[3 * i + 1]
            t[0][i], t[1][i] = t0 + t1, (t0 - t1) * SEC[3 * i + 2]
            t[2][i], t[3][i] = t3 + t2, (t3 - t2) * SEC[3 * i + 2]
        for x in t:
            x0, x1, x2, x3, x4, x5, x6, x7 = x
            xt, x0 = x0 - x7, x0 + x7
            x7, x1 = x1 - x6, x1 + x6
            x6, x2 = x2 - x5, x2 + x5
            x5, x3 = x3 - x4, x3 + x4
            x4, x0 = x0 - x3, x0 + x3
            x3, x1 = x1 - x2, x1 + x2
            x[0] = x0 + x1
            x[4] = (x0 - x1) * .7071067690849304
            x5 += x6
            x6 = (x6 + x7) * .7071067690849304
            x7 += xt
            x3 = (x3 + x4) * .7071067690849304
            x5 -= x7 * .1989123672246933
            x7 += x5 * .3826834261417389
            x5 -= x7 * .1989123672246933
            x0, xt = xt - x6, xt + x6
            x[1] = (xt + x7) * .509795606136322
            x[2] = (x4 + x3) * .5411961078643799
            x[3] = (x0 - x5) * .601344883441925
            x[5] = (x0 + x5) * .8999761939048767
            x[6] = (x4 - x3) * 1.3065630197525024
            x[7] = (xt - x7) * 2.562915563583374
        for i in range(7):
            offset = k + i * 72
            values[offset] = t[0][i]
            values[offset + 18] = t[2][i] + t[3][i] + t[3][i + 1]
            values[offset + 36] = t[1][i] + t[1][i + 1]
            values[offset + 54] = t[2][i + 1] + t[3][i] + t[3][i + 1]
        offset = k + 7 * 72
        values[offset] = t[0][7]
        values[offset + 18] = t[2][7] + t[3][7]
        values[offset + 36] = t[1][7]
        values[offset + 54] = t[3][7]


def _synth_pair(lins, offset):
    a = (lins[offset + 14 * 64] - lins[offset]) * 29
    a += (lins[offset + 64] + lins[offset + 13 * 64]) * 213
    a += (lins[offset + 12 * 64] - lins[offset + 2 * 64]) * 459
    a += (lins[offset + 3 * 64] + lins[offset + 11 * 64]) * 2037
    a += (lins[offset + 10 * 64] - lins[offset + 4 * 64]) * 5153
    a += (lins[offset + 5 * 64] + lins[offset + 9 * 64]) * 6574
    a += (lins[offset + 8 * 64] - lins[offset + 6 * 64]) * 37489
    a += lins[offset + 7 * 64] * 75038
    offset += 2
    b = (lins[offset + 14 * 64] * 104 + lins[offset + 12 * 64] * 1567 +
         lins[offset + 10 * 64] * 9727 + lins[offset + 8 * 64] * 64019 +
         lins[offset + 6 * 64] * -9975 + lins[offset + 4 * 64] * -45 +
         lins[offset + 2 * 64] * 146 + lins[offset] * -5)
    return a / 32768, b / 32768


def _synth(values, qmf):
    for channel in values:
        _dct32(channel)
    left, right = values[0], values[-1]
    lins = qmf[:] + [0.] * (18 * 64)
    output = [[0.] * 576 for _ in values]
    for k in range(0, 18, 2):
        base = 960 + k * 64
        lins[base + 60:base + 64] = [left[k + 288], right[k + 288], left[k], right[k]]
        lins[base + 124:base + 128] = [left[k + 289], right[k + 289], left[k + 1], right[k + 1]]
        for ch in range(len(values)):
            output[ch][k * 32], output[ch][k * 32 + 16] = _synth_pair(lins, k * 64 + 60 + ch)
            output[ch][k * 32 + 32], output[ch][k * 32 + 48] = _synth_pair(lins, k * 64 + 124 + ch)
        for i in range(14, -1, -1):
            position = base + 4 * i
            lins[position:position + 4] = [left[k + 18 * (31 - i)], right[k + 18 * (31 - i)],
                                           left[k + 1 + 18 * (31 - i)], right[k + 1 + 18 * (31 - i)]]
            lins[position + 64:position + 66] = [left[k + 1 + 18 * (1 + i)], right[k + 1 + 18 * (1 + i)]]
            lins[position - 62:position - 60] = [left[k + 18 * (1 + i)], right[k + 18 * (1 + i)]]
            a, b = [0.] * 4, [0.] * 4
            start = (14 - i) * 16
            for tap in range(8):
                w0, w1 = WIN[start + 2 * tap:start + 2 * tap + 2]
                iz, iy = position - tap * 64, position - (15 - tap) * 64
                sign = -1 if tap & 1 else 1
                for lane in range(4):
                    vz, vy = lins[iz + lane], lins[iy + lane]
                    b[lane] += vz * w1 + vy * w0
                    a[lane] += sign * (vz * w0 - vy * w1)
            for ch in range(len(values)):
                dest = output[ch]
                dest[k * 32 + 15 - i], dest[k * 32 + 17 + i] = a[ch] / 32768, b[ch] / 32768
                dest[k * 32 + 47 - i], dest[k * 32 + 49 + i] = a[ch + 2] / 32768, b[ch + 2] / 32768
    qmf[:] = lins[18 * 64:]
    return output


def transform(values, granules, header, overlaps, qmf):
    for channel, gr, overlap in zip(values, granules, overlaps):
        long_bands = (2 if gr.mixed_block_flag else 0) << (header.sr_index == 2)
        aa_bands = 31
        if gr.n_short_sfb:
            aa_bands = long_bands - 1
            position, reordered = long_bands * 18, []
            bands = gr.sfbtab[gr.n_long_sfb:]
            for count in bands[::3]:
                if not count:
                    break
                for i in range(count):
                    reordered.extend((channel[position + i], channel[position + count + i], channel[position + 2 * count + i]))
                position += 3 * count
            channel[long_bands * 18:position] = reordered
        if long_bands == 4 and gr.n_short_sfb:
            aa_bands = 1
        for band in range(aa_bands):
            for i in range(8):
                upper, lower = band * 18 + 18 + i, band * 18 + 17 - i
                u, d = channel[upper], channel[lower]
                channel[upper] = u * AA[0][i] - d * AA[1][i]
                channel[lower] = u * AA[1][i] + d * AA[0][i]
        # Match the pinned mp3float consumer's 8 kHz mixed-block hybrid
        # boundary: reordering starts at 72 lines, antialias/IMDCT at 36.
        hybrid_long = 2 if long_bands == 4 else long_bands
        _imdct(channel, overlap, gr.block_type, hybrid_long)
        for band in range(1, 32, 2):
            for i in range(1, 18, 2):
                channel[band * 18 + i] = -channel[band * 18 + i]
    result = _synth(values, qmf)
    return [list(struct.unpack('<' + str(len(ch)) + 'f', struct.pack('<' + str(len(ch)) + 'f', *ch))) for ch in result]
