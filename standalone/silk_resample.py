"""Stateful integer SILK SDK resampling for 8–48 kHz PCM16.

Derived from the Skype SILK SDK resampler, AR2/ARMA4 and allpass routines.
Original copyright and BSD-style notice: licenses/skype-silk-sdk.txt;
reviewed source hashes: silk-provenance.json. No native codec is loaded.

The SDK restarts its Q16 interpolation index in each input batch. Keeping its
batch boundaries, signed arithmetic and filter histories is part of the wire
decoder's bit-exact output contract, including nonintegral output rates.
"""

from math import gcd

from standalone import silk_tables as T
from standalone.silk_fixed import i32, rshift_round, sat16, smlawb, smulwb, smulww


def _up2_hq(state, samples):
    output = []
    c0, c1, notch = T.SKP_Silk_resampler_up2_hq_0, T.SKP_Silk_resampler_up2_hq_1, T.SKP_Silk_resampler_up2_hq_notch
    for sample in samples:
        source = sample << 10
        difference = i32(source - state[0])
        value = smulwb(difference, c0[0])
        first, state[0] = i32(state[0] + value), i32(source + value)
        difference = i32(first - state[1])
        value = smlawb(difference, difference, c0[1])
        second, state[1] = i32(state[1] + value), i32(first + value)
        second = smlawb(second, state[5], notch[2])
        second = smlawb(second, state[4], notch[1])
        first = smlawb(second, state[4], notch[0])
        state[5] = i32(second - state[5])
        output.append(sat16(smlawb(256, first, notch[3]) >> 9))

        difference = i32(source - state[2])
        value = smulwb(difference, c1[0])
        first, state[2] = i32(state[2] + value), i32(source + value)
        difference = i32(first - state[3])
        value = smlawb(difference, difference, c1[1])
        second, state[3] = i32(state[3] + value), i32(first + value)
        second = smlawb(second, state[4], notch[2])
        second = smlawb(second, state[5], notch[1])
        first = smlawb(second, state[5], notch[0])
        state[4] = i32(second - state[4])
        output.append(sat16(smlawb(256, first, notch[3]) >> 9))
    return output


def _up2_lq(state, samples):
    output = []
    for sample in samples:
        source = sample << 10
        difference = i32(source - state[0])
        value = smulwb(difference, T.SKP_Silk_resampler_up2_lq_0)
        first, state[0] = i32(state[0] + value), i32(source + value)
        output.append(sat16(rshift_round(first, 10)))
        difference = i32(source - state[1])
        value = smlawb(difference, difference, T.SKP_Silk_resampler_up2_lq_1)
        second, state[1] = i32(state[1] + value), i32(source + value)
        output.append(sat16(rshift_round(second, 10)))
    return output


def _down2(state, samples):
    output = []
    for index in range(len(samples) // 2):
        source = samples[2 * index] << 10
        difference = i32(source - state[0])
        value = smlawb(difference, difference, T.SKP_Silk_resampler_down2_1)
        filtered, state[0] = i32(state[0] + value), i32(source + value)
        source = samples[2 * index + 1] << 10
        difference = i32(source - state[1])
        value = smulwb(difference, T.SKP_Silk_resampler_down2_0)
        filtered = i32(i32(filtered + state[1]) + value)
        state[1] = i32(source + value)
        output.append(sat16(rshift_round(filtered, 11)))
    return output


def _ar2(state, samples, coefficients):
    output = []
    for sample in samples:
        value = i32(state[0] + (sample << 8))
        output.append(value)
        value = i32(value << 2)
        state[0] = smlawb(state[1], value, coefficients[0])
        state[1] = smulwb(value, coefficients[1])
    return output


def _arma4(state, samples, coefficients):
    output = []
    for sample in samples:
        source = sample << 8
        first = i32(source + i32(state[0] << 2))
        second = i32(first + i32(state[2] << 2))
        state[0] = smlawb(smlawb(state[1], source, coefficients[0]), first, coefficients[2])
        state[2] = smlawb(smlawb(state[3], first, coefficients[1]), second, coefficients[4])
        state[1] = smlawb(source >> 2, first, coefficients[3])
        state[3] = smlawb(first >> 2, second, coefficients[5])
        output.append(sat16(smlawb(128, second, coefficients[6]) >> 8))
    return output


class Resampler:
    """An independent SDK filter state; create a new instance after rate changes.

    ``process`` accepts at most one second of signed PCM16 samples per call.
    Hermes feeds 20 ms decoded frames. The optional divide-by-two stage follows
    the SDK's floor-length behavior for an odd final input sample.
    """

    def __init__(self, in_rate, out_rate):
        if any(type(rate) is not int or not 8000 <= rate <= 48000 for rate in (in_rate, out_rate)):
            raise ValueError("SILK resampling rates must be integers between 8000 and 48000 Hz")
        self.in_rate, self.out_rate = in_rate, out_rate
        self._iir, self._down = [0] * 6, [0] * 2
        self._fir = [0] * 12
        self._batch = in_rate // 100
        if self._batch * 100 != in_rate:
            cycle = in_rate // gcd(in_rate, out_rate)
            self._batch = max(1, 480 // cycle) * cycle if cycle <= 480 else 480
        self._up = self._pre_down = 0
        self._fracs, self._coefficients = 0, ()
        self._method = "copy"
        if out_rate == 2 * in_rate:
            self._method = "up2"
        elif out_rate > in_rate:
            self._method, self._up = "iir", 1
        elif out_rate < in_rate:
            for numerator, denominator, fracs, coefficients, down in (
                (3, 4, 3, T.SKP_Silk_Resampler_3_4_COEFS, 0),
                (2, 3, 2, T.SKP_Silk_Resampler_2_3_COEFS, 0),
                (1, 2, 1, T.SKP_Silk_Resampler_1_2_COEFS, 0),
                (3, 8, 3, T.SKP_Silk_Resampler_3_8_COEFS, 0),
                (1, 3, 1, T.SKP_Silk_Resampler_1_3_COEFS, 0),
                (1, 4, 1, T.SKP_Silk_Resampler_1_2_COEFS, 1),
                (1, 6, 1, T.SKP_Silk_Resampler_1_3_COEFS, 1),
            ):
                if out_rate * denominator == in_rate * numerator:
                    self._method, self._fracs = "down", fracs
                    self._coefficients, self._pre_down = coefficients, down
                    break
            else:
                self._method = "iir"
                for numerator in (80, 120, 160, 240, 320):
                    if out_rate * 441 == in_rate * numerator:
                        self._coefficients = getattr(T, f"SKP_Silk_Resampler_{numerator}_441_ARMA4_COEFS")
                        break
                else:
                    self._up = 1
        self._step = ((in_rate << (14 + self._up - self._pre_down)) // out_rate) << 2
        while smulww(self._step, out_rate << self._pre_down) < in_rate << self._up:
            self._step += 1

    def process(self, samples):
        if not isinstance(samples, list) or len(samples) > self.in_rate:
            raise ValueError("SILK resampling requires a bounded PCM16 sample list")
        if any(type(sample) is not int or not -32768 <= sample <= 32767 for sample in samples):
            raise ValueError("SILK resampling requires signed PCM16 integers")
        if self._method == "copy":
            return list(samples)
        if self._method == "up2":
            return _up2_hq(self._iir, samples)
        if not samples:
            return []
        if self._method == "down":
            return self._process_down(samples)
        return self._process_iir(samples)

    def _process_iir(self, samples):
        output = []
        for offset in range(0, len(samples), self._batch):
            chunk = samples[offset:offset + self._batch]
            if self._up:
                filtered = (_up2_hq if self.in_rate <= 24000 else _up2_lq)(self._iir, chunk)
            else:
                filtered = _arma4(self._iir, chunk, self._coefficients)
            buffer = self._fir[:6] + filtered
            for position in range(0, len(filtered) << 16, self._step):
                index, fraction = position >> 16, ((position & 0xffff) * 144) >> 16
                first, second = T.SKP_Silk_resampler_frac_FIR_144[fraction], T.SKP_Silk_resampler_frac_FIR_144[143 - fraction]
                value = 0
                for tap, coefficient in enumerate((*first, *reversed(second))):
                    value = i32(value + buffer[index + tap] * coefficient)
                output.append(sat16(rshift_round(value, 15)))
            self._fir[:6] = buffer[len(filtered):len(filtered) + 6]
        return output

    def _process_down(self, samples):
        output, offset = [], 0
        while offset < len(samples):
            chunk = samples[offset:offset + self._batch]
            if self._pre_down:
                filtered = _ar2(self._iir, _down2(self._down, chunk), self._coefficients)
            else:
                filtered = _ar2(self._iir, chunk, self._coefficients)
            buffer = self._fir + filtered
            coefficients = self._coefficients[2:]
            for position in range(0, len(filtered) << 16, self._step):
                index, value = position >> 16, 0
                if self._fracs == 1:
                    for tap in range(6):
                        pair = i32(buffer[index + tap] + buffer[index + 11 - tap])
                        value = smlawb(value, pair, coefficients[tap])
                else:
                    phase = ((position & 0xffff) * self._fracs) >> 16
                    reverse = self._fracs - 1 - phase
                    for tap in range(6):
                        value = smlawb(value, buffer[index + tap], coefficients[6 * phase + tap])
                    for tap in range(6):
                        value = smlawb(value, buffer[index + 11 - tap], coefficients[6 * reverse + tap])
                output.append(sat16(rshift_round(value, 6)))
            self._fir = buffer[len(filtered):len(filtered) + 12]
            offset += len(filtered) << self._pre_down
            if len(samples) - offset <= self._pre_down:
                break
        return output
