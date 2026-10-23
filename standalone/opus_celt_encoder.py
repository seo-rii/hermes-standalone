"""CELT audio encoding using the Opus 1.3.1 transform/PVQ bitstream.

Copyright (c) 2003–2011 Xiph.Org Foundation, CSIRO, Mark Borgerding,
Timothy B. Terriberry and Jean-Marc Valin. Original notices: licenses/opus*.txt.
"""
from math import atan2, floor, isfinite, log2, pi, sqrt
from struct import unpack
from .opus import pack_frames, parse_packet
from .opus_range import OpusRangeEncoder
from .opus_celt_alloc import allocation, bits2pulses, pulses2bits, get_pulses, init_caps, cdiv
from .opus_celt_bands import bitexact_cos, bitexact_log2tan, compute_qn, frac_mul, haar, interleave
from .opus_celt_energy import _TF
from .opus_celt_dsp import fft
from .opus_celt_pvq import encode_pulses, rotation
from .opus_tables import EBANDS, CACHE_BITS, CACHE_INDEX, ENERGY_MEAN, ENERGY_PROB, LOG_N, MDCT_TRIG, PREEMPH, WINDOW


def mdct(samples, lm):
    size = 120 << lm
    quarter = size >> 1
    shift = 3 - lm
    trig_offset = sum(1920 >> i for i in range(1,shift + 1))
    trig = MDCT_TRIG[trig_offset:trig_offset + size]
    rotated = []
    for i in range(quarter):
        first,second = 60 + 2 * i,size - 1 + 60 - 2 * i
        if i < 30:
            w1,w2 = WINDOW[60 + 2 * i],WINDOW[59 - 2 * i]
            real = w2 * samples[first + size] + w1 * samples[second]
            imaginary = w1 * samples[first] - w2 * samples[second - size]
        elif i < quarter - 30:
            real,imaginary = samples[second],samples[first]
        else:
            index = 2 * (i - quarter + 30)
            w1,w2 = WINDOW[index],WINDOW[119 - index]
            real = -w1 * samples[first - size] + w2 * samples[second]
            imaginary = w2 * samples[first] + w1 * samples[second + size]
        rotated.append(complex((real * trig[i] - imaginary * trig[quarter + i]) / quarter,
                               (imaginary * trig[i] + real * trig[quarter + i]) / quarter))
    transformed = fft(rotated)
    result = [0.] * size
    for i,value in enumerate(transformed):
        result[2 * i] = value.imag * trig[quarter + i] - value.real * trig[i]
        result[size - 1 - 2 * i] = value.real * trig[quarter + i] + value.imag * trig[i]
    return result


def pvq_search(values, pulses):
    signs = [value < 0 for value in values]
    absolute = [abs(value) for value in values]
    dimensions = len(values)
    integers, doubled = [0] * dimensions,[0] * dimensions
    xy = yy = 0.
    left = pulses
    if pulses > dimensions >> 1:
        total = sum(absolute)
        if not 1e-15 < total < 64:
            absolute = [1.] + [0.] * (dimensions - 1)
            total = 1.
        reciprocal = (pulses + .8) / total
        for i in range(dimensions):
            integers[i] = floor(reciprocal * absolute[i])
            yy += integers[i] ** 2
            xy += absolute[i] * integers[i]
            doubled[i] = integers[i] * 2
            left -= integers[i]
    if left > dimensions + 3:
        yy += left * left + left * doubled[0]
        integers[0] += left
        left = 0
    for _ in range(left):
        best = 0
        yy += 1
        numerator = (xy + absolute[0]) ** 2
        denominator = yy + doubled[0]
        for j in range(1,dimensions):
            candidate = (xy + absolute[j]) ** 2
            magnitude = yy + doubled[j]
            if denominator * candidate > magnitude * numerator:
                numerator,denominator,best = candidate,magnitude,j
        xy += absolute[best]
        yy += doubled[best]
        doubled[best] += 2
        integers[best] += 1
    return [-value if sign else value for value,sign in zip(integers,signs)]


class _AllocationWriter:
    def __init__(self,encoder):
        self.encoder = encoder

    def bit_logp(self,logp):
        # Keep the last usable band and use dual stereo below its intensity cut.
        self.encoder.bit_logp(1,logp)
        return 1

    def uint(self,total):
        self.encoder.uint(total - 1,total)
        return total - 1


class _BandsWriter:
    def __init__(self,encoder,spread):
        self.encoder,self.spread = encoder,spread
        self.band = self.remaining = 0

    def theta(self,first,second,bits,lm,original_blocks):
        dimensions = len(first)
        pulse_cap = LOG_N[self.band] + lm * 8
        qn = compute_qn(dimensions,bits,(pulse_cap >> 1) - 4,pulse_cap,False)
        angle = 0
        before = self.encoder.tell_frac()
        if qn != 1:
            middle = sqrt(1e-15 + sum(value * value for value in first))
            side = sqrt(1e-15 + sum(value * value for value in second))
            unquantized = floor(.5 + 16384 * 2 / pi * atan2(side,middle))
            quantized = (unquantized * qn + 8192) >> 14
            if original_blocks > 1:
                self.encoder.uint(quantized,qn + 1)
            else:
                total = ((qn >> 1) + 1) ** 2
                frequency = quantized + 1 if quantized <= qn >> 1 else qn + 1 - quantized
                low = quantized * (quantized + 1) >> 1 if quantized <= qn >> 1 else total - ((qn + 1 - quantized) * (qn + 2 - quantized) >> 1)
                self.encoder.encode(low,low + frequency,total)
            angle = quantized * 16384 // qn
        allocated = self.encoder.tell_frac() - before
        bits -= allocated
        if angle == 0:
            delta = -16384
        elif angle == 16384:
            delta = 16384
        else:
            middle,side = bitexact_cos(angle),bitexact_cos(16384 - angle)
            delta = frac_mul((dimensions - 1) << 7,bitexact_log2tan(side,middle))
        return bits,delta,angle,allocated

    def partition(self,values,bits,lm,blocks=1):
        dimensions = len(values)
        offset = CACHE_INDEX[(lm + 1) * 21 + self.band]
        if lm != -1 and bits > CACHE_BITS[offset + CACHE_BITS[offset]] + 12 and dimensions > 2:
            half = dimensions >> 1
            first,second = values[:half],values[half:]
            bits,delta,angle,allocated = self.theta(first,second,bits,lm - 1,blocks)
            if blocks > 1 and angle & 0x3fff:
                if angle > 8192:
                    delta -= delta >> (5 - lm)
                else:
                    delta = min(0,delta + (half << 3 >> (6 - lm)))
            next_blocks = (blocks + 1) >> 1
            middle_bits = max(0,min(bits,cdiv(bits - delta,2)))
            side_bits = bits - middle_bits
            self.remaining -= allocated
            before = self.remaining
            if middle_bits >= side_bits:
                self.partition(first,middle_bits,lm - 1,next_blocks)
                rebalance = middle_bits - (before - self.remaining)
                if rebalance > 24 and angle != 0:
                    side_bits += rebalance - 24
                self.partition(second,side_bits,lm - 1,next_blocks)
            else:
                self.partition(second,side_bits,lm - 1,next_blocks)
                rebalance = side_bits - (before - self.remaining)
                if rebalance > 24 and angle != 16384:
                    middle_bits += rebalance - 24
                self.partition(first,middle_bits,lm - 1,next_blocks)
            return
        pulse_index = bits2pulses(self.band,lm,bits)
        used = pulses2bits(self.band,lm,pulse_index)
        self.remaining -= used
        while self.remaining < 0 and pulse_index > 0:
            self.remaining += used
            pulse_index -= 1
            used = pulses2bits(self.band,lm,pulse_index)
            self.remaining -= used
        if pulse_index:
            pulses = get_pulses(pulse_index)
            values = values[:]
            rotation(values,1,blocks,pulses,self.spread)
            encode_pulses(self.encoder,pvq_search(values,pulses))

    def transform_band(self,values,blocks,tf_change):
        dimensions = len(values)
        n_block = dimensions // blocks
        long_blocks = blocks == 1
        recombine = max(0,tf_change)
        for k in range(recombine):
            haar(values,dimensions >> k,1 << k)
        blocks >>= recombine
        n_block <<= recombine
        while not n_block & 1 and tf_change < 0:
            haar(values,n_block,blocks)
            blocks <<= 1
            n_block >>= 1
            tf_change += 1
        if blocks > 1:
            interleave(values,n_block >> recombine,blocks << recombine,long_blocks)
        return blocks

    def all_bands(self,spectra,pulses,total_bits,balance,lm,coded,dual,intensity,blocks,tf,end_band):
        channels = len(spectra)
        for band in range(end_band):
            self.band = band
            tell = self.encoder.tell_frac()
            if band != 0:
                balance -= tell
            self.remaining = total_bits - tell - 1
            bits = max(0,min(16383,self.remaining + 1,pulses[band] + cdiv(balance,min(3,coded - band)))) if band < coded else 0
            begin,end = EBANDS[band] << lm,EBANDS[band + 1] << lm
            inputs = spectra
            if channels == 2 and (not dual or band >= intensity) and end - begin > 1:
                # With too few bits to reserve stereo parameters, the bitstream
                # implies intensity stereo. Code its shared mid once.
                inverse = int(sum(x*y for x,y in zip(spectra[0][begin:end],spectra[1][begin:end])) < 0)
                before = self.encoder.tell_frac()
                if bits > 16 and self.remaining > 16:
                    self.encoder.bit_logp(inverse,2)
                else:
                    inverse = 0
                allocated = self.encoder.tell_frac() - before
                bits -= allocated
                self.remaining -= allocated
                sign = -1 if inverse else 1
                mixed = [x + sign*y for x,y in zip(spectra[0][begin:end],spectra[1][begin:end])]
                changed_blocks = self.transform_band(mixed,blocks,tf[band])
                self.partition(mixed,bits,lm,changed_blocks)
                balance += pulses[band] + tell
                continue
            for values in inputs:
                if end - begin == 1:
                    if self.remaining >= 8:
                        self.encoder.bits(int(values[begin] < 0),1)
                        self.remaining -= 8
                else:
                    transformed = values[begin:end]
                    changed_blocks = self.transform_band(transformed,blocks,tf[band])
                    self.partition(transformed,bits // channels,lm,changed_blocks)
            balance += pulses[band] + tell


class CeltEncoder:
    """48 kHz mono/stereo CELT encoder; the coding mode remains CELT-only.

    The encoder currently selects dual stereo, no pitch
    postfilter, and no dynamic band boosts. SILK/hybrid/FEC coding is separate.
    All frequency bands and the full Opus CELT PVQ resolution remain available.
    """
    def __init__(self,channels=2,bitrate=128000,bandwidth='full'):
        if channels not in (1,2) or type(bitrate) is not int or not 16000 <= bitrate <= 524288:
            raise ValueError('Invalid Opus CELT encoder configuration')
        self.channels,self.bitrate = channels,bitrate
        self.set_bandwidth(bandwidth)
        self.reset()

    def set_bitrate(self,bitrate):
        if type(bitrate) is not int or not 16000 <= bitrate <= 524288:
            raise ValueError('Invalid Opus CELT encoder bitrate')
        self.bitrate = bitrate
        return bitrate

    def set_bandwidth(self,bandwidth):
        bands = {'narrow':13,'medium':17,'wide':17,'superwide':19,'full':21}
        if bandwidth not in bands:
            raise ValueError('Invalid Opus CELT encoder bandwidth')
        self.end = bands[bandwidth]
        self.bandwidth = bandwidth

    def reset(self):
        self.overlap = [[0.] * 120 for _ in range(self.channels)]
        self.preemphasis_memory = [0.] * self.channels
        self.old_energy = [0.] * 42
        self.frame_count,self.rng = 0,0

    def _coarse(self,target,encoder,lm,intra):
        coefficients = (29440/32768,26112/32768,21248/32768,16384/32768)
        betas = (30147/32768,22282/32768,12124/32768,6554/32768)
        coefficient = 0 if intra else coefficients[lm]
        beta = 4915/32768 if intra else betas[lm]
        previous = [0.,0.]
        budget = encoder.storage * 8
        probabilities = ENERGY_PROB[lm][intra]
        for band in range(self.end):
            for channel in range(self.channels):
                index = band + channel * 21
                old = max(-9.,self.old_energy[index])
                difference = target[index] - coefficient * old - previous[channel]
                delta = floor(.5 + difference)
                left = budget - encoder.tell()
                safe = left - 3 * self.channels * (self.end - band)
                if band != 0 and safe < 30:
                    if safe < 24:
                        delta = min(1,delta)
                    if safe < 16:
                        delta = max(-1,delta)
                if left >= 15:
                    delta = encoder.laplace(delta,probabilities[2 * band] << 7,probabilities[2 * band + 1] << 6)
                elif left >= 2:
                    delta = max(-1,min(delta,1))
                    encoder.icdf(2 * delta ^ -int(delta < 0),(2,1,0),2)
                elif left >= 1:
                    delta = max(-1,min(delta,0))
                    encoder.bit_logp(-delta,1)
                else:
                    delta = -1
                self.old_energy[index] = coefficient * old + previous[channel] + delta
                previous[channel] += (1 - beta) * delta

    def encode_float(self,pcm,frame_size=960):
        if frame_size not in (120,240,480,960,1920,2880,3840,4800,5760) or len(pcm) != frame_size * self.channels:
            raise ValueError('Invalid Opus CELT PCM frame length')
        if any(not isfinite(value) for value in pcm):
            raise ValueError('Opus CELT PCM must be finite')
        if frame_size > 960:
            packets = [parse_packet(self.encode_float(pcm[offset:offset+960*self.channels],960))
                       for offset in range(0,len(pcm),960*self.channels)]
            return pack_frames(packets[0].toc,tuple(packet.frames[0] for packet in packets))
        lm = (frame_size // 120).bit_length() - 1
        spectra,target = [],[0.] * 42
        prepared = []
        signal = False
        for channel in range(self.channels):
            memory = self.preemphasis_memory[channel]
            samples = self.overlap[channel][:]
            for i in range(frame_size):
                value = max(-65536.,min(65536.,float(pcm[i * self.channels + channel]) * 32768.))
                samples.append(value - memory)
                memory = PREEMPH[0] * value
            self.preemphasis_memory[channel] = memory
            self.overlap[channel] = samples[frame_size:frame_size + 120]
            signal |= any(samples)
            prepared.append(samples)
        energies = [sum(value * value for samples in prepared for value in samples[120 + block * 120:240 + block * 120]) for block in range(1 << lm)]
        transient = int(lm > 0 and max(energies) > max(1.,8 * min(energies)))
        for channel,samples in enumerate(prepared):
            if transient:
                frequencies = [0.] * frame_size
                for block in range(1 << lm):
                    part = mdct(samples[block * 120:block * 120 + 240],0)
                    frequencies[block::1 << lm] = part
            else:
                frequencies = mdct(samples,lm)
            for band in range(self.end):
                begin,end = EBANDS[band] << lm,EBANDS[band + 1] << lm
                energy = sqrt(1e-27 + sum(value * value for value in frequencies[begin:end]))
                target[channel * 21 + band] = log2(energy) - ENERGY_MEAN[band]
                for i in range(begin,end):
                    frequencies[i] /= energy
            spectra.append(frequencies)
        storage = max(2,min(1275,(self.bitrate * frame_size + 383999) // 384000 - 1))
        encoder = OpusRangeEncoder(storage)
        encoder.bit_logp(int(not signal),15)
        toc = ({13:0x80,17:0xa0,19:0xc0,21:0xe0}[self.end]) | (lm << 3) | (4 if self.channels == 2 else 0)
        if not signal:
            payload = encoder.done()
            self.old_energy[:] = [-28.] * 42
            for channel in range(2):
                self.old_energy[channel*21+self.end:channel*21+21] = [0.] * (21-self.end)
            self.rng,self.frame_count = encoder.rng,self.frame_count + 1
            return pack_frames(toc,(payload,))
        budget = storage * 8
        if encoder.tell() + 16 <= budget:
            encoder.bit_logp(0,1)
        if lm > 0 and encoder.tell() + 3 <= budget:
            encoder.bit_logp(transient,3)
        else:
            transient = 0
        intra = int(self.frame_count == 0)
        if encoder.tell() + 3 <= budget:
            encoder.bit_logp(intra,3)
        else:
            intra = 0
        self._coarse(target,encoder,lm,intra)
        logp = 2 if transient else 4
        reserve = int(lm > 0 and encoder.tell() + logp + 1 <= budget)
        tf,current,changed = [0] * 21,0,0
        for band in range(self.end):
            if encoder.tell() + logp <= budget - reserve:
                target_flag = int(bool(transient))
                encoder.bit_logp(current ^ target_flag,logp)
                current = target_flag
                changed |= current
            tf[band] = _TF[lm][4 * transient + current]
            logp = 4 if transient else 5
        if reserve and _TF[lm][4 * transient + changed] != _TF[lm][4 * transient + 2 + changed]:
            encoder.bit_logp(0,1)
        if encoder.tell() + 4 <= budget:
            encoder.icdf(2,(25,23,2,0),5)
        tell = encoder.tell_frac()
        for _ in range(self.end):
            if tell + 48 < storage * 64:
                encoder.bit_logp(0,6)
                tell = encoder.tell_frac()
        if tell + 48 <= storage * 64:
            encoder.icdf(5,(126,124,119,109,87,41,19,9,4,2,0),7)
        cap = init_caps(lm,self.channels)
        bits = storage * 64 - encoder.tell_frac() - 1
        collapse_reserve = 8 if transient and lm >= 2 and bits >= (lm + 2) << 3 else 0
        coded,intensity,dual,balance,pulses,fine,priority = allocation(0,self.end,[0]*21,cap,5,bits-collapse_reserve,self.channels,lm,_AllocationWriter(encoder))
        for band in range(self.end):
            if fine[band] <= 0:
                continue
            for channel in range(self.channels):
                index = band + channel * 21
                levels = 1 << fine[band]
                quantized = max(0,min(levels - 1,floor((target[index] - self.old_energy[index] + .5) * levels)))
                encoder.bits(quantized,fine[band])
                self.old_energy[index] += (quantized + .5) / levels - .5
        _BandsWriter(encoder,2).all_bands(spectra,pulses,storage*64-collapse_reserve,balance,lm,coded,dual,intensity,(1 << lm) if transient else 1,tf,self.end)
        if collapse_reserve:
            encoder.bits(0,1)
        bits_left = budget - encoder.tell()
        for pass_priority in (0,1):
            for band in range(self.end):
                if bits_left < self.channels:
                    break
                if fine[band] >= 8 or priority[band] != pass_priority:
                    continue
                for channel in range(self.channels):
                    index = band + channel * 21
                    quantized = int(target[index] >= self.old_energy[index])
                    encoder.bits(quantized,1)
                    self.old_energy[index] += (quantized - .5) / (1 << (fine[band] + 1))
                    bits_left -= 1
        payload = encoder.done()
        self.rng = encoder.rng
        self.frame_count += 1
        if self.channels == 1:
            self.old_energy[21:] = self.old_energy[:21]
        for channel in range(2):
            self.old_energy[channel*21+self.end:channel*21+21] = [0.] * (21-self.end)
        return pack_frames(toc,(payload,))

    def encode(self,pcm,frame_size=960):
        if len(pcm) != frame_size * self.channels * 2:
            raise ValueError('Invalid Opus CELT signed PCM16 byte length')
        values = unpack('<' + 'h' * (len(pcm) // 2),pcm)
        return self.encode_float([value / 32768. for value in values],frame_size)
