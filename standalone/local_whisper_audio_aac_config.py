"""AudioSpecificConfig/PCE, native SBR/PS policy and channel routing (LGPL)."""
from dataclasses import dataclass
from .local_whisper_audio_aac_bits import Bits, AacError, UnsupportedAac

RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
# Element types: SCE=0, CPE=1, CCE=2, LFE=3. Position groups follow the
# source AAC_CHANNEL_FRONT/SIDE/BACK/LFE/CC enum.
_DEFAULTS = {
    1: ((0, 0, 1),), 2: ((1, 0, 1),),
    3: ((0, 0, 1), (1, 0, 1)),
    4: ((0, 0, 1), (1, 0, 1), (0, 1, 3)),
    5: ((0, 0, 1), (1, 0, 1), (1, 1, 3)),
    6: ((0, 0, 1), (1, 0, 1), (1, 1, 3), (3, 0, 4)),
    7: ((0, 0, 1), (1, 0, 1), (1, 1, 3), (1, 2, 3), (3, 0, 4)),
    11: ((0, 0, 1), (1, 0, 1), (1, 1, 3), (0, 1, 3), (3, 0, 4)),
    12: ((0, 0, 1), (1, 0, 1), (1, 1, 3), (1, 2, 3), (3, 0, 4)),
    13: ((0, 0, 1), (1, 0, 1), (1, 1, 1), (1, 2, 3), (1, 3, 3), (0, 1, 3),
         (3, 0, 4), (3, 1, 4), (0, 2, 1), (1, 4, 1), (1, 5, 2), (0, 3, 2),
         (1, 6, 3), (0, 4, 3), (0, 5, 1), (1, 7, 1)),
    14: ((0, 0, 1), (1, 0, 1), (1, 1, 3), (3, 0, 4), (1, 2, 1)),
}
_POSITIONS = (
    ((2, 6, 7, 0, 1, None), (-2, None, None, None, None, None),
     (-2, 9, 10, 4, 5, 8), (3, 35, None, None, None, None)),
    ((13, None, None, 12, 14, None), (-2, 36, 37, None, None, 11),
     (-2, None, None, 15, 17, 16), (None,) * 6),
    ((38, None, None, 39, 40, None), (None,) * 6, (None,) * 6, (None,) * 6),
)


@dataclass
class Config:
    object_type: int
    sampling_index: int
    channel_config: int
    frame_length: int = 1024
    layout_map: tuple = ()
    sbr: int = -1
    ps: int = -1
    extension_rate: int = 0

    @property
    def rate(self):
        return RATES[self.sampling_index]

    @property
    def output_rate(self):
        return self.rate << int(self.sbr == 1 and self.extension_rate > self.rate)


def defaults(channel_config):
    try:
        return _DEFAULTS[channel_config]
    except KeyError:
        raise AacError('Invalid AAC default channel configuration') from None


def _count_pairing(mapping, start, position):
    count = parity = first_pair = 0
    for kind, _, group in mapping[start:]:
        if group != position:
            break
        if kind == 1:
            if parity:
                if position == 1 and not first_pair:
                    parity = 0
                else:
                    return -1
            count += 2
            first_pair = 1
        else:
            count += 1
            parity ^= position != 4
    return -1 if parity and position == 1 and first_pair else count


def _assign_group(mapping, start, position, layer, assigned):
    count = _count_pairing(mapping, start, position)
    if count < 0 or count > 5:
        return start, True
    index, j = start, 0
    locations = _POSITIONS[layer][position - 1]
    if position == 4:
        while count:
            if j >= len(locations) or locations[j] is None:
                return start, False
            kind, identity, _ = mapping[index]
            assigned.append((1 << locations[j], kind, identity))
            index, j, count = index + 1, j + 1, count - 1
        return index, True
    if count & 1:
        if locations[0] is None:
            return start, False
        if locations[0] != -2:
            kind, identity, _ = mapping[index]
            assigned.append((1 << locations[0], kind, identity))
            index, count = index + 1, count - 1
    j = 3 if position != 2 and count <= 3 else 1
    while count >= 2:
        if j + 1 >= len(locations) or locations[j] is None or locations[j + 1] is None:
            return start, False
        kind, identity, _ = mapping[index]
        if kind == 1:
            assigned.append(((1 << locations[j]) | (1 << locations[j + 1]), kind, identity))
            index += 1
        else:
            for k in range(2):
                kind, identity, _ = mapping[index]
                assigned.append((1 << locations[j + k], kind, identity))
                index += 1
        j, count = j + 2, count - 2
    if count:
        if locations[5] is None:
            return start, False
        kind, identity, _ = mapping[index]
        assigned.append((1 << locations[5], kind, identity))
        index += 1
    return index, True


def channel_order(mapping):
    """Return the source's ordered element keys and actual channel mask."""
    assigned, current = [], 0
    for layer in range(3):
        for position in range(1, 5):
            current, valid = _assign_group(mapping, current, position, layer, assigned)
            if not valid:
                return tuple((kind, identity) for kind, identity, pos in mapping if pos != 5), None
    if current != sum(group != 5 for _, _, group in mapping):
        return tuple((kind, identity) for kind, identity, pos in mapping if pos != 5), None
    mask = 0
    for bits, _, _ in assigned:
        mask |= bits
    mask_22point2 = sum(1 << i for i in (*range(18), 35, 36, 37, 38, 39, 40))
    if mask == mask_22point2:
        for a, b in ((2, 0), (2, 1), (6, 2), (4, 3), (6, 4), (7, 6), (9, 8), (11, 10), (12, 11)):
            assigned[a], assigned[b] = assigned[b], assigned[a]
    else:
        assigned.sort(key=lambda row: row[0])
    return tuple((kind, identity) for _, kind, identity in assigned), mask


def parse_pce(bits, sampling_index, *, align_reference=0):
    bits.get(2)
    bits.get(4)  # The native decoder warns, but retains the configured rate.
    front, side, back, lfe, associated, coupling = (bits.get(n) for n in (4, 4, 4, 2, 3, 4))
    if bits.get(1):
        bits.skip(4)
    if bits.get(1):
        bits.skip(4)
    if bits.get(1):
        bits.skip(3)
    mapping = []
    for group, count in ((1, front), (2, side), (3, back), (4, lfe)):
        for _ in range(count):
            kind = bits.get(1) if group != 4 else 3
            mapping.append((kind, bits.get(4), group))
    bits.skip(4 * associated)
    for _ in range(coupling):
        bits.get(1)
        mapping.append((2, bits.get(4), 5))
    bits.skip((align_reference - bits.pos) & 7)
    bits.skip(bits.get(8) * 8)
    keys = [(kind, identity) for kind, identity, _ in mapping]
    if len(keys) != len(set(keys)) or not any(group != 5 for _, _, group in mapping):
        raise AacError('Invalid AAC program channel configuration')
    return tuple(mapping)


def _object_type(bits):
    value = bits.get(5)
    return bits.get(6) + 32 if value == 31 else value


def _sampling(bits):
    index = bits.get(4)
    if index == 15:
        bits.get(24)
        # aacdec.c rejects sampling_index > 12 even when an explicit
        # frequency is a canonical LC rate. Retain that actual boundary.
        raise AacError('Explicit AAC sampling frequency index is rejected by the pinned decoder')
    if index >= len(RATES):
        raise AacError('Reserved AAC sampling frequency index')
    return index


def _extension_rate(bits):
    index = bits.get(4)
    if index == 15:
        return bits.get(24)
    if index >= len(RATES):
        raise AacError('Reserved AAC extension sampling frequency')
    return RATES[index]


def parse_config(data):
    bits = Bits(data)
    kind, index, channel_config = _object_type(bits), _sampling(bits), bits.get(4)
    sbr = ps = -1
    extension_rate = 0
    explicit = kind == 5 or (kind == 29 and not (bits.peek(3) & 3 and not bits.peek(9) & 63))
    if explicit:
        ps = 1 if kind == 29 else -1
        sbr, extension_rate, kind = 1, _extension_rate(bits), _object_type(bits)
    if kind != 2:
        raise UnsupportedAac('AAC core object type ' + str(kind) + ' is not AAC-LC')
    # mpeg4audio.c scans all remaining ASC bits, before the GA parser runs.
    if not explicit:
        import copy
        scan = copy.copy(bits)
        while scan.limit - scan.pos > 15:
            if scan.peek(11) == 0x2b7:
                scan.skip(11)
                if _object_type(scan) == 5:
                    sbr = scan.get(1)
                    if sbr:
                        extension_rate = _extension_rate(scan)
                        if extension_rate == RATES[index]:
                            sbr = -1
                if scan.limit - scan.pos > 11 and scan.get(11) == 0x548:
                    ps = scan.get(1)
                break
            scan.skip(1)
    if sbr == 0:
        ps = 0
    frame_length = 960 if bits.get(1) else 1024
    if frame_length == 960 and sbr == 1:
        # The pinned decoder explicitly disables its unimplemented 960 SBR.
        sbr = ps = 0
    if bits.get(1):
        bits.skip(14)
    extension = bits.get(1)
    if channel_config:
        mapping = defaults(channel_config)
    else:
        bits.get(4)
        mapping = parse_pce(bits, index)
    if extension:
        bits.get(1)
    channels = sum(2 if element == 1 else 1 for element, _, group in mapping if group != 5)
    if channels > 1:
        ps = 0
    elif sbr == 1 and ps == -1:
        ps = 1
    return Config(kind, index, channel_config, frame_length, mapping, sbr, ps, extension_rate)
