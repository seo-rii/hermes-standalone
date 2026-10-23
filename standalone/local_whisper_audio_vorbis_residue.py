"""Source-order Vorbis residue0/1/2 entropy and channel deinterleaving."""
from standalone.local_whisper_audio_aac_dsp import f32
from .local_whisper_audio_vorbis_bits import VorbisError


class Residue:
    def __init__(self, bits, books):
        self.kind = bits.read(16)
        self.begin, self.end, self.size = bits.read(24), bits.read(24), bits.read(24) + 1
        if self.begin > self.end or (self.end - self.begin) // self.size > 65535:
            raise VorbisError('Invalid Vorbis residue partition range')
        self.classifications = bits.read(6) + 1
        self.classbook = books[bits.index(8, len(books))]
        self.books = books
        cascades = []
        for _ in range(self.classifications):
            low = bits.read(3)
            high = bits.read(5) if bits.read(1) else 0
            cascades.append(low | (high << 3))
        self.passes = max((cascade.bit_length() for cascade in cascades), default=0)
        self.stages = [[bits.index(8, len(books)) if cascade & (1 << stage) else -1
                        for stage in range(8)] for cascade in cascades]

    def decode(self, bits, values, start, length, channels, skipped, channels_left):
        if self.kind > 2:
            raise VorbisError('Invalid Vorbis residue type')
        count = (self.end - self.begin) // self.size
        if self.kind == 2:
            skipped = [all(skipped)]
            if skipped[0]:
                return
            used, maximum = 1, (channels - 1) * length + self.end // channels
        else:
            used, maximum = channels, (channels - 1) * length + self.end
        extra = maximum > channels_left * length
        if extra:
            if maximum > channels_left * length + self.size * used // channels:
                raise VorbisError('Vorbis residue exceeds the output buffer')
            count -= 1
        classifications = [[0] * count for _ in range(used)]
        dimensions = self.classbook.dimensions
        for stage in range(max(1, self.passes)):
            self._pass(bits, values, start, length, channels, skipped, count, used, dimensions, classifications, stage)
            if extra and not stage:
                for ch in range(used):
                    if not skipped[ch]:
                        self.classbook.read(bits)

    def _classify(self, bits, skipped, used, partition, dimensions, count, classifications):
        for ch in range(used):
            if skipped[ch]:
                continue
            temp = self.classbook.read(bits)
            for index in range(partition + dimensions - 1, partition - 1, -1):
                temp, classification = divmod(temp, self.classifications)
                if index < count:
                    classifications[ch][index] = classification

    def _pass(self, bits, values, start, length, channels, skipped, count, used, dimensions, classifications, stage):
        partition, voffset = 0, self.begin
        while partition < count:
            if stage == 0:
                self._classify(bits, skipped, used, partition, dimensions, count, classifications)
            for _ in range(min(dimensions, count - partition)):
                for ch in range(used):
                    if skipped[ch]:
                        continue
                    book_index = self.stages[classifications[ch][partition]][stage]
                    if book_index < 0 or self.books[book_index].vectors is None:
                        continue
                    self._add_vectors(bits, values, start, length, channels, ch, voffset, self.books[book_index])
                partition += 1
                voffset += self.size

    def _add_vectors(self, bits, values, start, length, channels, ch, voffset, book):
        step = self.size // book.dimensions
        if self.kind == 0:
            for k in range(step):
                for component, value in enumerate(book.vector(bits)):
                    offset = start + voffset + ch * length + k + component * step
                    values[offset] = f32(values[offset] + value)
        elif self.kind == 1:
            offset = start + voffset + ch * length
            for _ in range(step):
                for value in book.vector(bits):
                    values[offset] = f32(values[offset] + value)
                    offset += 1
        else:
            index = voffset
            for _ in range(step):
                for value in book.vector(bits):
                    column, mapped = divmod(index, channels)
                    offset = start + column + mapped * length
                    values[offset] = f32(values[offset] + value)
                    index += 1
