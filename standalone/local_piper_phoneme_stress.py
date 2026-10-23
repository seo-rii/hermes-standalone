"""Original language-specific syllable weights and word-stress assignment."""
STRESS_PHONEMES = (3, 2, 4, 5, 6, 7, 26)
NONSYLLABIC = 1 << 20
LONG = 1 << 21

def _previous_primary(levels,count,stressed_syllable,maximum,primary):
    index=count-1
    while index>0 and stressed_syllable==0 and levels[index]<4:
        if levels[index] not in (0,1):
            levels[index]=4
            if maximum<4:maximum=4;primary=index
            for earlier in range(1,index):
                if levels[earlier]==4:levels[earlier]=3
            break
        index-=1
    return maximum,primary

def _vowel_priority(tr,levels,count,stressed_syllable,maximum,primary):
    if stressed_syllable>0:
        if stressed_syllable>=count:stressed_syllable=count-1
        levels[stressed_syllable]=4;maximum=4;primary=stressed_syllable
    if maximum==5:
        for index in range(1,count):
            if levels[index]==4:levels[index]=1 if tr.langopts.stress_flags&0x20000 else 3
            if levels[index]==5:levels[index]=4;primary=index
        maximum=4
    return primary,maximum

def vowel_stress(tr, phonemes, stressed_syllable=0, control=0):
    table = tr.phoneme_tab
    output = bytearray()
    levels = [0] * 100
    levels[0] = 1
    count = 1
    maximum = -1
    stress = -1
    primary = 0
    for code in bytes(phonemes).split(b'\x00', 1)[0]:
        if count >= 99:
            break
        phone = table[code]
        if phone is None:
            continue
        if phone.type == 1 and phone.program == 0:
            if code == 8:
                maximum,primary=_previous_primary(levels,count,stressed_syllable,maximum,primary)
            elif phone.std_length < 4 or stressed_syllable == 0:
                stress = phone.std_length
                if stress > maximum:
                    maximum = stress
            continue
        if phone.type == 2 and (not phone.phflags & NONSYLLABIC):
            levels[count] = stress
            if stress >= 4 and stress >= maximum:
                primary = count
                maximum = stress
            if stress < 0 and control & 1 and phone.phflags & 2:
                levels[count] = 1
            count += 1
            stress = -1
        elif code == 20:
            levels[count] = stress
            if stress < 0 and control & 1:
                levels[count] = 1
            count += 1
        output.append(code)
    levels[count] = 1
    primary,maximum=_vowel_priority(tr,levels,count,stressed_syllable,maximum,primary)
    return (bytes(output), levels, count, primary, maximum)

class _WordStress:

    def __init__(self, tr, phonemes, dictionary_flags, tonic, control):
        self.tr = tr
        self.phonemes = phonemes
        self.dictionary_flags = dictionary_flags
        self.tonic = tonic
        self.control = control

    def prepare(self):
        self.table = self.tr.phoneme_tab
        self.options = self.tr.langopts
        self.stressflags = self.options.stress_flags
        self.dflags = 0 if self.dictionary_flags is None else self.dictionary_flags[0]
        self.phonemes = bytes((13 if code >= self.tr.n_phoneme_tab else code for code in bytes(self.phonemes).split(b'\x00', 1)[0][:200]))
        if not self.phonemes:
            return False
        self.final = self.table[self.phonemes[-1]]
        self.previous = self.table[self.phonemes[-2] if len(self.phonemes) > 1 else self.phonemes[-1]]
        self.unstressed_word = bool(self.dflags & 8)
        self.stressed = self.dflags & (3 if self.unstressed_word else 7)
        self.phonemes, self.levels, self.count, self.stressed, self.maximum = vowel_stress(self.tr, self.phonemes, self.stressed, 1)
        self.input_maximum = self.maximum
        if self.maximum < 0 and self.dictionary_flags is not None:
            self.maximum = 0
        return True

    def syllable_weights(self):
        self.weights = [0] * 100
        self.lengths = [0] * 100
        index = 1
        position = 0
        self.padded = self.phonemes + b'\x00\x00\x00'
        while position < len(self.phonemes):
            phone = self.table[self.padded[position]]
            if phone.type == 2 and (not phone.phflags & NONSYLLABIC):
                lengthened = self.table[self.padded[position + 1]].code == 12
                weight = int(lengthened or bool(phone.phflags & LONG))
                self.lengths[index] = weight
                if lengthened:
                    position += 1
                next_phone = self.table[self.padded[position + 1]]
                if 3 <= next_phone.type <= 9 and (self.table[self.padded[position + 2]].type != 2 or next_phone.phflags & LONG):
                    weight += 1
                self.weights[index] = weight
                index += 1
            position += 1
        self.rule = self.options.stress_rule

    def second(self):
        if (self.rule != 8 or (self.weights[1] == 0 and self.weights[2] > 0)) and self.stressed == 0 and (self.count > 2):
            self.stressed = 2
            if self.maximum == 0:
                self.levels[self.stressed] = 4
            self.maximum = 4

    def penultimate(self):
        self.maximum = 4
        if self.count > 2:
            self.stressed = self.count - 2
            if self.stressflags & 512 and self.final.type != 2:
                mnemonic = self.final.mnemonic
                language = self.tr.translator_name
                if language in (24942, 25441):
                    if mnemonic not in (115, 110) or self.previous.type != 2:
                        self.stressed = self.count - 1
                elif language == 26977:
                    if mnemonic != 115 or self.previous.type != 2:
                        self.stressed = self.count - 1
                elif not (mnemonic == 115 and self.previous.type == 8):
                    if self.final.type != 8 and mnemonic != 115 or self.previous.type != 2:
                        self.stressed = self.count - 1
            if self.stressflags & 524288 and self.lengths[self.count - 1] > self.lengths[self.count - 2]:
                self.stressed = self.count - 1
            if self.levels[self.stressed] in (0, 1):
                self.stressed += -1 if self.stressed > 1 else 1
        else:
            self.stressed = 1
        if self.levels[self.stressed] < 0 and (self.levels[self.stressed - 1] < 4 or self.levels[self.stressed + 1] < 4):
            self.levels[self.stressed] = self.maximum

    def last_unstressed(self):
        self.stressed = self.count - 1
        while self.stressed > 0:
            if self.levels[self.stressed] < 0:
                self.levels[self.stressed] = 4
                break
            self.stressed -= 1
        self.maximum = 4

    def antepenultimate(self):
        self.stressed = max(1, self.count - 3)
        if self.maximum == 0:
            self.levels[self.stressed] = 4
        self.maximum = 4

    def guessed(self):
        self.stressed = self.count - 3
        if self.count < 16:
            guesses = ((0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11), (0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10), (0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10))
            self.stressed = guesses[1 if self.final.type == 2 else 2 if self.final.type == 4 else 0][self.count]
        self.levels[self.stressed] = 4
        self.maximum = 4

    def heaviest(self):
        max_weight = -1
        for index in range(1, self.count - 1):
            if self.levels[index] < 0 and self.weights[index] >= max_weight:
                max_weight = self.weights[index]
                self.stressed = index
        if self.weights[self.count - 1] == 2 and max_weight < 2:
            self.stressed = self.count - 1
        elif max_weight <= 0:
            self.stressed = 1
        self.levels[self.stressed] = 4
        self.maximum = 4

    def before_unstressed(self):
        self.stressed = self.count - 1
        for index in range(1, self.count):
            if self.levels[index] == 1:
                self.stressed = index - 1
                break
        self.levels[self.stressed] = 4
        self.maximum = 4

    def all_syllables(self):
        for index in range(1, self.count):
            if self.levels[index] < 0:
                self.levels[index] = 4

    def last_long(self):
        long_vowel = 0
        for index in range(1, self.count):
            if self.levels[index] == 4:
                self.levels[index] = 3
            if self.lengths[index] > 0:
                long_vowel = index
                self.levels[index] = 3
        if self.stressed == 0:
            self.stressed = long_vowel or (self.count - 3 if self.count > 5 else self.count - 1)
        self.levels[self.stressed] = 4
        self.maximum = 4

    def first_heavy(self):
        self.stressed = 2 if self.lengths[1] == 0 and self.count > 2 and (self.lengths[2] > 0) else 1
        self.levels[self.stressed] = 4
        self.maximum = 4

    def second_with_final_secondary(self):
        for index in range(1, self.count):
            self.levels[index] = 0
        self.stressed = 2
        if self.maximum == 0:
            self.levels[self.stressed] = 4
        self.maximum = 4
        if self.count > 3:
            self.levels[self.count - 1] = 3

    def language_rule(self):
        if self.rule in (1, 8):
            self.second()
        elif self.rule == 2 and self.stressed == 0:
            self.penultimate()
        elif self.rule == 3 and self.stressed == 0:
            self.last_unstressed()
        elif self.rule == 4 and self.stressed == 0:
            self.antepenultimate()
        elif self.rule == 5 and self.stressed == 0:
            self.guessed()
        elif self.rule == 6 and self.stressed == 0:
            self.heaviest()
        elif self.rule == 7 and self.stressed == 0:
            self.before_unstressed()
        elif self.rule == 9:
            self.all_syllables()
        elif self.rule == 12:
            self.last_long()
        elif self.rule == 13 and self.stressed == 0:
            self.first_heavy()
        elif self.rule == 15 and self.stressed == 0 and (self.count > 2):
            self.second_with_final_secondary()

    def end_and_secondary(self):
        if self.stressflags & 256 and (not self.control & 2) and (self.count > 2) and (self.input_maximum < 3) and (self.levels[self.count - 1] == 4) and (self.final.type == 2):
            self.levels[self.count - 1] = 1
            self.levels[self.count - 2] = 4
        stress = 4 if self.maximum < 4 else 3
        if not self.unstressed_word:
            if self.stressflags & 4096 and self.count == 3:
                if self.levels[1] == 4:
                    self.levels[2] = 3
                if self.levels[2] == 4:
                    self.levels[1] = 3
            if self.stressflags & 8192 and self.levels[1] < 0 and (self.count > 3) and (self.levels[2] >= 4):
                self.levels[1] = 3
        done = False
        first_primary = 0
        for index in range(1, self.count):
            if self.levels[index] < 0:
                if self.stressflags & 16 and stress < 4 and (index == self.count - 1):
                    pass
                elif self.stressflags & 32768 and (not done):
                    self.levels[index] = stress
                    done = True
                    stress = 3
                elif self.levels[index - 1] <= 1 and (self.levels[index + 1] <= 1 or (stress == 4 and self.levels[index + 1] <= 2)):
                    if stress == 3 and self.stressflags & 32:
                        continue
                    if index > 1 and self.stressflags & 64 and (self.weights[index] == 0):
                        if any((self.weights[later] > 0 for later in range(index, self.count - 1))):
                            continue
                        if self.weights[index + 1] > 0:
                            continue
                    self.levels[index] = stress
                    done = True
                    stress = 3
            if self.levels[index] >= 4:
                if first_primary == 0:
                    first_primary = index
                elif self.stressflags & 128:
                    self.levels[index] = 3

    def tonic_stress(self):
        if self.unstressed_word and self.tonic < 0:
            self.tonic = self.options.unstressed_wd1 if self.count <= 2 else self.options.unstressed_wd2
        self.maximum = 0
        self.max_position = 0
        for index in range(1, self.count):
            if self.levels[index] >= self.maximum:
                self.maximum = self.levels[index]
                self.max_position = index
        if self.tonic >= 0:
            if self.tonic > self.maximum or self.maximum <= 4:
                self.levels[self.max_position] = self.tonic
            self.maximum = self.tonic

    def initial_vowel_pause(self):
        position = 0
        if not self.control & 1 and self.table[self.padded[0]] is not None:
            while self.table[self.padded[position]].type == 1 or self.padded[position] == 15:
                position += 1
            if self.options.vowel_pause & 48 and self.table[self.padded[position]].type == 2:
                self.output.append(11 if self.options.vowel_pause & 32 and self.levels[1] >= 4 else 23)

    def emit(self):
        self.output = bytearray()
        position = 0
        vowel = 1
        self.initial_vowel_pause()
        position = 0
        while position < len(self.phonemes) and len(self.output) < 197:
            code = self.padded[position]
            position += 1
            phone = self.table[code]
            if phone is None:
                continue
            if phone.type == 0:
                self.tr.prev_last_stress = 0
            elif phone.type == 2 and (not phone.phflags & NONSYLLABIC) or self.padded[position] == 20:
                stress = self.levels[vowel]
                self.tr.prev_last_stress = stress
                if stress <= 1:
                    if vowel > 1 and self.maximum >= 2 and self.stressflags & 4 and (vowel == self.count - 1):
                        stress = 0
                    elif self.stressflags & 2 or vowel == 1 or vowel == self.count - 1:
                        stress = 1
                    elif vowel == self.count - 2 and self.levels[self.count - 1] <= 1:
                        stress = 1
                    elif self.levels[vowel - 1] < 0 or not self.stressflags & 65536:
                        stress = 0
                        self.levels[vowel] = stress
                if stress == 0 or stress > 1:
                    self.output.append(STRESS_PHONEMES[stress])
                if self.levels[vowel] > self.maximum:
                    self.maximum = self.levels[vowel]
                length_option = self.options.param[1]
                if self.padded[position] == 12 and length_option & 1:
                    if vowel != self.max_position if length_option & 16 else stress < 4:
                        position += 1
                vowel += 1
            if code != 1:
                self.output.append(code)
        return bytes(self.output)

def set_word_stress(tr, phonemes, dictionary_flags=None, tonic=-1, control=0):
    work = _WordStress(tr, phonemes, dictionary_flags, tonic, control)
    if not work.prepare():
        return work.phonemes
    work.syllable_weights()
    work.language_rule()
    work.end_and_secondary()
    work.tonic_stress()
    return work.emit()
