"""Independent original eSpeak clause/token/word/context observations."""
import base64, hashlib, importlib.abc, json, pathlib, struct, sys, types, unittest, zlib
from contextlib import contextmanager
from standalone import local_piper_phoneme_unicode as u
from standalone.local_piper_phoneme_text import ClauseController, FirstItem
from standalone.local_piper_phoneme_text_tokenizer import Tokenizer, WordTab
from standalone.local_piper_phoneme_text_characters import Characters
from standalone.local_piper_phoneme_words import WordBuilder
from standalone.local_piper_phoneme_session import Session
from standalone.local_piper_phoneme_clause import ClauseReader
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / 'assets/piper/espeak-ng-data'
PACK = struct.Struct('<HBBHBB')

class NoNative(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'ctypes', 'subprocess', 'numpy', 'piper', 'espeakbridge', 'onnxruntime', 'torch'}:
            raise AssertionError('Native dependency requested by qualified clause port ' + fullname)

def restore(packed):
    values = []
    for kind, payload in packed['nodes']:
        if kind == 0:
            value = payload
        elif kind == 1:
            value = [values[index] for index in payload]
        else:
            value = {key: values[index] for key, index in payload}
        values.append(value)
    return {name: values[index] for name, index in packed['documents'].items()}

def cbytes(buffer, cursor, prefix=0):
    return bytes(buffer[cursor - prefix:buffer.index(0, cursor) + 1])

def tr_set(tr, values):
    for key, value in values.items():
        if key == 'prev_dict_flags':
            value = [(v + 2147483648) % 4294967296 - 2147483648 for v in value]
        if key == 'phonemes_repeat':
            value = bytes.fromhex(value)
        setattr(tr, key, value)

def state_from(value):
    state = types.SimpleNamespace(**{k: v for k, v in value.items() if k not in ('translator', 'ph_list2_raw')})
    state.word_phonemes = bytes.fromhex(state.word_phonemes)
    state.embedded_list = list(state.embedded_list) + [0] * (250 - len(state.embedded_list))
    items = [FirstItem(*parts) for parts in PACK.iter_unpack(bytes.fromhex(value['ph_list2_raw']))]
    state.ph_list2 = items + [FirstItem() for _ in range(1000 - len(items))]
    state.dictionary_name = 'en'
    return state

def compare_state(actual, tr, expected, context):
    for key, wanted in expected.items():
        if key == 'translator':
            for field, value in wanted.items():
                result = getattr(tr, field)
                if field == 'prev_dict_flags':
                    result = [v & 4294967295 for v in result]
                if field == 'phonemes_repeat':
                    result = result.hex()
                assert result == value, (context, 'translator', field, result, value)
            continue
        if key == 'ph_list2_raw':
            value = b''.join((PACK.pack(*(getattr(p, n) for n in ('synthflags', 'phcode', 'stresslevel', 'sourceix', 'wordstress', 'tone_ph'))) for p in actual.ph_list2[:actual.n_ph_list2])).hex()
        elif key == 'embedded_list':
            value = actual.embedded_list[:actual.embedded_ix]
        elif key == 'word_phonemes':
            value = actual.word_phonemes.hex()
        else:
            value = getattr(actual, key)
        assert value == wanted, (context, key, str(value)[:500], str(wanted)[:500])

class Tape:

    @contextmanager
    def outside_clause(self):
        yield

    def next(self, kind):
        event = self.events.pop(0)
        assert event['kind'] == kind, (self.context, kind, event['kind'])
        return event

    def select_event(self, event):
        self.phones = tables[event['selection']]
        self.table_number = event['table_number']

    def select(self, number):
        event = self.next(10)
        assert event['first'] == number, (self.context, number, event['first'])
        self.select_event(event)

    def set_translator2(self, language):
        before = self.next(8)
        assert bytes.fromhex(before['text']) == language.encode()
        compare_state(self.state, self.primary, before['state'], self.context)
        after = self.next(9)
        self.select_event(after)
        following = next((e for e in self.events if e['kind'] == 6 and e['translator_slot'] == 2), None)
        self.secondary = Translator(self)
        if following:
            tr_set(self.secondary, following['state']['translator'])
        return after['first']

    def number(self, language):
        return self.catalog[language]

    def translate_word(self, tr, buffer, cursor, words, index, *, word_out):
        before = self.next(6)
        compare_state(self.state, tr, before['state'], self.context)
        assert before['first'] == int(word_out) and before['buffer'] == cbytes(buffer, cursor, 1).hex(), (self.context, 'word_input', word_out, cbytes(buffer, cursor, 1), before['buffer'])
        assert before['words'] == [vars(w) for w in words[index:index + 2]], (self.context, 'word_tables', before['words'], [vars(w) for w in words[index:index + 2]])
        after = self.next(7)
        self.select_event(after)
        raw = bytes.fromhex(after['buffer'])
        buffer[cursor - 1:cursor - 1 + len(raw)] = raw
        for item, value in zip(words[index:index + 2], after['words']):
            vars(item).update(value)
        for key, value in after['state'].items():
            if key in ('translator', 'ph_list2_raw') or value == before['state'][key]:
                continue
            if key == 'embedded_list':
                self.state.embedded_list[:len(value)] = value
            elif key == 'word_phonemes':
                self.state.word_phonemes = bytes.fromhex(value)
            else:
                setattr(self.state, key, value)
        tr_set(tr, after['state']['translator'])
        return types.SimpleNamespace(flags=after['second'], phonemes=bytes.fromhex(after['state']['word_phonemes']), replacement=None if after['replacement'] is None else bytes.fromhex(after['replacement']))

class Translator:

    def __init__(self, session):
        self.session = session

    @property
    def phoneme_tab(self):
        return self.session.phones

    def encode_phonemes(self, buffer, cursor):
        before = self.session.next(11)
        assert bytes.fromhex(before['text']) == cbytes(buffer, cursor)[:-1]
        after = self.session.next(12)
        return types.SimpleNamespace(phonemes=bytes.fromhex(after['state']['word_phonemes']))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def rows(sequence, keys):
    output = []
    for item in sequence:
        row = {key: getattr(item, key) for key in keys if key != 'ph'}
        row['ph'] = None if item.ph is None else vars(item.ph)
        output.append(row)
    return output

class OriginalTextTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wrapper = json.loads((ROOT / 'fixtures/local_piper_phoneme_text.json').read_text())
        raw = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64']))
        assert hashlib.sha256(raw).hexdigest() == wrapper['data_sha256']
        cls.packed = json.loads(raw)
        cls.documents = restore(cls.packed)

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def _retained_value(self, state, tr, controller, key, value, row_keys):
        if key == 'phoneme_list':
            value2 = rows(state.phoneme_list, row_keys)
        elif key == 'ph_list2_raw':
            value2 = b''.join((PACK.pack(*(getattr(item, k) for k in ('synthflags', 'phcode', 'stresslevel', 'sourceix', 'wordstress', 'tone_ph'))) for item in state.ph_list2[:state.n_ph_list2])).hex()
        elif key == 'word_phonemes':
            value2 = state.word_phonemes.hex()
        elif key == 'embedded_list':
            value2 = state.embedded_list[:state.embedded_ix]
        elif key == 'translator':
            value2 = {field: getattr(tr, field).hex() if field == 'phonemes_repeat' else [v & 4294967295 for v in tr.prev_dict_flags] if field == 'prev_dict_flags' else getattr(tr, field) for field in value}
        elif key in ('ungot_char', 'ungot_char2'):
            value2 = getattr(controller.reader, key)
        else:
            value2 = getattr(state, key)
        return value2

    def test_original_516_utf8_tokenizer_and_embedded_contexts(self):
        fixture = self.documents['tokenizer']
        count = 0
        fields = ('pre_pause', 'option_sayas2', 'count_sayas_digits', 'word_emphasis', 'embedded_ix', 'embedded_list')
        for voice in fixture['voices']:
            for case in voice['cases']:
                before, expected = case['events'][:2]
                assert before['kind'] == 0 and expected['kind'] == 1
                config = case['configuration']
                tr = types.SimpleNamespace(**{k: v for k, v in config.items() if k != 'langopts'})
                tr.langopts = types.SimpleNamespace(**config['langopts'])
                tr.langopts.replace_chars = [tuple((bytes.fromhex(x) for x in pair)) for pair in tr.langopts.replace_chars]
                tr.is_letter = lambda cp, group: cp in config['vowel2']
                state = types.SimpleNamespace(**before['state'])
                state.embedded_list = list(state.embedded_list) + [0] * (250 - len(state.embedded_list))
                charix = before['charix'] + [0] * 4
                top = before['charix_top']
                charix[top + 1:top + 4] = [0, 32767, 0]
                result = Tokenizer(state, Characters()).tokenize(tr, bytearray.fromhex(before['source']), charix, top, before['second'])
                actual = dict(buffer=result.buffer[:result.ix + 1].hex(), words=[vars(w) for w in result.words[:result.word_count + 1]], word_count=result.word_count, ix=result.ix)
                wanted = dict(buffer=expected['buffer'], words=expected['words'], word_count=expected['first'], ix=expected['second'])
                assert actual == wanted, (voice['voice'], case['text'][:120], case['options'], {key: (str(value)[:600], str(wanted[key])[:600]) for key, value in actual.items() if value != wanted[key]})
                for field in fields:
                    value = getattr(state, field)
                    value = value[:state.embedded_ix] if field == 'embedded_list' else value
                    assert value == expected['state'][field], (voice['voice'], case['text'][:120], field, value, expected['state'][field])
                count += 1
        self.assertEqual(count, 516)

    def test_original_6093_word2_states_and_every_dependency_input(self):
        global tables
        fixture = self.documents['words']
        catalog = self.documents['catalog']
        count = 0
        tables = [[None if p is None else types.SimpleNamespace(**p) for p in table] for table in fixture['phone_tables']]
        for voice in fixture['voices']:
            for case in voice['cases']:
                events = list(case['events'])
                tokenized = events[1]
                position = 0
                while position < len(events):
                    before = events[position]
                    position += 1
                    if before['kind'] != 2:
                        continue
                    nested = []
                    while events[position]['kind'] != 3:
                        nested.append(events[position])
                        position += 1
                    expected = events[position]
                    position += 1
                    tape = Tape()
                    tape.events = nested
                    tape.context = (voice['voice'], case['text'][:80], count)
                    tape.state = state_from(before['state'])
                    tape.primary = Translator(tape)
                    tape.catalog = catalog
                    tape.data = tape
                    tape.selection = tape
                    tape.unicode = u
                    tape.voice = types.SimpleNamespace(phoneme_tab_ix=case['configuration']['phoneme_tab_ix'])
                    tape.select_event(before)
                    tr = tape.primary
                    tr_set(tr, before['state']['translator'])
                    tr.langopts = types.SimpleNamespace(**case['configuration']['langopts'])
                    tr.translator_name = case['configuration']['translator_name']
                    previous = next((tokenized['words'][i - 1]['flags'] for i, w in enumerate(tokenized['words']) if i > 0 and w['start'] == before['words'][0]['start']), 0)
                    words = [WordTab(flags=previous)] + [WordTab(**w) for w in before['words']]
                    buffer = bytearray.fromhex(before['buffer']) + bytearray(20)
                    controller = ClauseController(tape.state)
                    result = WordBuilder(controller, tape.translate_word).translate(tr, buffer, 3, words, 1, before['first'])
                    assert result == expected['second'] and cbytes(buffer, 3, 3).hex() == expected['buffer'] and ([vars(w) for w in words[1:]] == expected['words']), (tape.context, 'result', result, expected['second'], cbytes(buffer, 3, 3).hex(), expected['buffer'], [vars(w) for w in words[1:]], expected['words'])
                    compare_state(tape.state, tr, expected['state'], tape.context)
                    assert not tape.events, (tape.context, 'remaining', len(tape.events))
                    count += 1
        self.assertEqual(count, 6093)

    def test_original_516_whole_clauses_with_real_dictionary_word_and_vm(self):
        fixture = self.documents['words']
        count = 0
        selection = set()
        for voice in fixture['voices']:
            if selection and voice['voice'] not in selection:
                continue
            for case in voice['cases']:
                session = Session(DATA)
                tr = session.select_voice(voice['voice'])
                state = session.state
                controller = ClauseController.from_session(session, reader=ClauseReader(case['text']))
                for key, value in case['options'].items():
                    if key in ('option_phoneme_input', 'option_capitals', 'option_punctuation'):
                        setattr(controller.reader, key, value)
                    else:
                        setattr(state, key, value)
                result = controller.translate_clause(tr)
                expected = case['expected']
                actual = []
                for item in state.phoneme_list:
                    row = {key: getattr(item, key) for key in expected['state']['phoneme_list'][len(actual)] if key != 'ph'}
                    row['ph'] = None if item.ph is None else vars(item.ph)
                    actual.append(row)
                differences = {}
                for key, value in [('tone', result.tone), ('terminator', result.terminator), ('voice_change', None if result.voice_change is None else result.voice_change.hex()), ('phoneme_list', actual)]:
                    wanted = expected['state'][key] if key == 'phoneme_list' else expected[key]
                    if value != wanted:
                        differences[key] = (str(value)[:500], str(wanted)[:500])
                if differences:
                    raise AssertionError(differences)
                count += 1
        self.assertEqual(count, 516)

    def test_original_288_retained_clauses_modes_1001_slots_and_256_phones(self):
        fixture = self.documents['retained']
        count = 0
        session = Session(DATA)
        for voice in fixture['voices']:
            state = session.state
            fresh = type(state)()
            vars(state).clear()
            vars(state).update(vars(fresh))
            tr = session.select_voice(voice['voice'])
            tr.option_sayas = 0
            for target in (session.secondary, session.tertiary):
                if target is None:
                    continue
                for field in fixture['voices'][0]['steps'][0]['clauses'][0]['state']['translator']:
                    setattr(target, field, [0, 0] if field == 'prev_dict_flags' else b'' if field == 'phonemes_repeat' else 0)
            controller = ClauseController.from_session(session, reader=ClauseReader())
            for step in voice['steps']:
                if step['revoice']:
                    tr = session.select_voice(voice['voice'])
                controller.reader.option_phoneme_input = 1
                controller.reader.bind_text(step['text'])
                for index, wanted in enumerate(step['clauses']):
                    context = (voice['voice'], step['text'][:60], index)
                    actual = controller.translate_clause(tr)
                    result = dict(tone=actual.tone, terminator=actual.terminator, voice_change=None if actual.voice_change is None else actual.voice_change.hex(), cursor=None if controller.reader.position == len(controller.reader.text) else controller.reader.position)
                    for key, value in result.items():
                        assert value == wanted[key], (context, key, value, wanted[key])
                    expected = wanted['state']
                    row_keys = expected['phoneme_list'][0].keys()
                    for key, value in expected.items():
                        value2 = self._retained_value(state, tr, controller, key, value, row_keys)
                        assert value2 == value, (context, key, str(value2)[:450], str(value)[:450])
                    assert digest(rows(state.phoneme_buffer, row_keys)) == wanted['output_1001_sha256'], (context, 'retained-output1001')
                    assert digest([None if p is None else vars(p) for p in session.selection.phones]) == wanted['active_table_sha256'], (context, 'retained-active-map')
                    count += 1
        self.assertEqual(count, 288)
        self.assertEqual(count, fixture['clause_count'])

    def test_lossless_fixture_structure_restores_original_documents(self):
        self.assertEqual(self.packed['schema'], 1)
        for name, value in self.documents.items():
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(len(raw), self.packed['source_documents'][name]['canonical_bytes'], name)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), self.packed['source_documents'][name]['canonical_sha256'], name)
if __name__ == '__main__':
    unittest.main(verbosity=2)
