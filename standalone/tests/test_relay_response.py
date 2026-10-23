"""Original selected codec response properties, including partial tool calls."""
import json
from pathlib import Path
import unittest

from standalone import relay_codecs


class RelayResponseTests(unittest.TestCase):
    def test_native_provider_response_fields_extensions_finish_and_type_refusals(self):
        fixture=json.loads((Path(__file__).with_name('fixtures')/'relay_response.json').read_text())
        self.assertGreater(len(fixture['cases']),150)
        codecs={'chat':relay_codecs.OpenAIChatCodec(),'responses':relay_codecs.OpenAIResponsesCodec(),
                'anthropic':relay_codecs.AnthropicMessagesCodec()}
        for index,case in enumerate(fixture['cases']):
            with self.subTest(case=index,mode=case['mode']):
                if case.get('refused'):
                    with self.assertRaises((ValueError,TypeError)):
                        codecs[case['mode']].decode_response(case['body'])
                else:
                    response=codecs[case['mode']].decode_response(case['body'])
                    self.assertEqual({key:getattr(response,key) for key in case['attributes']},case['attributes'])


if __name__=='__main__':
    unittest.main()
