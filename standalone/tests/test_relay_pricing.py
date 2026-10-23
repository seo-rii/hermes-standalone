"""Pinned native catalog/usage/cost fixtures; no native module is imported."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from standalone import relay_codecs, relay_model_pricing, relay_plugin
from standalone.relay_pricing import PricingResolver, set_active


class RelayPricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((Path(__file__).with_name('fixtures')/'relay_pricing.json').read_text())

    def tearDown(self):
        set_active(None)

    def test_original_catalog_schema_aliases_rates_and_schedules(self):
        self.assertEqual(len(self.fixture['validations']),100)
        with tempfile.TemporaryDirectory(prefix='relay-pricing-validation-') as temporary:
            for case in self.fixture['validations']:
                settings=deepcopy(case['config'])
                sources=settings.get('sources')
                if isinstance(sources,list):
                    for source in sources:
                        if isinstance(source,dict) and source.get('path')=='$MISSING':
                            source['path']=str(Path(temporary)/'missing.json')
                with self.subTest(case=case['name']):
                    try:
                        PricingResolver(settings)
                        accepted=True
                    except ValueError:
                        accepted=False
                    self.assertEqual(accepted,case['accepted'])
                    report=relay_plugin.validate_exact({'version':1,'components':[{'kind':'pricing','config':settings}]})
                    self.assertEqual(not any(item['level']=='error' for item in report['config']['diagnostics']),case['accepted'])

    def test_original_provider_usage_routing_cache_cost_rounding_and_precedence(self):
        self.assertEqual(sum(len(case['cases']) for case in self.fixture['scenarios']),750)
        codecs={'chat':relay_codecs.OpenAIChatCodec(),'responses':relay_codecs.OpenAIResponsesCodec(),
                'anthropic':relay_codecs.AnthropicMessagesCodec()}
        async def exercise():
            for scenario in self.fixture['scenarios']:
                activation=await relay_plugin.initialize({'version':1,'components':[{'kind':'pricing','config':scenario['config']}]})
                try:
                    for number,case in enumerate(scenario['cases']):
                        with self.subTest(scenario=scenario['name'],case=number,mode=case['mode']):
                            annotation=codecs[case['mode']].decode_response(case['body'])
                            self.assertEqual(annotation.usage,case['usage'])
                finally:
                    await activation.close()
        asyncio.run(exercise())
        self.assertNotIn('cost',relay_codecs.OpenAIChatCodec().decode_response(
            {'model':'fixture-model','usage':{'prompt_tokens':5}}).usage)

    def test_file_sources_are_snapshots_and_source_order_is_preserved(self):
        original=deepcopy(self.fixture['scenarios'][0]['config']['sources'][0]['catalog'])
        with tempfile.TemporaryDirectory(prefix='relay-pricing-file-') as temporary:
            source=Path(temporary)/'catalog.json';source.write_text(json.dumps(original))
            resolver=PricingResolver({'sources':[{'type':'file','path':str(source)}]})
            original['entries'][0]['rates']['input_per_million']=100
            source.write_text(json.dumps(original))
            self.assertEqual(resolver.pricing_for('openai','fixture-model')['rates']['input_per_million'],1.25)
            newer=PricingResolver({'sources':[{'type':'file','path':str(source)}]})
            self.assertEqual(newer.pricing_for('openai','fixture-model')['rates']['input_per_million'],100)
            source.write_text('{"version":1,"entries":[')
            with self.assertRaises(ValueError):
                PricingResolver({'sources':[{'type':'file','path':str(source)}]})

    def test_original_config_helpers_and_single_owner_lifecycle(self):
        pricing=relay_model_pricing
        settings=pricing.PricingConfig([pricing.InlineSource(pricing.PricingCatalog([
            pricing.ModelPricing(provider='openai',model_id='fixture-model',pricing_as_of='2026-01-01',
                                 pricing_source='https://fixture.invalid/rates',
                                 rates=pricing.TokenPricingRates(1.25,7.125))]))])
        self.assertFalse(pricing.validate_config(settings)['diagnostics'])
        document={'components':[pricing.ComponentSpec(settings)]}
        report=relay_plugin.validate_exact(document)
        self.assertFalse(report['config']['diagnostics'])
        both={'components':[pricing.ComponentSpec(settings),pricing.ComponentSpec(settings)]}
        self.assertTrue(any(item['level']=='error' for item in relay_plugin.validate_exact(both)['config']['diagnostics']))
        async def exercise():
            active=await relay_plugin.initialize(document)
            try:
                with self.assertRaises(RuntimeError):
                    await relay_plugin.initialize(document)
            finally:
                await active.close()
            await active.close()
        asyncio.run(exercise())

    def test_nonfinite_rates_and_invalid_token_types_refuse(self):
        for rate in (float('nan'),float('inf'),float('-inf')):
            settings=deepcopy(self.fixture['scenarios'][0]['config'])
            settings['sources'][0]['catalog']['entries'][0]['rates']['input_per_million']=rate
            with self.assertRaises(ValueError):
                PricingResolver(settings)
        for value in (-1,True,1.0,'1',2**64):
            for codec,key in ((relay_codecs.OpenAIChatCodec(),'prompt_tokens'),
                              (relay_codecs.OpenAIResponsesCodec(),'input_tokens'),
                              (relay_codecs.AnthropicMessagesCodec(),'input_tokens')):
                with self.subTest(value=value,codec=type(codec).__name__),self.assertRaises(ValueError):
                    codec.decode_response({'model':'fixture-model','usage':{key:value}})


if __name__=='__main__':
    unittest.main()
