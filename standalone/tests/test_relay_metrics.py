"""Independent frozen native Relay schema, aggregation and resource-lifecycle contracts."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import math
from standalone import relay_metrics as metrics
from standalone.relay_metrics import MetricKind, MetricValueType, MetricMeasurement, metric_payload, METRIC_SCHEMA
from standalone.relay_metrics_aggregate import MetricProcessor
from standalone.relay_metrics_route import MetricRouter
from standalone.relay_metrics_otlp import resolve_metric_endpoints, validate_metrics
ROOT = Path(__file__).with_name('fixtures')

def restored(value):
    if isinstance(value, dict):
        if set(value) == {'special_float'}:
            return float(value['special_float'])
        return {key: restored(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restored(item) for item in value]
    return value

def normalized(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value

class MetricSchemaTests(unittest.TestCase):
    def test_original_constructor_properties_and_atomic_publication(self):
        fixture = restored(json.loads((ROOT / 'relay_metrics_schema.json').read_text()))
        for vector in fixture['cases']:
            actual = {}
            try:
                measurements = []
                for arguments in restored(vector['arguments']):
                    arguments['kind'] = getattr(metrics.MetricKind, arguments['kind'])
                    arguments['value_type'] = getattr(metrics.MetricValueType, arguments['value_type'])
                    measurements.append(metrics.MetricMeasurement(**arguments))
                actual['properties'] = [{name: getattr(value, name) for name in ['name','value','unit','description','attributes','boundaries']}
                                        for value in measurements]
            except Exception as error:
                actual.update(phase='constructor', error_type=type(error).__name__, error=str(error))
            else:
                try:
                    payload = metrics.metric_payload(measurements)
                except Exception as error:
                    actual.update(phase='publication', error_type=type(error).__name__, error=str(error))
                else:
                    actual.update(phase='accepted', data=payload, data_schema=metrics.METRIC_SCHEMA)
            expected = {key: value for key, value in vector.items() if key not in ('id','arguments')}
            with self.subTest(case=vector['id']):
                self.assertEqual(normalized(actual), normalized(expected))

    def test_original_enum_protocol_and_owned_properties(self):
        fixture = restored(json.loads((ROOT / 'relay_metrics_schema.json').read_text()))
        for enum_name, rows in fixture['enum_contract'].items():
            for row in rows:
                value = getattr(getattr(metrics, enum_name), row['name'])
                self.assertEqual((str(value), repr(value), int(value), value == row['int']),
                                 (row['str'],row['repr'],row['int'],row['equal_to_int']))
                with self.assertRaises(TypeError):
                    hash(value)
        attributes = {'array': [1,2]}
        value = metrics.MetricMeasurement('owned', metrics.MetricKind.Gauge, metrics.MetricValueType.I64, 1, attributes=attributes)
        attributes['array'].append(3)
        self.assertEqual(value.attributes, {'array':[1,2]})
        copy = value.attributes
        copy['array'].append(4)
        self.assertEqual(value.attributes, {'array':[1,2]})
        with self.assertRaises(AttributeError):
            value.value = 3

def event(mark):
    if 'raw' in mark:
        payload, schema = mark['raw'], mark.get('schema', METRIC_SCHEMA)
    else:
        rows = []
        for args in mark['measurements']:
            args = dict(args)
            args['kind'] = getattr(MetricKind, args['kind'])
            args['value_type'] = getattr(MetricValueType, args['value_type'])
            rows.append(MetricMeasurement(**args))
        payload, schema = metric_payload(rows), METRIC_SCHEMA
    return dict(name=mark['name'], data=payload, data_schema=schema)

def without_clocks(metrics):
    metrics = deepcopy(metrics)
    for metric in metrics:
        for point in metric['data_points']:
            del point['start_time_unix_nano'], point['time_unix_nano']
    return metrics

class MetricAggregationTests(unittest.TestCase):
    def test_native_export_failure_keeps_the_next_delta_collection_empty(self):
        fixture=json.loads((ROOT/'relay_metrics_failure.json').read_text())
        for case in fixture['cases']:
            processor=MetricProcessor(case['temporality'])
            for index,collection in enumerate(case['collections']):
                if index in (0,2):
                    processor.process(event({'name':'failure.metric','measurements':[{
                        'name':'failure.counter','kind':'Counter','value_type':'U64','value':3 if index==0 else 7}]}))
                actual=without_clocks(processor.collect())
                expected=[metric for message in collection['messages'] for metric in message['metrics']]
                with self.subTest(temporality=case['temporality'],collection=index):
                    self.assertEqual(actual,expected)

    def test_original_native_otlp_collections(self):
        fixture = restored(json.loads((ROOT / 'relay_metrics_export.json').read_text()))
        for scenario in fixture['scenarios']:
            processor = MetricProcessor(temporality=scenario['temporality'], max_instruments=scenario['max_instruments'],
                                        cardinality_limit=scenario['cardinality_limit'])
            for index, cycle in enumerate(scenario['cycles']):
                for mark in cycle['marks']:
                    processor.process(event(mark))
                expected = [metric for message in cycle['messages'] for metric in message['metrics']]
                actual = without_clocks(processor.collect())
                with self.subTest(scenario=scenario['id'], collection=index):
                    self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
                    self.assertEqual(processor.diagnostics.snapshot(), cycle['diagnostics'])

class MetricRoutingTests(unittest.TestCase):
    def test_original_native_plugin_resources_inheritance_late_marks_and_budget(self):
        fixture = json.loads((ROOT/'relay_metrics_lifecycle.json').read_text())
        for scenario in fixture['resources']:
            router = MetricRouter(scenario['endpoint'], {})
            for event in scenario['events']:
                router.process(event)
            actual = [dict(resource=resource, metrics=strip_clocks(metrics)) for resource,metrics in router.collect()]
            expected = []
            for message in scenario['messages']:
                resource = {key:value for key,value in message['metadata'][0]['resource'].items() if not key.startswith('telemetry.sdk.')}
                expected.append(dict(resource=resource,metrics=message['metrics']))
            by_resource = lambda value:json.dumps(value['resource'],sort_keys=True)
            with self.subTest(scenario=scenario['id']):
                self.assertEqual(sorted(actual,key=by_resource),sorted(expected,key=by_resource))
                native = [{key:row[key] for key in ('code','message','count')} for row in scenario['report']['config']['runtime_diagnostics']]
                self.assertEqual(router.diagnostics.snapshot(),sorted(native,key=lambda value:value['code']))

    def test_original_native_stream_first_chunk_histogram_and_provider_projection(self):
        fixture = json.loads((ROOT/'relay_metrics_lifecycle.json').read_text())
        for scenario in fixture['streams']:
            router = MetricRouter({'service_name':'owned-streams'}, {})
            for event in scenario['events']:
                router.process(event)
            actual = [metric for _,metrics in router.collect() for metric in strip_clocks(metrics)]
            expected = [metric for message in scenario['messages'] for metric in message['metrics']]
            with self.subTest(scenario=scenario['id']):
                self.assertEqual(actual,expected)

class ReservedMarkTests(unittest.TestCase):
    def test_native_reserved_mark_deserialization(self):
        for case in json.loads((ROOT/'relay_metrics_raw.json').read_text())['cases']:
            processor=MetricProcessor()
            processor.process(dict(name='raw.fixture',data=case['payload'],data_schema=case.get('schema',METRIC_SCHEMA)))
            with self.subTest(case=case['id']):
                self.assertEqual(processor.diagnostics.snapshot(),case['diagnostics'])
                self.assertEqual(bool(processor.collect()),case['exported'])

strip_clocks = without_clocks


class MetricEndpointTests(unittest.TestCase):
    def test_native_plugin_endpoint_resolution_and_temporality(self):
        for case in json.loads((ROOT/'relay_metrics_resolution.json').read_text())['cases']:
            with self.subTest(case=case['id']):
                if not case['accepted']:
                    with self.assertRaises(ValueError):
                        validate_metrics(case['opentelemetry'])
                else:
                    from urllib.parse import urlsplit
                    from standalone.relay_otlp import _signal_url
                    endpoints=validate_metrics(case['opentelemetry'])
                    paths=[urlsplit(_signal_url(endpoint['endpoint'],'metrics')).path for endpoint in endpoints]
                    self.assertEqual(paths,case['paths'])

    def test_delta_clock_advances_on_idle_collection_and_diagnostics_are_bounded(self):
        from standalone.relay_metrics_aggregate import MetricDiagnostics
        stamps=iter((100,200,300,400))
        processor=MetricProcessor('delta',clock=lambda:next(stamps))
        processor.process(event({'name':'clock','measurements':[{'name':'clock.counter','kind':'Counter','value_type':'U64','value':3}]}))
        point=processor.collect()[0]['data_points'][0]
        self.assertEqual((point['start_time_unix_nano'],point['time_unix_nano']),(100,200))
        self.assertEqual(processor.collect(),[])
        processor.process(event({'name':'clock','measurements':[{'name':'clock.counter','kind':'Counter','value_type':'U64','value':5}]}))
        point=processor.collect()[0]['data_points'][0]
        self.assertEqual((point['start_time_unix_nano'],point['time_unix_nano'],point['as_int']),(300,400,5))
        diagnostics=MetricDiagnostics()
        for index in range(40):diagnostics.record('diagnostic.'+str(index),'가'*1100)
        self.assertEqual(len(diagnostics.snapshot()),32)
        self.assertTrue(all(len(value['message'])==1024 and value['message'].endswith('…') for value in diagnostics.snapshot()))

if __name__ == '__main__':
    unittest.main()
