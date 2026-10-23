"""Pinned Relay model-pricing catalogs and normalized provider usage.

Rates come only from the caller's validated catalog snapshots. No pricing,
network lookup, package installation, or currency conversion is implicit.
The source and native reference contracts are in relay-pricing-provenance.json.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import threading


_LOCK = threading.RLock()
_ACTIVE = None
_U64 = 2**64 - 1
_AMOUNTS = ('input', 'output', 'cache_read', 'cache_write')
_PROVENANCE = ('pricing_provider', 'pricing_model', 'pricing_as_of', 'pricing_source')
_LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')


def _lower(value):
    return value.translate(_LOWER)


def _provider(value):
    return _lower(value.strip().strip('/'))


def _object(value, allowed, required=(), *, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError('Invalid model pricing ' + label)
    return value


def _uint(value, label):
    if type(value) is not int or not 0 <= value <= _U64:
        raise ValueError('Invalid unsigned model pricing ' + label)
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Invalid numeric model pricing ' + label)
    try:
        value = float(value)
    except OverflowError:
        raise ValueError('Invalid numeric model pricing ' + label) from None
    if not math.isfinite(value):
        raise ValueError('Nonfinite model pricing ' + label)
    return value


def _rates(value):
    fields = ('input_per_million', 'output_per_million', 'cache_read_per_million', 'cache_write_per_million')
    _object(value, fields, fields[:2], label='token rates')
    result = {}
    for key in fields:
        rate = value.get(key)
        if rate is None and key not in fields[:2]:
            continue
        rate = _number(rate, key)
        if rate < 0:
            raise ValueError('Negative model pricing rate')
        result[key] = rate
    return result


def _entry(value):
    text = ('provider', 'model_id', 'pricing_as_of', 'pricing_source')
    _object(value, (*text, 'aliases', 'currency', 'unit', 'rates', 'rate_schedule', 'prompt_cache'),
            (*text, 'prompt_cache'), label='entry')
    result = deepcopy(value)
    result.setdefault('currency', 'USD')
    result.setdefault('unit', 'per_token')
    result.setdefault('aliases', [])
    for key in (*text, 'currency'):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError('Empty or invalid model pricing ' + key)
    if not isinstance(result['aliases'], list) or any(not isinstance(item, str) for item in result['aliases']):
        raise ValueError('Invalid model pricing aliases')
    if result['unit'] not in ('per_token', 'per_request', 'per_second', 'gpu_hour'):
        raise ValueError('Invalid model pricing billing unit')
    cache = _object(result['prompt_cache'], ('read_accounting',), ('read_accounting',), label='prompt cache')
    if cache['read_accounting'] not in ('included_in_prompt_tokens', 'separate'):
        raise ValueError('Invalid model pricing cache accounting')
    if result.get('rates') is not None:
        result['rates'] = _rates(result['rates'])
    if result.get('rate_schedule') is not None:
        schedule = _object(result['rate_schedule'], ('type', 'applies_to', 'tiers'), ('type', 'tiers'), label='rate schedule')
        if schedule['type'] != 'prompt_token_threshold' or schedule.get('applies_to', 'full_request') != 'full_request':
            raise ValueError('Unsupported model pricing rate schedule')
        tiers = schedule['tiers']
        if not isinstance(tiers, list) or not tiers:
            raise ValueError('Empty model pricing rate schedule tiers')
        for tier in tiers:
            _object(tier, ('min_prompt_tokens', 'max_prompt_tokens', 'rates'), ('rates',), label='rate tier')
            for bound in ('min_prompt_tokens', 'max_prompt_tokens'):
                if tier.get(bound) is not None:
                    _uint(tier[bound], bound)
            if (tier.get('min_prompt_tokens') is not None and tier.get('max_prompt_tokens') is not None
                    and tier['min_prompt_tokens'] > tier['max_prompt_tokens']):
                raise ValueError('Reversed model pricing token bounds')
            tier['rates'] = _rates(tier['rates'])
    if result['unit'] == 'per_token' and result.get('rates') is None and result.get('rate_schedule') is None:
        raise ValueError('Per-token model pricing requires rates or a rate schedule')
    return result


def _catalog(value):
    _object(value, ('version', 'entries'), ('version', 'entries'), label='catalog')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Unsupported model pricing catalog version')
    if not isinstance(value['entries'], list):
        raise ValueError('Invalid model pricing catalog entries')
    entries, seen = [], set()
    for value in value['entries']:
        item = _entry(value)
        for model in (item['model_id'], *item['aliases']):
            model = _lower(model.strip())
            if not model:
                continue
            key = _provider(item['provider']) + '/' + model
            if key in seen:
                raise ValueError('Duplicate model pricing alias')
            seen.add(key)
        entries.append(item)
    return entries


def _lookup(provider, model):
    model = _lower(model.strip())
    if not model:
        return ()
    parts = [part.strip() for part in model.split('/') if part.strip()]
    keys = []
    explicit = _provider(provider) if provider is not None else ''
    terminal = parts[-1] if parts else model
    if explicit:
        keys.extend(((explicit, model), (explicit, terminal)))
    elif len(parts) > 1:
        keys.append(('/'.join(parts[:-1]), terminal))
    keys.extend((None, '/'.join(parts[index:])) for index in range(len(parts)))
    return tuple(dict.fromkeys(keys))


def _round_cost(value):
    # Rust f64::round uses ties away from zero; Python round uses ties to even.
    # Splitting the fraction also avoids an extra rounding from adding 0.5.
    scaled = value * 1e12
    if not math.isfinite(scaled):
        return None  # serde_json serializes nonfinite f64 values as null.
    fraction, whole = math.modf(scaled)
    return (whole + (1.0 if fraction >= .5 else 0.0)) / 1e12


class PricingResolver:
    def __init__(self, config):
        _object(config, ('sources',), label='configuration')
        sources = config.get('sources', [])
        if not isinstance(sources, list):
            raise ValueError('Invalid model pricing sources')
        catalogs = []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError('Invalid model pricing source')
            kind = source.get('type')
            if kind == 'inline':
                _object(source, ('type', 'catalog'), ('catalog',), label='inline source')
                catalog = source['catalog']
            elif kind == 'file':
                _object(source, ('type', 'path'), ('path',), label='file source')
                if not isinstance(source['path'], str):
                    raise ValueError('Invalid model pricing catalog file path')
                try:
                    catalog = json.loads(Path(source['path']).read_text(encoding='utf-8'))
                except (OSError, ValueError) as error:
                    raise ValueError('Cannot read model pricing catalog file') from error
            else:
                raise ValueError('Unsupported model pricing source')
            catalogs.append(_catalog(catalog))
        self._catalogs = tuple(catalogs)

    def pricing_for(self, provider, model):
        for entries in self._catalogs:
            for route, name in _lookup(provider, model):
                for entry in entries:
                    if route is not None and _provider(entry['provider']) != route:
                        continue
                    if any(_lower(value.strip()) == name for value in (entry['model_id'], *entry['aliases'])):
                        return deepcopy(entry)
        return None

    def estimate_cost(self, provider, model, usage):
        entry = self.pricing_for(provider, model)
        if entry is None or entry['unit'] != 'per_token':
            return None
        rates = entry.get('rates')
        if schedule := entry.get('rate_schedule'):
            tokens = usage.get('prompt_tokens')
            if tokens is None:
                return None
            rates = next((tier['rates'] for tier in schedule['tiers'] if
                          (tier.get('min_prompt_tokens') is None or tokens >= tier['min_prompt_tokens']) and
                          (tier.get('max_prompt_tokens') is None or tokens <= tier['max_prompt_tokens'])), None)
        if rates is None:
            return None
        prompt, output, read, write = (_uint(usage.get(key, 0), key) for key in
                                       ('prompt_tokens', 'completion_tokens', 'cache_read_tokens', 'cache_write_tokens'))
        if not (prompt or output or read or write):
            return None
        if entry['prompt_cache']['read_accounting'] == 'included_in_prompt_tokens':
            prompt = max(0, prompt-read)
        result = {'currency':entry['currency'], 'source':'model_pricing',
                  'pricing_provider':entry['provider'], 'pricing_model':entry['model_id'],
                  'pricing_as_of':entry['pricing_as_of'], 'pricing_source':entry['pricing_source']}
        for name, tokens, rate in zip(_AMOUNTS, (prompt,output,read,write),
                                     ('input_per_million','output_per_million','cache_read_per_million','cache_write_per_million')):
            if tokens > 0 and rate in rates:
                result[name] = _round_cost(float(tokens)*rates[rate]/1e6)
        if not (read > 0 and 'cache_read_per_million' not in rates or write > 0 and 'cache_write_per_million' not in rates):
            values = [result[name] for name in _AMOUNTS if name in result]
            result['total'] = None if None in values else _round_cost(sum(values))
        return result


def set_active(resolver):
    global _ACTIVE
    with _LOCK:
        _ACTIVE = resolver


def _reported_cost(scalar, detailed):
    detailed = {} if detailed is None else detailed
    if not isinstance(detailed, dict):
        raise ValueError('Invalid provider cost object')
    amounts = {}
    for key in ('total', *_AMOUNTS):
        if detailed.get(key) is not None:
            amounts[key] = _number(detailed[key], 'provider cost')
    currency = detailed.get('currency')
    if currency is not None and not isinstance(currency, str):
        raise ValueError('Invalid provider cost currency')
    if scalar is not None:
        scalar = _number(scalar, 'provider cost')
        if currency is not None and _lower(currency) != 'usd':
            amounts = {}
    components = {key:amounts[key] for key in _AMOUNTS if key in amounts}
    total = scalar if scalar is not None else amounts.get('total')
    if total is None and components:
        total = sum(components.values())
    if total is None and not components:
        return None
    result = {**components, 'source':'provider_reported',
              'currency':'USD' if scalar is not None else (currency if currency is not None and amounts else 'USD')}
    if total is not None:
        result['total'] = total
    for key in _PROVENANCE:
        if detailed.get(key) is not None:
            if not isinstance(detailed[key], str):
                raise ValueError('Invalid provider cost provenance')
            result[key] = detailed[key]
    return result


def decode_usage(mode, response):
    raw = response.get('usage')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('Provider usage must be an object')
    usage = {}
    if mode == 'chat':
        names = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        detail_name = 'prompt_tokens_details'
    else:
        names = ('input_tokens', 'output_tokens', 'total_tokens')
        detail_name = 'input_tokens_details'
    for source, target in zip(names, ('prompt_tokens','completion_tokens','total_tokens')):
        if raw.get(source) is not None:
            usage[target] = _uint(raw[source], source)
    eligible = True
    if mode == 'anthropic':
        usage.pop('total_tokens', None)
        iterations = raw.get('iterations')
        if iterations is not None and not isinstance(iterations, list):
            raise ValueError('Anthropic usage iterations must be an array')
        for source, target in (('cache_read_input_tokens','cache_read_tokens'),
                               ('cache_creation_input_tokens','cache_write_tokens')):
            if raw.get(source) is not None:
                usage[target] = _uint(raw[source], source)
        if iterations:
            for item in iterations:
                if not isinstance(item, dict):
                    raise ValueError('Anthropic usage iteration must be an object')
            for source, target in (('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),
                                   ('cache_read_input_tokens','cache_read_tokens'),('cache_creation_input_tokens','cache_write_tokens')):
                values = [_uint(item[source], source) for item in iterations if item.get(source) is not None]
                if values:
                    usage[target] = min(_U64, sum(values))
            eligible = all(item.get('model') is None or item['model'] == response.get('model') for item in iterations)
        if 'prompt_tokens' in usage:
            usage['uncached_input_tokens'] = usage['prompt_tokens']
        if 'prompt_tokens' in usage and 'completion_tokens' in usage:
            usage['total_tokens'] = min(_U64, usage['prompt_tokens']+usage['completion_tokens'])
    else:
        details = raw.get(detail_name)
        if details is not None:
            if not isinstance(details, dict):
                raise ValueError('Provider token details must be an object')
            if details.get('cached_tokens') is not None:
                usage['cache_read_tokens'] = _uint(details['cached_tokens'], 'cached_tokens')
        if 'prompt_tokens' in usage and 'cache_read_tokens' in usage and usage['prompt_tokens'] >= usage['cache_read_tokens']:
            usage['uncached_input_tokens'] = usage['prompt_tokens']-usage['cache_read_tokens']
    scalar, detailed = raw.get('cost_usd'), raw.get('cost')
    if mode == 'chat' and detailed is not None and not isinstance(detailed, dict):
        cost_scalar = _number(detailed, 'provider cost')
        if scalar is None:
            scalar = cost_scalar
        detailed = None
    cost = _reported_cost(scalar, detailed)
    if cost is None and eligible and isinstance(response.get('model'), str):
        model = response['model']
        parts = [part.strip() for part in _lower(model.strip()).split('/') if part.strip()]
        provider = '/'.join(parts[:-1]) if len(parts) > 1 else ('anthropic' if mode == 'anthropic' else 'openai')
        with _LOCK:
            resolver = _ACTIVE
        if resolver is not None:
            cost = resolver.estimate_cost(provider, model, usage)
    if cost is not None:
        usage['cost'] = cost
    return usage
