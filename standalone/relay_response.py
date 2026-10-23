"""Pinned Relay response projections for the three provider surfaces Hermes uses."""

from copy import deepcopy
import json

from .relay_pricing import decode_usage, _uint
from .relay_types import AnnotatedLLMResponse


def _optional(body, key, expected):
    value=body.get(key)
    if value is not None and not isinstance(value, expected):
        raise ValueError('Invalid provider response '+key)
    return value


def _arguments(value):
    if value is None:
        return {}
    try:
        def reject_constant(value):
            raise ValueError('Invalid JSON numeric constant')
        return json.loads(value, parse_constant=reject_constant)
    except ValueError:
        return value


def _reason(value, mapping):
    if value is None:
        return None, None
    if value in mapping:
        return mapping[value], mapping[value]
    # The Python property is a string; the serialized enum retains Unknown.
    return value, {'unknown':value}


def decode_response(mode, body):
    if not isinstance(body, dict):
        raise ValueError('Provider response must be an object')
    for key in ('id','model','service_tier'):
        _optional(body,key,str)
    message, calls, reason, finish_json = None, None, None, None
    common={'id','model','usage'}
    if mode=='chat':
        known=common|{'choices','system_fingerprint','service_tier'}
        _optional(body,'system_fingerprint',str)
        choices=_optional(body,'choices',list)
        for choice in choices or []:
            if not isinstance(choice,dict):
                raise ValueError('Invalid Chat response choice')
            content=_optional(choice,'message',dict)
            _optional(choice,'finish_reason',str)
            if content is not None:
                _optional(content,'content',str)
                tool_calls=_optional(content,'tool_calls',list)
                for call in tool_calls or []:
                    if not isinstance(call,dict):
                        raise ValueError('Invalid Chat response tool call')
                    _optional(call,'id',str)
                    function=_optional(call,'function',dict)
                    if function is not None:
                        _optional(function,'name',str)
                        _optional(function,'arguments',str)
        first=choices[0] if choices else {}
        content=first.get('message') or {}
        message=content.get('content')
        if content.get('tool_calls') is not None:
            calls=[{'id':call.get('id') or '', 'name':call['function']['name'],
                    'arguments':_arguments(call['function'].get('arguments'))}
                   for call in content['tool_calls'] if call.get('function') is not None
                   and call['function'].get('name') is not None]
        reason,finish_json=_reason(first.get('finish_reason'),{'stop':'complete','length':'length',
                       'tool_calls':'tool_use','function_call':'tool_use','content_filter':'content_filter'})
        api={'api':'openai_chat', **{key:body[key] for key in ('system_fingerprint','service_tier') if body.get(key) is not None}}
        if first.get('logprobs') is not None:
            api['logprobs']=deepcopy(first['logprobs'])
    elif mode=='responses':
        api_fields=('status','incomplete_details','previous_response_id','store','service_tier','truncation','reasoning')
        known=common|set(api_fields)|{'output'}
        for key in ('status','previous_response_id'):
            _optional(body,key,str)
        _optional(body,'store',bool)
        output=_optional(body,'output',list)
        texts, calls=[],[]
        for item in output or []:
            if not isinstance(item,dict):
                continue
            kind=item.get('type')
            if kind=='message':
                content=item.get('content')
                if isinstance(content,list):
                    texts.extend(block['text'] for block in content if isinstance(block,dict)
                                 and block.get('type')=='output_text' and isinstance(block.get('text'),str))
            elif kind=='output_text' and isinstance(item.get('text'),str):
                texts.append(item['text'])
            elif kind=='function_call':
                calls.append({'id':item['call_id'] if isinstance(item.get('call_id'),str) else '',
                              'name':item['name'] if isinstance(item.get('name'),str) else '',
                              'arguments':_arguments(item['arguments'] if isinstance(item.get('arguments'),str) else None)})
        message='\n'.join(texts) if texts else (body.get('output_text') if isinstance(body.get('output_text'),str) and body['output_text'] else None)
        calls=calls or None
        status=body.get('status')
        if status=='completed':
            reason=finish_json='complete'
        elif status=='incomplete':
            details=body.get('incomplete_details')
            incomplete=details.get('reason') if isinstance(details,dict) else None
            if not isinstance(incomplete,str):
                incomplete='incomplete'
            reason,finish_json=_reason(incomplete,{'max_output_tokens':'length','content_filter':'content_filter'})
        else:
            reason,finish_json=_reason(status,{})
        api={'api':'openai_responses', **{key:deepcopy(body[key]) for key in api_fields if body.get(key) is not None}}
        if output is not None:
            api['output_items']=deepcopy(output)
        raw=body.get('usage')
        if isinstance(raw,dict):
            for key,field in (('input_tokens_details','cached_tokens'),('output_tokens_details','reasoning_tokens')):
                details=_optional(raw,key,dict)
                if details is not None:
                    if details.get(field) is not None:
                        _uint(details[field],field)
                    api[key]=deepcopy({name:value for name,value in details.items() if name!=field or value is not None})
    else:
        api_fields=('role','stop_reason','stop_sequence','service_tier','container')
        known=common|set(api_fields)|{'type','content'}
        for key in ('type','role','stop_reason','stop_sequence'):
            _optional(body,key,str)
        content=_optional(body,'content',list)
        texts, calls=[],[]
        for item in content or []:
            if not isinstance(item,dict):
                continue
            if item.get('type')=='text' and isinstance(item.get('text'),str):
                texts.append(item['text'])
            elif item.get('type')=='tool_use' and isinstance(item.get('id'),str) and isinstance(item.get('name'),str) and 'input' in item:
                calls.append({'id':item['id'],'name':item['name'],'arguments':deepcopy(item['input'])})
        message='\n'.join(texts) if texts else None
        calls=calls or None
        reason,finish_json=_reason(body.get('stop_reason'),{'end_turn':'complete','max_tokens':'length','tool_use':'tool_use'})
        api={'api':'anthropic_messages', **{key:deepcopy(body[key]) for key in api_fields if body.get(key) is not None}}
        if body.get('type') is not None:
            api['object_type']=body['type']
        if content is not None:
            api['content_blocks']=deepcopy(content)
    result=AnnotatedLLMResponse(id=body.get('id'),model=body.get('model'),message=message,tool_calls=calls,
        finish_reason=reason,usage=decode_usage(mode,body),api_specific=api,
        extra={key:deepcopy(value) for key,value in body.items() if key not in known})
    result._finish_source, result._finish_json = reason, finish_json
    return result
