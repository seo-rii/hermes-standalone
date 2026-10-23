"""Owned stdlib pumps and ordered, asynchronous Relay stream publication.

Derived from nemo-relay 0.10.0 stream.rs and its Python iterator bridge.
The two 32-item channels preserve producer and consumer backpressure.
"""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
import contextvars
from copy import deepcopy
from dataclasses import dataclass
import re
import time

from standalone.relay_metrics import _json


class DeferredEvent:
    """A reserved dispatcher sequence built and delivered on its owning loop."""
    def __init__(self, runtime, factory, loop, context):
        self.runtime, self.factory, self.loop, self.context = runtime, factory, loop, context

    def dispatch(self, callbacks):
        completed = Future()

        async def deliver():
            token = self.runtime._CALLBACK.set(True)
            try:
                event = await self.factory()
                if event is not None:
                    for callback in callbacks:
                        try:
                            await self.runtime._await(callback(event))
                        except Exception:
                            self.runtime.logger.warning('Relay subscriber failed', exc_info=True)
            finally:
                self.runtime._CALLBACK.reset(token)

        def finished(task):
            try:
                completed.set_result(task.result())
            except BaseException as error:
                completed.set_exception(RuntimeError('Relay stream publication was cancelled')
                                        if isinstance(error, asyncio.CancelledError) else error)

        def start():
            task = self.context.copy().run(self.loop.create_task, deliver())
            task.add_done_callback(finished)

        self.loop.call_soon_threadsafe(start)
        completed.result()


class _Closed(Exception):
    pass


@dataclass
class _Failure:
    error: BaseException


class _Channel:
    def __init__(self):
        self.items, self.closed, self.condition = deque(), False, asyncio.Condition()

    async def send(self, value):
        async with self.condition:
            await self.condition.wait_for(lambda: self.closed or len(self.items) < 32)
            if self.closed:
                raise _Closed
            self.items.append(value)
            self.condition.notify_all()

    async def receive(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.closed or self.items)
            if not self.items:
                raise _Closed
            value = self.items.popleft()
            self.condition.notify_all()
            return value

    async def close(self, *, discard=False):
        async with self.condition:
            self.closed = True
            if discard:
                self.items.clear()
            self.condition.notify_all()


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else default


def _uint(value):
    return type(value) is int and 0 <= value <= 18446744073709551615


def _usage(value, mode):
    result = {}
    fields = (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens'),
              ('total_tokens', 'total_tokens')) if mode == 'chat' else (
              ('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'))
    if mode == 'responses':
        fields += (('total_tokens', 'total_tokens'),)
    for source, destination in fields:
        item = _get(value, source)
        if _uint(item):
            result[destination] = item
    if mode == 'anthropic':
        if 'prompt_tokens' in result and 'completion_tokens' in result:
            total = result['prompt_tokens'] + result['completion_tokens']
            if _uint(total):
                result['total_tokens'] = total
        for source, destination in (('cache_read_input_tokens', 'cache_read_tokens'),
                                    ('cache_creation_input_tokens', 'cache_write_tokens')):
            item = _get(value, source)
            if _uint(item):
                result[destination] = item
    else:
        key = 'prompt_tokens_details' if mode == 'chat' else 'input_tokens_details'
        cached = _get(_get(value, key), 'cached_tokens')
        if _uint(cached):
            result['cache_read_tokens'] = cached
    return result


def chunk_receipt(index, chunk):
    """Publish only structural receipt fields; raw tokens stay in the stream."""
    result = {'chunk_index': index, 'provider': 'unknown'}
    kind = _get(chunk, 'object')
    if kind == 'chat.completion.chunk':
        result.update(provider='openai_chat_completions', event_type=kind)
        choices = _get(chunk, 'choices')
        if isinstance(choices, list):
            indices, reasons = [], []
            for choice in choices:
                number, reason = _get(choice, 'index'), _get(choice, 'finish_reason')
                if _uint(number):
                    indices.append(number)
                if isinstance(reason, str):
                    item = {'finish_reason': reason}
                    if _uint(number):
                        item['choice_index'] = number
                    reasons.append(item)
            if indices:
                result['choice_indices'] = indices
            if reasons:
                result['finish_reasons'] = reasons
        usage = _usage(_get(chunk, 'usage'), 'chat')
    else:
        kind = _get(chunk, 'type')
        if isinstance(kind, str) and kind.startswith('response.'):
            result.update(provider='openai_responses', event_type=kind)
            keys, mode = ('output_index', 'content_index'), 'responses'
            nested = _get(chunk, 'response')
            status = _get(nested, 'status', _get(chunk, 'status'))
            if isinstance(status, str):
                result['status'] = status
            reason = _get(_get(nested, 'incomplete_details'), 'reason')
            if isinstance(reason, str):
                result['finish_reason'] = reason
        elif kind in ('message_start', 'content_block_start', 'content_block_delta',
                      'content_block_stop', 'message_delta', 'message_stop', 'ping'):
            result.update(provider='anthropic_messages', event_type=kind)
            keys, mode = ('index',), 'anthropic'
            nested = _get(chunk, 'message')
            reason = _get(_get(chunk, 'delta'), 'stop_reason', _get(nested, 'stop_reason'))
            if isinstance(reason, str):
                result['stop_reason'] = reason
        else:
            return result
        indices = {key: _get(chunk, key) for key in keys if _uint(_get(chunk, key))}
        if indices:
            result['indices'] = indices
        usage = _usage(_get(chunk, 'usage', _get(nested, 'usage')), mode)
    if usage:
        result['usage'] = usage
    return result


def _metadata(handle, termination, error=None):
    result = deepcopy(handle.metadata) if isinstance(handle.metadata, dict) else {}
    result['otel.status_code'] = 'OK' if termination == 'complete' else 'ERROR'
    if termination == 'complete':
        result.pop('otel.status_description', None)
    elif error is None:
        result['otel.status_description'] = 'stream dropped before clean completion'
    else:
        result.update({'error.type': 'internal_error', 'exception.type': type(error).__name__,
                       'otel.status_description': 'internal error: ' + type(error).__name__ + ': ' + str(error)})
    return result


def _annotation(runtime, profile):
    """Rebuild the side-channel annotation only from the final sanitized event."""
    body = _get(profile, 'annotated_response')
    if not isinstance(body, dict):
        return None
    known = ('id', 'model', 'message', 'tool_calls', 'finish_reason', 'usage', 'api_specific')
    values = {key: deepcopy(body[key]) for key in known if key in body}
    finish = values.get('finish_reason')
    if isinstance(finish, dict) and isinstance(finish.get('unknown'), str):
        values['finish_reason'] = finish['unknown']
    result = runtime.AnnotatedLLMResponse(**values, extra={key: deepcopy(value) for key, value in body.items() if key not in known})
    return result


def _typed_optional(body, key, expected):
    value = body.get(key)
    if value is not None:
        accepted = type(value) in (int, float) if expected is float else isinstance(value, expected)
        if not accepted:
            raise ValueError('Invalid Relay profile ' + key)
    return value


def _profile(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Invalid Relay category profile')
    result = deepcopy(value)
    for key in ('model_name', 'tool_call_id', 'subtype'):
        _typed_optional(result, key, str)
    _typed_optional(result, 'time_to_first_chunk', float)
    for key in ('model_name', 'tool_call_id', 'subtype', 'tool_result_annotation',
                'annotated_request', 'annotated_response', 'time_to_first_chunk'):
        if result.get(key) is None:
            result.pop(key, None)
    response = result.get('annotated_response')
    if response is not None:
        if not isinstance(response, dict):
            raise ValueError('Invalid Relay response annotation')
        for key in ('id', 'model'):
            _typed_optional(response, key, str)
        _typed_optional(response, 'message', (str, list))
        calls = _typed_optional(response, 'tool_calls', list)
        for call in calls or []:
            if not isinstance(call, dict) or not isinstance(call.get('id'), str) or not isinstance(call.get('name'), str) or 'arguments' not in call:
                raise ValueError('Invalid Relay response tool call')
        reason = response.get('finish_reason')
        if reason is not None and reason not in ('complete', 'length', 'tool_use', 'content_filter'):
            if not isinstance(reason, dict) or set(reason) != {'unknown'} or not isinstance(reason['unknown'], str):
                raise ValueError('Invalid Relay response finish reason')
        usage = _typed_optional(response, 'usage', dict)
        if usage is not None:
            keys = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cache_read_tokens',
                    'cache_write_tokens', 'uncached_input_tokens', 'cost')
            response['usage'] = {key: deepcopy(usage[key]) for key in keys if usage.get(key) is not None}
            for key, item in response['usage'].items():
                if key != 'cost' and not _uint(item):
                    raise ValueError('Invalid Relay response usage')
        api = _typed_optional(response, 'api_specific', dict)
        if api is not None:
            variants = {
                'openai_chat': ('logprobs', 'system_fingerprint', 'service_tier'),
                'openai_responses': ('output_items', 'status', 'incomplete_details', 'previous_response_id',
                    'store', 'service_tier', 'truncation', 'reasoning', 'input_tokens_details', 'output_tokens_details'),
                'anthropic_messages': ('object_type', 'role', 'stop_reason', 'stop_sequence', 'service_tier', 'container', 'content_blocks'),
                'oci_genai': ('api_format', 'model_version'),
                'gemini_generate_content': ('thoughts_tokens', 'safety_ratings', 'grounding_metadata', 'citation_metadata'),
                'custom': ('api_name', 'data'),
            }
            tag = api.get('api')
            if not isinstance(tag, str) or tag not in variants:
                raise ValueError('Invalid Relay response API tag')
            for key in ('system_fingerprint', 'service_tier', 'status', 'previous_response_id', 'object_type',
                        'role', 'stop_reason', 'stop_sequence', 'api_format', 'model_version', 'api_name'):
                _typed_optional(api, key, str)
            for key in ('output_items', 'content_blocks'):
                _typed_optional(api, key, list)
            _typed_optional(api, 'store', bool)
            if tag == 'custom' and (not isinstance(api.get('api_name'), str) or 'data' not in api):
                raise ValueError('Invalid Relay custom API annotation')
            response['api_specific'] = {'api': tag, **{key: deepcopy(item) for key, item in api.items()
                if key != 'api' and (tag == 'gemini_generate_content' or key in variants[tag])
                and (item is not None or tag == 'custom' and key == 'data')}}
        for key in ('id', 'model', 'message', 'tool_calls', 'finish_reason', 'usage', 'optimization_summary', 'api_specific'):
            if response.get(key) is None:
                response.pop(key, None)
    return result


async def _sanitize_event(runtime, record, *, entries=None):
    record = _json(record)
    kind = 'mark_sanitize' if record['kind'] == 'mark' else 'scope_sanitize_' + record['scope_category']
    fields = {key: deepcopy(record.get(key)) for key in ('data', 'category_profile', 'metadata')}
    event = runtime.Event(record, annotated_response=_annotation(runtime, record.get('category_profile')))
    for _, _, _, callback in runtime.guardrails.snapshot(kind) if entries is None else entries:
        try:
            updated = await runtime._await(callback(event, deepcopy(fields)))
            if not isinstance(updated, dict) or set(updated) != set(fields):
                raise ValueError('Invalid Relay sanitizer result')
            fields = _json(updated)
            fields['category_profile'] = _profile(fields['category_profile'])
        except Exception:
            fields = dict.fromkeys(fields)
            break
    record.update(fields)
    for _, _, _, callback in runtime.event_metadata.snapshot('injector'):
        try:
            additions = _json(await runtime._await(callback(runtime.Event(record))))
            if not isinstance(additions, dict) or any(not re.fullmatch(r'[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*', key) for key in additions):
                raise ValueError('Invalid Relay metadata injector keys')
            for value in additions.values():
                if isinstance(value, list):
                    if any(type(item) not in (str, bool, int, float) for item in value) or len({type(item) for item in value}) > 1:
                        raise ValueError('Relay metadata arrays must be homogeneous scalar values')
                elif type(value) not in (str, bool, int, float):
                    raise ValueError('Relay metadata must contain scalar values')
            record['metadata'] = {**additions, **(record.get('metadata') or {})}
        except Exception:
            runtime.logger.warning('Relay metadata injector failed', exc_info=True)
    return runtime.Event(record, annotated_response=_annotation(runtime, record.get('category_profile')))


def _reserve(runtime, factory, loop, context, callbacks):
    runtime.subscribers.publish(DeferredEvent(runtime, factory, loop, context), callbacks=callbacks, force=True)


def failed_start(runtime, handle, error, callbacks):
    record = runtime._record(handle, 'end', None, _metadata(handle, 'failed', error))
    entries = runtime.guardrails.snapshot('scope_sanitize_end')
    _reserve(runtime, lambda: _sanitize_event(runtime, record, entries=entries),
             asyncio.get_running_loop(), contextvars.copy_context(), callbacks)
    with runtime._STATE_LOCK:
        runtime._HANDLES.pop(handle.uuid, None)


class _State:
    def __init__(self, runtime, source, handle, collector, finalizer, codec, started, callbacks):
        self.runtime, self.source, self.handle = runtime, source, handle
        self.collector, self.finalizer, self.codec, self.started = collector, finalizer, codec, started
        self.loop, self.context = asyncio.get_running_loop(), contextvars.copy_context()
        self.callbacks = callbacks
        self.raw, self.output = _Channel(), _Channel()
        self.first_chunk, self.chunk_index, self.ended, self.exhausted = None, 0, False, False
        self.close_error, self.source_cleanup = None, None
        self.producer = self.loop.create_task(self._produce())
        self.manager = self.loop.create_task(self._forward())
        self.close_task = None

    async def _produce(self):
        try:
            while True:
                try:
                    chunk = await self.source.__anext__()
                except StopAsyncIteration:
                    self.exhausted = True
                    break
                await self.raw.send(_json(chunk))
        except (asyncio.CancelledError, _Closed):
            raise
        except BaseException as error:
            await self.raw.send(_Failure(error))
        finally:
            await self.raw.close()

    async def _stop_source(self):
        if self.source_cleanup is None:
            self.source_cleanup = self.loop.create_task(self._cleanup_source(), context=self.context.copy())
        await asyncio.shield(self.source_cleanup)

    async def _cleanup_source(self):
        if not self.producer.done():
            self.producer.cancel()
        await asyncio.gather(self.producer, return_exceptions=True)
        close = getattr(self.source, 'aclose', None)
        if close is not None and not self.exhausted:
            try:
                await close()
            except BaseException as error:
                self.close_error = error
        await self.raw.close(discard=True)

    def _chunk(self, chunk):
        runtime = self.runtime
        record = {'kind': 'mark', 'atof_version': '0.1', 'uuid': runtime._id(),
                  'parent_uuid': self.handle.uuid, 'timestamp': runtime.timestamp(), 'name': 'llm.chunk',
                  'data': chunk_receipt(self.chunk_index, chunk), 'data_schema': None,
                  'metadata': None, 'severity': None, 'category': None, 'category_profile': None}
        self.chunk_index += 1
        entries = runtime.guardrails.snapshot('mark_sanitize')
        _reserve(runtime, lambda: _sanitize_event(runtime, record, entries=entries), self.loop, self.context, self.callbacks)

    def _finish(self, termination, error=None):
        if self.ended:
            return
        self.ended = True
        runtime, at = self.runtime, self.runtime.timestamp()
        try:
            response = _json(self.context.copy().run(self.finalizer))
        except BaseException:
            runtime.logger.warning('Relay stream finalizer failed; omitting its payload', exc_info=True)
            response = None
        if response is None:
            response = deepcopy(self.handle.data)
        response_entries = runtime.guardrails.snapshot('llm_sanitize_response')
        metadata = _metadata(self.handle, termination, error)

        async def build():
            sanitized = response
            for _, _, _, callback in response_entries:
                try:
                    sanitized = _json(await runtime._await(callback(deepcopy(sanitized), runtime._codec_context(self.codec))))
                except Exception:
                    sanitized = None
                if sanitized is None:
                    break
            annotation = None
            if sanitized is not None and self.codec is not None:
                try:
                    annotation = self.codec.decode_response(sanitized)
                except Exception:
                    pass  # The source omits only the failed codec projection.
            prepared_metadata = deepcopy(metadata)
            if termination == 'dropped' and annotation is not None:
                usage, reason = annotation.usage, annotation.finish_reason
                authoritative = isinstance(usage, dict) and (usage.get('total_tokens') is not None or (
                    usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None))
                if authoritative and reason in ('complete', 'length', 'tool_use', 'content_filter'):
                    prepared_metadata['otel.status_code'] = 'OK'
                    prepared_metadata.pop('otel.status_description', None)
            record = runtime._record(self.handle, 'end', sanitized, prepared_metadata)
            record['timestamp'] = at
            if self.first_chunk is not None:
                record['category_profile']['time_to_first_chunk'] = self.first_chunk
            if annotation is not None:
                record['category_profile']['annotated_response'] = annotation.to_dict()
            return await _sanitize_event(runtime, record)

        _reserve(runtime, build, self.loop, self.context, self.callbacks)
        with runtime._STATE_LOCK:
            runtime._HANDLES.pop(self.handle.uuid, None)

    async def _forward(self):
        try:
            while True:
                try:
                    chunk = await self.raw.receive()
                except _Closed:
                    self._finish('complete')
                    break
                if isinstance(chunk, _Failure):
                    self._finish('failed', chunk.error)
                    await self.output.send(chunk)
                    break
                if self.first_chunk is None:
                    self.first_chunk = (time.monotonic_ns() - self.started) / 1_000_000_000
                self._chunk(chunk)
                try:
                    self.collector(deepcopy(chunk))
                except BaseException as error:
                    self._finish('failed', error)
                    await self.output.send(_Failure(error))
                    break
                await self.output.send(chunk)
        except (asyncio.CancelledError, _Closed):
            pass
        finally:
            await self._stop_source()
            self._finish('dropped')
            await self.output.close()

    def begin_close(self):
        if self.close_task is None:
            self.close_task = self.loop.create_task(self.close(), context=self.context.copy())
            self.close_task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return self.close_task

    async def close(self):
        await self.output.close(discard=True)
        if not self.manager.done():
            self.manager.cancel()
        await asyncio.gather(self.manager, return_exceptions=True)
        # A task cancelled before its first poll never enters its finally block.
        await self._stop_source()
        self._finish('dropped')
        if self.close_error is not None:
            raise RuntimeError('internal error: ' + type(self.close_error).__name__ + ': ' + str(self.close_error))


class LlmStream:
    """Public receiver; pumps retain only _State so receiver drop can cancel them."""
    def __init__(self, runtime, source, handle, collector, finalizer, codec, started, callbacks):
        self._state = _State(runtime, source, handle, collector, finalizer, codec, started, callbacks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        state = self._state
        async def receive():
            try:
                result = await state.output.receive()
            except _Closed:
                raise StopAsyncIteration from None
            if isinstance(result, _Failure):
                raise result.error
            return result
        if asyncio.get_running_loop() is state.loop:
            return await receive()
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(receive(), state.loop))

    async def aclose(self):
        state = self._state
        async def close():
            await asyncio.shield(state.begin_close())
        if asyncio.get_running_loop() is state.loop:
            await close()
        else:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(close(), state.loop))

    def __del__(self):
        state = getattr(self, '_state', None)
        if state is None or state.loop.is_closed() or state.manager.done():
            return
        state.loop.call_soon_threadsafe(state.begin_close)
