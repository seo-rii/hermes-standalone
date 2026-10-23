"""Frozen native Relay streaming lifecycle and compact receipt contracts."""
import asyncio
import contextvars
import gc
import json
from pathlib import Path
import unittest

from standalone import relay
from standalone.relay_codecs import OpenAIChatCodec

ROOT=Path(__file__).with_name('fixtures')


async def run(mode,sanitize):
    events=[];late_events=[];counts={'produced':0,'collected':0,'finalized':0,'response_sanitized':0,'end_sanitized':0}
    relay.subscribers.register('stream-native-capture',lambda event:events.append(event.to_dict()))
    release=asyncio.Event();entered=asyncio.Event();drop_gate=asyncio.Event()
    def guard_response(response,context):
        counts['response_sanitized']+=1
        if sanitize=='response_null':return None
        if sanitize=='response_error':raise RuntimeError('private sanitizer canary')
        if sanitize=='response_redact':
            response={**response,'choices':[{'message':{'role':'assistant','content':'redacted'}}]}
        return response
    async def async_response(response,context):
        counts['response_sanitized']+=1;entered.set();await release.wait()
        return {**response,'choices':[{'message':{'role':'assistant','content':'async-redacted'}}]}
    def guard_end(event,fields):
        counts['end_sanitized']+=1
        if sanitize=='end_null':return {**fields,'data':None,'category_profile':None}
        if sanitize=='end_error':raise RuntimeError('private event canary')
        return fields
    relay.guardrails.register_llm_sanitize_response('stream-response',1,async_response if sanitize=='async_response' else guard_response)
    relay.guardrails.register_scope_sanitize_end('stream-end',1,guard_end)
    async def source(request):
        await asyncio.sleep(.005)
        if mode in ('subscriber_snapshot','subscriber_snapshot_error'):
            relay.subscribers.deregister('stream-native-capture')
            relay.subscribers.register('stream-native-late',lambda event:late_events.append(event.to_dict()))
        if mode in ('callback_error','subscriber_snapshot_error'):raise ValueError('owned callback failure')
        async def chunks():
            if mode=='source_error':raise ValueError('owned source failure')
            if mode=='empty':return
            for index in range(1000 if mode=='bounded_idle' else 2):
                if mode=='drop_before' or mode=='drop_after' and index==1:
                    await drop_gate.wait()
                if mode!='bounded_idle':await asyncio.sleep(.005)
                if mode=='after_error' and index==1:raise ValueError('owned late failure')
                counts['produced']+=1
                yield {'object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':'private chunk canary'}}]}
        if mode in ('cleanup_error','cleanup_error_complete'):
            class BrokenClose:
                def __init__(self):self.iterator=chunks()
                def __aiter__(self):return self
                async def __anext__(self):return await self.iterator.__anext__()
                async def aclose(self):
                    await self.iterator.aclose()
                    raise ValueError('owned cleanup failure')
            return BrokenClose()
        return chunks()
    def collector(chunk):
        counts['collected']+=1
        if mode=='collector_error':raise ValueError('owned collector failure')
    def finalizer():
        counts['finalized']+=1
        if mode=='finalizer_error':raise ValueError('owned finalizer failure')
        if mode in ('finalizer_null','fallback'):return None
        result={'model':'fixture-response','choices':[{'message':{'role':'assistant','content':'private final canary'}}]}
        if mode in ('authoritative_close','unknown_close'):
            result['choices'][0]['finish_reason']='stop'
            result['usage']={'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}
        if mode in ('unknown_finish','unknown_close'):
            result['choices'][0]['finish_reason']='vendor_finish'
        if mode=='extra_collision':
            result.update(message='owned extra message',finish_reason='owned extra finish',
                api_specific={'api':'owned-extra','collision':True},tool_calls=['owned extra call'])
        if mode=='decode_failure':
            result['choices'][0]['message']['content']=['invalid content array']
        return result
    error=None;idle=None;stream=None
    try:
        stream=await relay.llm.stream_execute('openai.chat',relay.LLMRequest({}, {'model':'fixture-request','messages':[]}),
            source,collector,finalizer,model_name='fixture-request',response_codec=OpenAIChatCodec(),
            data={'fallback':'owned-fallback'} if mode=='fallback' else None)
        if mode in ('idle_consumer','bounded_idle'):
            await asyncio.sleep(.08)
            idle=dict(counts)
        if mode in ('drop_before','drop_after'):
            if mode=='drop_after':await stream.__anext__()
            stream=None;gc.collect();await asyncio.sleep(.05)
        elif mode in ('close_before','bounded_idle'):
            await asyncio.wait_for(stream.aclose(),1)
        else:
            async def consume():
                async for chunk in stream:
                    if mode in ('close_after','authoritative_close','unknown_close','cleanup_error'):
                        await stream.aclose();break
            await asyncio.wait_for(consume(),1)
    except Exception as failure:
        error={'type':type(failure).__name__,'message':str(failure)}
    before_release=dict(counts)
    flush_waited=None
    if sanitize=='async_response':
        flushing=asyncio.create_task(relay.subscribers.flush_async())
        await asyncio.sleep(.01)
        flush_waited=not flushing.done()
        release.set();await flushing
    else:
        release.set();await relay.subscribers.flush_async()
    close_errors=[]
    if stream is not None:
        for repeat in range(2):
            try:await stream.aclose()
            except Exception as failure:close_errors.append({'type':type(failure).__name__,'message':str(failure)})
        await relay.subscribers.flush_async()
    relay.subscribers.deregister('stream-native-capture')
    relay.subscribers.deregister('stream-native-late')
    relay.guardrails.deregister_llm_sanitize_response('stream-response')
    relay.guardrails.deregister_scope_sanitize_end('stream-end')
    normalized=[]
    for event in events:
        if event.get('scope_category')=='end' or event.get('name')=='llm.chunk':
            row={key:event.get(key) for key in ('kind','scope_category','name','data','metadata','category','category_profile')}
            profile=row.get('category_profile')
            if profile and 'time_to_first_chunk' in profile:
                row['first_chunk_elapsed']=profile.pop('time_to_first_chunk')
                profile['time_to_first_chunk']='observed'
            normalized.append(row)
    return dict(id=mode+'-'+sanitize,mode=mode,sanitize=sanitize,error=error,late_events=late_events,close_errors=close_errors,flush_waited=flush_waited,idle=idle,before_release=before_release,counts=counts,events=normalized)



class StreamTests(unittest.TestCase):
    def test_stream_callbacks_keep_the_origin_context_and_clear_private_annotation(self):
        marker=contextvars.ContextVar('owned-stream-context',default='unset')
        async def scenario(close_before):
            observations=[];events=[]
            marker.set('owner')
            relay.subscribers.register('owned-context',lambda event:events.append(event))
            async def source(request):
                async def chunks():
                    observations.append(('source',marker.get()))
                    yield {'delta':'private source canary'}
                return chunks()
            def collector(chunk):observations.append(('collector',marker.get()))
            def finalizer():
                observations.append(('finalizer',marker.get()))
                return {'choices':[{'message':{'content':'private final canary'}}]}
            async def response_guard(response,context):
                await asyncio.sleep(0)
                observations.append(('response_guard',marker.get()))
                return response
            def end_guard(event,fields):
                observations.append(('end_guard',marker.get()))
                return {**fields,'data':None,'category_profile':None}
            relay.guardrails.register_llm_sanitize_response('owned-context-response',1,response_guard)
            relay.guardrails.register_scope_sanitize_end('owned-context-end',1,end_guard)
            try:
                stream=await relay.llm.stream_execute('owned.context',relay.LLMRequest({},{}),source,collector,finalizer,
                    response_codec=OpenAIChatCodec(),model_name='owned')
                marker.set('consumer')
                if close_before:await stream.aclose()
                else:
                    async for chunk in stream:pass
                await relay.subscribers.flush_async()
                self.assertEqual([value for name,value in observations],['owner']*len(observations))
                ends=[event for event in events if getattr(event,'scope_category',None)=='end']
                self.assertEqual(len(ends),1)
                self.assertIsNone(ends[0].data)
                self.assertIsNone(ends[0].category_profile)
                self.assertIsNone(ends[0].annotated_response)
                await stream.aclose()
            finally:
                relay.subscribers.deregister('owned-context')
                relay.guardrails.deregister_llm_sanitize_response('owned-context-response')
                relay.guardrails.deregister_scope_sanitize_end('owned-context-end')
        for close_before in (False,True):
            with self.subTest(close_before=close_before):asyncio.run(scenario(close_before))

    def test_original_native_compact_chunk_receipts(self):
        from standalone.relay_stream import chunk_receipt
        cases=json.loads((ROOT/'relay_stream_receipts.json').read_text())['cases']
        for index,case in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(chunk_receipt(index,case['chunk']),case['receipt'])

    def test_original_native_stream_lifecycle_and_privacy(self):
        cases=json.loads((ROOT/'relay_stream_native.json').read_text())['cases']
        for expected in cases:
            with self.subTest(case=expected['id']):
                actual=asyncio.run(asyncio.wait_for(run(expected['mode'],expected['sanitize']),5))
                self.assertEqual(actual['error'],expected['error'])
                self.assertEqual(actual['close_errors'],expected['close_errors'])
                self.assertEqual(actual['late_events'],expected['late_events'])
                self.assertEqual(actual['flush_waited'],expected['flush_waited'])
                self.assertEqual(actual['counts'],expected['counts'])
                self.assertEqual(actual['idle'],expected['idle'])
                for row in actual['events']:
                    elapsed=row.pop('first_chunk_elapsed',None)
                    if elapsed is not None:
                        self.assertGreaterEqual(elapsed,.004 if expected['mode']=='bounded_idle' else .009)
                        self.assertLess(elapsed,2)
                frozen=[{key:value for key,value in row.items() if key!='first_chunk_elapsed'} for row in expected['events']]
                self.assertEqual(actual['events'],frozen)
                if expected['sanitize']=='async_response':
                    self.assertEqual(actual['before_release']['end_sanitized'],0)

