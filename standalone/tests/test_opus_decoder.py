"""Whole native Opus decoding, transitions and Discord PCM contracts."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

from standalone.opus_decoder import DecodeError, Decoder, OpusDecoder, soft_clip


def _fixture(name):
    wrapper=json.loads((Path(__file__).parent/'fixtures'/name).read_text())
    data=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
    if hashlib.sha256(data).hexdigest()!=wrapper['data_sha256']:
        raise AssertionError('Native Opus wrapper fixture digest mismatch')
    return json.loads(data)


def _state(decoder):
    modes={0:0,'silk':1000,'hybrid':1001,'celt':1002}
    return [decoder.stream_channels,decoder.bandwidth,modes[decoder.mode],modes[decoder.prev_mode],
        decoder.frame_size,int(decoder.prev_redundancy),decoder.last_packet_duration,decoder.final_range,
        decoder.gain_q8,decoder._silk_rate,decoder._silk_channels,decoder._silk_payload_ms,
        decoder._silk_pitch,decoder.pitch,int(decoder.phase_inversion_disabled)]


class OpusDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.streams=_fixture('opus_decoder.json')['streams']

    def _replay(self,predicate):
        count=features=0
        for stream in self.streams:
            if not predicate(stream['name']):
                continue
            decoder=OpusDecoder(stream['sample_rate'],stream['channels'])
            for number,frame in enumerate(stream['frames']):
                with self.subTest(stream=stream['name'],frame=number):
                    if frame.get('reset'):
                        decoder.reset()
                    if 'gain' in frame:
                        decoder.set_gain(frame['gain'])
                    if 'phase' in frame:
                        decoder.set_phase_inversion_disabled(frame['phase'])
                    packet=frame['packet']
                    packet=None if packet is None else base64.b64decode(packet,validate=True)
                    integer=frame.get('integer',False)
                    actual=(decoder.decode if integer else decoder.decode_float)(packet,frame['frame_size'],fec=frame.get('fec',False))
                    encoded=base64.b64decode(frame['output'],validate=True)
                    code='h' if integer else 'f'
                    expected=struct.unpack('<'+str(len(encoded)//(2 if integer else 4))+code,encoded)
                    self.assertEqual(len(actual),len(expected))
                    self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,expected)),2 if integer else 2/32768)
                    self.assertEqual(_state(decoder),frame['state'])
                    self.assertLess(max(abs(a-b) for a,b in zip(decoder.softclip_memory,frame['softclip'])),2e-6)
                    features|=frame['features']
                    count+=1
        return count,features

    def test_native_all_modes_rates_channels_and_multiple_frames(self):
        count,_=self._replay(lambda name: not name.startswith(('transitions-','transition-losses-','gain-softclip-')) and 'loss-fec-' not in name)
        self.assertGreater(count,600)

    def test_native_fec_repeated_loss_short_plc_and_mode_redundancy(self):
        count,features=self._replay(lambda name:name.startswith(('transitions-','transition-losses-')) or 'loss-fec-' in name)
        self.assertGreater(count,300)
        self.assertEqual(features&7,7,'Both CELT redundancy directions and transition fades must be exercised')

    def test_native_gain_softclip_phase_and_reset(self):
        count,_=self._replay(lambda name:name.startswith('gain-softclip-'))
        self.assertGreater(count,30)

    def test_discord_pcm_decoder_packet_duration_volume_and_loss(self):
        stream=next(stream for stream in self.streams if stream['name'].startswith('celt-multiframe-') and stream['channels']==2 and len(base64.b64decode(stream['frames'][0]['output']))==960*2*2)
        api=Decoder();reference=OpusDecoder()
        for frame in stream['frames']:
            packet=base64.b64decode(frame['packet'])
            self.assertEqual(api.decode(packet),reference.decode_pcm16(packet,5760))
            self.assertEqual(api._get_last_packet_duration(),reference.last_packet_duration)
            self.assertEqual(api.packet_get_nb_frames(packet)*api.packet_get_samples_per_frame(packet),reference.last_packet_duration)
        self.assertEqual(api.decode(None),reference.decode_pcm16(None,reference.last_packet_duration))
        self.assertEqual(api.set_volume(2),0)
        self.assertEqual(api.gain_q8,1541)
        api.reset()
        self.assertEqual(api.gain_q8,1541)
        self.assertEqual(len(api.decode(b'\xf8\xff\xfe')),3840)
        with self.assertRaises(TypeError):
            api.decode(None,fec=True)


class OpusDecoderBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extra=_fixture('opus_decoder_extra.json')

    def test_native_softclip_continuation_and_first_peak_ramps(self):
        for stream in self.extra['softclip']:
            memory=[0.,0.]
            for number,frame in enumerate(stream['frames']):
                with self.subTest(channels=stream['channels'],frame=number):
                    actual=list(frame['input'])
                    self.assertIs(soft_clip(actual,stream['channels'],memory),actual)
                    self.assertEqual(actual,frame['output'])
                    self.assertEqual(memory[:stream['channels']],frame['memory'])

    def test_native_errors_preserve_state_and_recovery(self):
        for stream in self.extra['errors']:
            decoder=OpusDecoder(stream['sample_rate'],stream['channels'])
            for number,frame in enumerate(stream['frames']):
                with self.subTest(stream=stream['name'],frame=number):
                    data=frame['packet']
                    data=None if data is None else base64.b64decode(data,validate=True)
                    if frame['result']<0:
                        with self.assertRaises(DecodeError) as raised:
                            decoder.decode_float(data,frame['frame_size'],fec=frame.get('fec',False))
                        self.assertEqual(raised.exception.code,frame['result'])
                    else:
                        actual=decoder.decode_float(data,frame['frame_size'],fec=frame.get('fec',False))
                        self.assertEqual(len(actual),frame['result']*stream['channels'])
                        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,frame['output'])),2/32768)
                    self.assertEqual(_state(decoder),frame['state'])


if __name__=='__main__':
    unittest.main()
