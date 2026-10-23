"""Actual native Opus SILK packet and FEC/PLC state sequences, stdlib only."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest

from standalone.opus_range import OpusRangeDecoder
from standalone.opus_silk import SilkDecoder


class OpusSilkPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import zlib
        fixture=json.loads((Path(__file__).parent/'fixtures/opus_silk_packets.json').read_text())
        data=zlib.decompress(base64.b64decode(fixture['data_zlib_base64'],validate=True))
        if hashlib.sha256(data).hexdigest()!=fixture['data_sha256']:
            raise AssertionError('Opus SILK packet fixture digest mismatch')
        cls.streams=json.loads(data)['streams']

    def test_native_all_rates_stereo_fec_losses_recovery_and_format_changes(self):
        self.assertEqual(len(self.streams),147)
        count=0
        for stream in self.streams:
            decoder=SilkDecoder()
            for number,frame in enumerate(stream['frames']):
                with self.subTest(stream=stream['name'],frame=number):
                    settings=frame['settings'];flag=frame['action']
                    entropy=OpusRangeDecoder(b'' if flag==1 else base64.b64decode(settings['payload'],validate=True))
                    pcm=decoder.decode(entropy,internal_rate=settings['internal_rate'],api_rate=stream['api_rate'],
                        duration_ms=settings['duration_ms'],channels=settings['channels'],out_channels=stream['output_channels'],lost=flag==1,fec=flag==2)
                    self.assertEqual(struct.pack('<'+'h'*len(pcm),*pcm),base64.b64decode(frame['pcm16le'],validate=True))
                    state=[getattr(entropy,name) for name in ('storage','end_offs','end_window','nend_bits','nbits_total','offs','rng','val','ext','rem','error')]+[entropy.tell(),entropy.tell_frac()]
                    for channel in decoder.channel_state:
                        state+=[getattr(channel,name) for name in ('fs_kHz','fs_API_hz','nb_subfr','frame_length','ltp_mem_length','nFramesDecoded','nFramesPerPacket','lossCnt','LastGainIndex','first_frame_after_reset','lagPrev','prevSignalType','prev_gain_Q16')]
                        state+=channel.prevNLSF_Q15+[channel.ec_prevSignalType,channel.ec_prevLagIndex]+channel.VAD_flags+[channel.LBRR_flag]+channel.LBRR_flags
                    state+=decoder.sStereo.pred_prev_Q13+decoder.sStereo.sMid+decoder.sStereo.sSide
                    state+=[decoder.nChannelsAPI,decoder.nChannelsInternal,decoder.prev_decode_only_middle,decoder.prev_pitch_lag]
                    self.assertEqual(state,frame['state'])
                    count+=1
        self.assertEqual(count,1014)

    def test_invalid_formats_leave_decoder_unmodified(self):
        decoder=SilkDecoder()
        from copy import deepcopy
        initial=deepcopy(decoder.__dict__)
        for settings in ({'internal_rate':24000},{'api_rate':44100},{'duration_ms':80},
                         {'channels':3},{'out_channels':0},{'channels':True},
                         {'api_rate':16000.0},{'fec':True,'lost':True},{'lost':1}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                decoder.decode(OpusRangeDecoder(b''),**settings)
        self.assertEqual(repr(decoder.__dict__),repr(initial))


if __name__=='__main__':
    unittest.main()
