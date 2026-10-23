"""Explicit discord.py codec boundary using the embedded Python Opus codec.

The sender selects legal CELT CBR encoding. Requested SILK in-band FEC is
reported separately from effective FEC, which is unavailable in this mode.
Discord API portions copyright (c) 2015-present Rapptz, MIT; the original
notice is retained in licenses/opus-decoder-notices.txt.
"""
import logging

from discord.errors import DiscordException
from .opus_celt_encoder import CeltEncoder
from .opus_decoder import Decoder as _Decoder, DecodeError
from .opus_range import OpusRangeError


_log = logging.getLogger('discord.opus')
_APPLICATIONS = {'audio':2049,'voip':2048,'lowdelay':2051}
_BANDS = ('narrow','medium','wide','superwide','full')
_SIGNALS = ('auto','voice','music')
_ERRORS = {0:'success',-1:'invalid argument',-2:'buffer too small',-3:'internal error',
           -4:'corrupted stream',-5:'request not implemented',-6:'invalid state',-7:'memory allocation failed'}


class OpusError(DiscordException):
    def __init__(self,code):
        self.code = code
        super().__init__(_ERRORS.get(code,'unknown error'))


class OpusNotLoaded(DiscordException):
    pass


def _call(function,*args,**kwargs):
    try:
        return function(*args,**kwargs)
    except DecodeError as error:
        raise OpusError(error.code) from error
    except OpusRangeError as error:
        raise OpusError(-2) from error
    except ValueError as error:
        raise OpusError(-1) from error


def get_opus_version():
    return _Decoder.get_opus_version()


def is_loaded():
    """The Python codec is imported with this module and needs no loader."""
    return True


def load_opus(name):
    raise OpusNotLoaded('The codec is embedded; external native Opus loading is disabled')


class _OpusStruct:
    SAMPLING_RATE = 48000
    CHANNELS = 2
    FRAME_LENGTH = 20
    SAMPLE_SIZE = 4
    SAMPLES_PER_FRAME = 960
    FRAME_SIZE = 3840
    get_opus_version = staticmethod(get_opus_version)


class Decoder(_Decoder):
    def decode(self,data,*,fec=False):
        return _call(super().decode,data,fec=fec)

    def _set_gain(self,adjustment):
        return _call(super()._set_gain,adjustment)

    @staticmethod
    def packet_get_nb_frames(data):
        return _call(_Decoder.packet_get_nb_frames,data)

    @staticmethod
    def packet_get_nb_channels(data):
        return _call(_Decoder.packet_get_nb_channels,data)

    @classmethod
    def packet_get_samples_per_frame(cls,data):
        return _call(_Decoder.packet_get_samples_per_frame,data)


class Encoder(_OpusStruct):
    codec_mode = 'celt'
    fec_enabled = False

    def __init__(self,*,application='audio',bitrate=128,fec=True,expected_packet_loss=.15,
                 bandwidth='full',signal_type='auto'):
        if application not in _APPLICATIONS:
            raise ValueError(f'{application} is not a valid application setting. Try one of: {"".join(_APPLICATIONS)}')
        if not 16 <= bitrate <= 512:
            raise ValueError(f'bitrate must be between 16 and 512, not {bitrate}')
        if not 0 < expected_packet_loss <= 1.:
            raise ValueError(f'expected_packet_loss must be a positive number less than or equal to 1, not {expected_packet_loss}')
        self.application = _APPLICATIONS[application]
        self._encoder = CeltEncoder(self.CHANNELS)
        self.expected_packet_loss_percent = 0
        self.set_bitrate(bitrate)
        self.set_fec(fec)
        if fec:
            self.set_expected_packet_loss_percent(expected_packet_loss)
        self.set_bandwidth(bandwidth)
        self.set_signal_type(signal_type)

    def set_bitrate(self,kbps):
        kbps = min(512,max(16,int(kbps)))
        self._encoder.set_bitrate(kbps * 1024)
        return kbps

    def set_bandwidth(self,req):
        if req not in _BANDS:
            raise KeyError(f'{req!r} is not a valid bandwidth setting. Try one of: {",".join(_BANDS)}')
        self.bandwidth_requested = req
        self._encoder.set_bandwidth(req)

    def set_signal_type(self,req):
        if req not in _SIGNALS:
            raise KeyError(f'{req!r} is not a valid signal type setting. Try one of: {",".join(_SIGNALS)}')
        # The explicit CELT allocation policy does not use this analysis hint.
        self.signal_type_requested = req

    def set_fec(self,enabled=True):
        self.fec_requested = bool(enabled)
        if self.fec_requested:
            _log.warning('Standalone Opus sends CELT without in-band FEC; fec=True records the request only')

    def set_expected_packet_loss_percent(self,percentage):
        self.expected_packet_loss_percent = min(100,max(0,int(percentage * 100)))

    def encode(self,pcm,frame_size):
        return _call(self._encoder.encode,pcm,frame_size)
