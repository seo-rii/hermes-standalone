"""Original Opus SILK packet, FEC, stereo and resampling state machine.

Adapted from BSD Opus 1.3.1 dec_API.c, decode_frame.c and decoder_set_fs.c.
Copyright (c) 2006-2011 Skype Limited; original notice: licenses/opus.txt.
"""
import copy
from types import SimpleNamespace

from standalone import opus_silk_params as parameters
from standalone import opus_silk_stereo as stereo
from standalone import opus_silk_tables as tables
from standalone import opus_silk_synthesis as synthesis
from standalone.opus_silk_resample import Resampler


def _channel():
    state=parameters.initialize(8,4)
    synthesis.initialize(state)
    state.fs_kHz=state.frame_length=state.fs_API_hz=0
    state.nb_subfr=state.subfr_length=state.ltp_mem_length=state.LPC_order=0
    state.lagPrev=state.LastGainIndex=0
    state.nFramesDecoded=state.nFramesPerPacket=0
    state.LBRR_flag=0;state.LBRR_flags=[0]*3
    state.resampler_state=None
    return state


def _set_frequency(state,frequency,api_rate,subframes):
    state.nb_subfr=subframes;state.subfr_length=frequency*5
    length=state.subfr_length*subframes
    if state.fs_kHz!=frequency or state.fs_API_hz!=api_rate:
        state.resampler_state=Resampler(frequency*1000,api_rate)
        state.fs_API_hz=api_rate
    if state.fs_kHz!=frequency:
        state.ltp_mem_length=frequency*20
        state.LPC_order=16 if frequency==16 else 10
        state.first_frame_after_reset=1
        state.lagPrev=100;state.LastGainIndex=10;state.prevSignalType=0
        state.outBuf[:]=[0]*len(state.outBuf)
        state.sLPC_Q14_buf[:]=[0]*16
    state.fs_kHz=frequency;state.frame_length=length


def _control():
    return SimpleNamespace(Gains_Q16=[0]*4,PredCoef_Q12=[[0]*16,[0]*16],
        pitchL=[0]*4,LTPCoef_Q14=[0]*20,LTP_scale_Q14=0)


def _decode_frame(state,decoder,lost,conditional):
    length=state.frame_length
    if lost==0 or lost==2 and state.LBRR_flags[state.nFramesDecoded]:
        parameters.decode_indices(state,decoder,state.nFramesDecoded,lost,conditional)
        pulses=parameters.decode_pulses(decoder,state.indices.signalType,state.indices.quantOffsetType,length)
        control=parameters.decode_parameters(state,conditional)
        pcm=synthesis.decode_core(state,control,pulses[:length])
        synthesis.plc(state,control,False)
        state.lossCnt=0;state.prevSignalType=state.indices.signalType
        state.first_frame_after_reset=0
    else:
        control=_control();state.indices.signalType=state.prevSignalType
        pcm=synthesis.plc(state,control,True)
    remaining=state.ltp_mem_length-length
    state.outBuf[:remaining]=state.outBuf[length:length+remaining]
    state.outBuf[remaining:remaining+length]=pcm
    synthesis.cng(state,control,pcm)
    synthesis.glue_frames(state,pcm)
    state.lagPrev=control.pitchL[state.nb_subfr-1]
    return pcm


class SilkDecoder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.channel_state=[_channel(),_channel()]
        self.sStereo=stereo.initialize()
        self.nChannelsAPI=self.nChannelsInternal=self.prev_decode_only_middle=0
        self.prev_pitch_lag=0

    def decode(self,range_decoder,*,internal_rate=16000,api_rate=48000,duration_ms=20,
               channels=1,out_channels=2,lost=False,fec=False):
        if any(type(value) is not int for value in (internal_rate,api_rate,duration_ms,channels,out_channels)) or type(lost) is not bool or type(fec) is not bool:
            raise ValueError('Opus SILK packet format requires integer rates and Boolean loss flags')
        if internal_rate not in (8000,12000,16000) or api_rate not in (8000,12000,16000,24000,48000):
            raise ValueError('Unsupported Opus SILK sampling rate')
        if duration_ms not in (0,10,20,40,60) or channels not in (1,2) or out_channels not in (1,2):
            raise ValueError('Invalid Opus SILK packet format')
        if lost and fec:raise ValueError('Opus SILK loss and FEC are separate operations')
        if range_decoder is None and not lost:raise ValueError('Opus SILK entropy state is required')
        flag=1 if lost else 2 if fec else 0
        frequency=internal_rate//1000;subframes=2 if duration_ms in (0,10) else 4
        frames=1 if duration_ms in (0,10,20) else duration_ms//20
        previous_channels=self.nChannelsInternal
        collapse=channels==1 and previous_channels==2 and frequency==self.channel_state[0].fs_kHz
        if channels>previous_channels:self.channel_state[1]=_channel()
        for state in self.channel_state[:channels]:
            state.nFramesDecoded=0;state.nFramesPerPacket=frames
            _set_frequency(state,frequency,api_rate,subframes)
        if out_channels==2 and channels==2 and (self.nChannelsAPI==1 or previous_channels==1):
            self.sStereo.pred_prev_Q13[:]=[0,0];self.sStereo.sSide[:]=[0,0]
            self.channel_state[1].resampler_state=copy.deepcopy(self.channel_state[0].resampler_state)
        self.nChannelsAPI,self.nChannelsInternal=out_channels,channels
        if not lost:
            self._header(range_decoder,channels,frames)
            if not fec:self._skip_redundancy(range_decoder,channels,frames)
        result=[]
        for frame in range(frames):
            result.extend(self._one(range_decoder,flag,channels,out_channels,collapse and frame==0))
        return result

    def _header(self,decoder,channels,frames):
        for state in self.channel_state[:channels]:
            for frame in range(frames):state.VAD_flags[frame]=decoder.bit_logp(1)
            state.LBRR_flag=decoder.bit_logp(1)
        for state in self.channel_state[:channels]:
            state.LBRR_flags[:]=[0]*3
            if state.LBRR_flag:
                symbol=1 if frames==1 else decoder.icdf(getattr(tables,'silk_LBRR_flags_'+str(frames)+'_iCDF'))+1
                for frame in range(frames):state.LBRR_flags[frame]=(symbol>>frame)&1

    def _skip_redundancy(self,decoder,channels,frames):
        for frame in range(frames):
            for number,state in enumerate(self.channel_state[:channels]):
                if not state.LBRR_flags[frame]:continue
                if channels==2 and number==0:
                    stereo.decode_pred(decoder)
                    if not self.channel_state[1].LBRR_flags[frame]:stereo.mid_only(decoder)
                conditional=2 if frame and state.LBRR_flags[frame-1] else 0
                parameters.decode_indices(state,decoder,frame,True,conditional)
                parameters.decode_pulses(decoder,state.indices.signalType,state.indices.quantOffsetType,state.frame_length)

    def _one(self,decoder,flag,channels,out_channels,collapse):
        mid,side=self.channel_state
        frame=mid.nFramesDecoded;only_mid=0;prediction=[0,0]
        if channels==2:
            if flag==0 or flag==2 and mid.LBRR_flags[frame]:
                prediction=stereo.decode_pred(decoder)
                if flag==0 and not side.VAD_flags[frame] or flag==2 and not side.LBRR_flags[frame]:
                    only_mid=stereo.mid_only(decoder)
            else:prediction[:]=self.sStereo.pred_prev_Q13
        if channels==2 and not only_mid and self.prev_decode_only_middle:
            side.outBuf[:]=[0]*len(side.outBuf);side.sLPC_Q14_buf[:]=[0]*16
            side.lagPrev=100;side.LastGainIndex=10;side.prevSignalType=0;side.first_frame_after_reset=1
        has_side=not only_mid if flag==0 else not self.prev_decode_only_middle or channels==2 and flag==2 and side.LBRR_flags[frame]
        buffers=[]
        for number,state in enumerate(self.channel_state[:channels]):
            if number==0 or has_side:
                if frame<=0:conditional=0
                elif flag==2:conditional=2 if state.LBRR_flags[frame-1] else 0
                elif number and self.prev_decode_only_middle:conditional=1
                else:conditional=2
                pcm=_decode_frame(state,decoder,flag,conditional)
            else:pcm=[0]*mid.frame_length
            buffers.append([0,0]+pcm);state.nFramesDecoded+=1
        length=mid.frame_length
        if out_channels==2 and channels==2:
            stereo.ms_to_lr(self.sStereo,buffers[0],buffers[1],prediction,mid.fs_kHz,length)
        else:
            buffers[0][:2]=self.sStereo.sMid
            self.sStereo.sMid[:]=buffers[0][length:length+2]
        output=[]
        for number in range(min(out_channels,channels)):
            output.append(self.channel_state[number].resampler_state.process(buffers[number][1:length+1]))
        if out_channels==2 and channels==1:
            output.append(side.resampler_state.process(buffers[0][1:length+1]) if collapse else list(output[0]))
        self.prev_pitch_lag=mid.lagPrev*{8:6,12:4,16:3}[mid.fs_kHz] if mid.prevSignalType==2 else 0
        if flag==1:
            for state in self.channel_state[:channels]:state.LastGainIndex=10
        else:self.prev_decode_only_middle=only_mid
        return [value for sample in zip(*output) for value in sample]
