"""Pinned faster-whisper segment, prompt, fallback and timestamp semantics.

Adapted from faster-whisper 1.2.1 (MIT). The original license is retained in
licenses/faster-whisper.txt. Inference is supplied by the qualified engine.
"""
from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import itertools
import logging
import statistics
import sys
import time
from typing import BinaryIO, Iterable, List, Optional, Tuple, Union
from warnings import warn
import zlib

from standalone.local_whisper_features import MelFeatures
from standalone.local_whisper_vad import SpeechTimestampsMap, VadOptions, collect_chunks, get_speech_timestamps


def _slice_features(features, start=0, stop=None):
    rows = features.rows if isinstance(features, MelFeatures) else features
    return MelFeatures(tuple(tuple(row[start:stop]) for row in rows))


def pad_or_trim(features, length=3000):
    return features.pad_or_trim(length)


class _Progress:
    def __init__(self, total, *, unit='seconds', disable=False):
        self.total, self.unit, self.disable = total, unit, disable
        self.value, self.last = 0, 0

    def update(self, value):
        self.value += value
        if not self.disable and time.monotonic()-self.last >= 1:
            self.last = time.monotonic()
            print(f'\r{self.value:.1f}/{self.total:.1f} {self.unit}', end='', file=sys.stderr, flush=True)

    def close(self):
        if not self.disable:
            print(file=sys.stderr)


tqdm = _Progress


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)

@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float]

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)

@dataclass
class TranscriptionOptions:
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]
    hotwords: Optional[str]

@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions

def restore_speech_timestamps(
    segments: Iterable[Segment],
    speech_chunks: List[dict],
    sampling_rate: int,
) -> Iterable[Segment]:
    ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

    for segment in segments:
        if segment.words:
            words = []
            for word in segment.words:
                # Ensure the word start and end times are resolved to the same chunk.
                middle = (word.start + word.end) / 2
                chunk_index = ts_map.get_chunk_index(middle)
                word.start = ts_map.get_original_time(word.start, chunk_index)
                word.end = ts_map.get_original_time(word.end, chunk_index)
                words.append(word)

            segment.start = words[0].start
            segment.end = words[-1].end
            segment.words = words

        else:
            segment.start = ts_map.get_original_time(segment.start)
            segment.end = ts_map.get_original_time(segment.end, is_end=True)

        yield segment

def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: Tuple[int],
) -> Optional[List[int]]:
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
            tokenizer.no_speech,
        ]
    )

    return tuple(sorted(set(suppress_tokens)))

def merge_punctuations(alignment: List[dict], prepended: str, appended: str) -> None:
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            # prepend it to the following word
            following["word"] = previous["word"] + following["word"]
            following["tokens"] = previous["tokens"] + following["tokens"]
            previous["word"] = ""
            previous["tokens"] = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            # append it to the previous word
            previous["word"] = previous["word"] + following["word"]
            previous["tokens"] = previous["tokens"] + following["tokens"]
            following["word"] = ""
            following["tokens"] = []
        else:
            i = j
        j += 1

def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )

def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )

class SegmentPipeline:
    def _split_segments_by_timestamps(
        self,
        tokenizer: Tokenizer,
        tokens: List[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ) -> List[List[int]]:
        current_segments = []
        single_timestamp_ending = (
            len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
        )

        consecutive_timestamps = [
            i
            for i in range(len(tokens))
            if i > 0
            and tokens[i] >= tokenizer.timestamp_begin
            and tokens[i - 1] >= tokenizer.timestamp_begin
        ]

        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
                end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
                start_time = (
                    time_offset + start_timestamp_position * self.time_precision
                )
                end_time = time_offset + end_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=start_time,
                        end=end_time,
                        tokens=sliced_tokens,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_position = (
                    tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * self.input_stride

        else:
            duration = segment_duration
            timestamps = [
                token for token in tokens if token >= tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                duration = last_timestamp_position * self.time_precision

            current_segments.append(
                dict(
                    seek=seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                )
            )

            seek += segment_size

        return current_segments, seek, single_timestamp_ending

    def generate_segments(
        self,
        features: MelFeatures,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        log_progress,
        encoder_output: Optional[object] = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - 1
        content_duration = float(content_frames * self.feature_extractor.time_per_frame)

        if isinstance(options.clip_timestamps, str):
            options.clip_timestamps = [
                float(ts)
                for ts in (
                    options.clip_timestamps.split(",")
                    if options.clip_timestamps
                    else []
                )
            ]

        seek_points: List[int] = [
            round(ts * self.frames_per_second) for ts in options.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
        last_speech_timestamp = 0.0
        # NOTE: This loop is obscurely flattened to make the diff readable.
        # A later commit should turn this into a simpler nested loop.
        # for seek_clip_start, seek_clip_end in seek_clips:
        #     while seek < seek_clip_end
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek_clip_end > content_frames:
                seek_clip_end = content_frames
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = seek * self.feature_extractor.time_per_frame
            window_end_time = float(
                (seek + self.feature_extractor.nb_max_frames)
                * self.feature_extractor.time_per_frame
            )
            segment_size = min(
                self.feature_extractor.nb_max_frames,
                content_frames - seek,
                seek_clip_end - seek,
            )
            segment = _slice_features(features, seek, seek + segment_size)
            segment_duration = segment_size * self.feature_extractor.time_per_frame
            segment = pad_or_trim(segment)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(time_offset)
                )

            previous_tokens = all_tokens[prompt_reset_since:]

            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            if options.multilingual:
                results = self.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]

                tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                tokenizer.language_code = language

            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
                hotwords=options.hotwords,
            )

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold

                if (
                    options.log_prob_threshold is not None
                    and avg_logprob > options.log_prob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

            previous_seek = seek

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: Optional[dict]) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: List[dict]) -> Optional[dict]:
                return next((s for s in segments if s["words"]), None)

            (
                current_segments,
                seek,
                single_timestamp_ending,
            ) = self._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=tokens,
                time_offset=time_offset,
                segment_size=segment_size,
                segment_duration=segment_duration,
                seek=seek,
            )

            if options.word_timestamps:
                self.add_word_timestamps(
                    [current_segments],
                    tokenizer,
                    encoder_output,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * self.frames_per_second)

                # skip silence before possible hallucinations
                if options.hallucination_silence_threshold is not None:
                    threshold = options.hallucination_silence_threshold

                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * self.frames_per_second)
                            continue

                    # skip silence before any possible hallucination that is surrounded
                    # by silence or more hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(
                                current_segments[si + 1 :]
                            )
                            if next_segment is not None:
                                hal_next_start = next_segment["words"][0]["start"]
                            else:
                                hal_next_start = time_offset + segment_duration
                            silence_before = (
                                segment["start"] - hal_last_end > threshold
                                or segment["start"] < threshold
                                or segment["start"] - time_offset < 2.0
                            )
                            silence_after = (
                                hal_next_start - segment["end"] > threshold
                                or is_segment_anomaly(next_segment)
                                or window_end_time - segment["end"] < 2.0
                            )
                            if silence_before and silence_after:
                                seek = round(
                                    max(time_offset + 1, segment["start"])
                                    * self.frames_per_second
                                )
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end
            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield Segment(
                    id=idx,
                    seek=previous_seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=result.no_speech_prob,
                    words=(
                        [Word(**word) for word in segment["words"]]
                        if options.word_timestamps
                        else None
                    ),
                )

            if (
                not options.condition_on_previous_text
                or temperature > options.prompt_reset_on_temperature
            ):
                if options.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        options.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)

            pbar.update(
                (min(content_frames, seek) - previous_seek)
                * self.feature_extractor.time_per_frame,
            )
        pbar.close()

    def generate_with_fallback(
        self,
        encoder_output: object,
        prompt: List[int],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ) -> Tuple[object, float, float, float]:
        decode_result = None
        all_results = []
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.max_length

        if max_length > self.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and `max_new_tokens` is: {max_length}. This exceeds the "
                f"`max_length` of the Whisper model: {self.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of `max_new_tokens`, "
                f"so that their combined length is less that {self.max_length}."
            )

        for temperature in options.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": options.beam_size,
                    "patience": options.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            # Recover the average log prob from the returned score.
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)

            text = tokenizer.decode(tokens).strip()
            compression_ratio = get_compression_ratio(text)

            decode_result = (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            )
            all_results.append(decode_result)

            needs_fallback = False

            if options.compression_ratio_threshold is not None:
                if compression_ratio > options.compression_ratio_threshold:
                    needs_fallback = True  # too repetitive

                    self.logger.debug(
                        "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
                        temperature,
                        compression_ratio,
                        options.compression_ratio_threshold,
                    )
                else:
                    below_cr_threshold_results.append(decode_result)

            if (
                options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = True  # average log probability is too low

                self.logger.debug(
                    "Log probability threshold is not met with temperature %.1f (%f < %f)",
                    temperature,
                    avg_logprob,
                    options.log_prob_threshold,
                )

            if (
                options.no_speech_threshold is not None
                and result.no_speech_prob > options.no_speech_threshold
                and options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break
        else:
            # all failed, select the result with the highest average log probability
            decode_result = max(
                below_cr_threshold_results or all_results, key=lambda x: x[1]
            )
            # to pass final temperature for prompt_reset_on_temperature
            decode_result = (
                decode_result[0],
                decode_result[1],
                temperature,
                decode_result[3],
            )

        return decode_result

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[int]:
        prompt = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
                if len(hotwords_tokens) >= self.max_length // 2:
                    hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
                prompt.extend(hotwords_tokens)
            if previous_tokens:
                prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.extend(tokenizer.sot_sequence)

        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            prefix_tokens = tokenizer.encode(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(prefix_tokens)

        return prompt

    def add_word_timestamps(
        self,
        segments: List[dict],
        tokenizer: Tokenizer,
        encoder_output: object,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        if len(segments) == 0:
            return

        text_tokens = []
        text_tokens_per_segment = []
        for segment in segments:
            segment_tokens = [
                [token for token in subsegment["tokens"] if token < tokenizer.eot]
                for subsegment in segment
            ]
            text_tokens.append(list(itertools.chain.from_iterable(segment_tokens)))
            text_tokens_per_segment.append(segment_tokens)

        alignments = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )
        median_max_durations = []
        for alignment in alignments:
            word_durations = [word["end"] - word["start"] for word in alignment
                              if word["end"] != word["start"]]
            median_duration = (
                statistics.median(word_durations) if len(word_durations) > 0 else 0.0
            )
            median_duration = min(0.7, float(median_duration))
            max_duration = median_duration * 2

            # hack: truncate long words at sentence boundaries.
            # a better segmentation algorithm based on VAD should be able to replace this.
            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
                # ensure words at sentence boundaries
                # are not longer than twice the median word duration.
                for i in range(1, len(alignment)):
                    if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                        if alignment[i]["word"] in sentence_end_marks:
                            alignment[i]["end"] = alignment[i]["start"] + max_duration
                        elif alignment[i - 1]["word"] in sentence_end_marks:
                            alignment[i]["start"] = alignment[i]["end"] - max_duration

            merge_punctuations(alignment, prepend_punctuations, append_punctuations)
            median_max_durations.append((median_duration, max_duration))

        for segment_idx, segment in enumerate(segments):
            word_index = 0
            time_offset = segment[0]["seek"] / self.frames_per_second
            median_duration, max_duration = median_max_durations[segment_idx]
            for subsegment_idx, subsegment in enumerate(segment):
                saved_tokens = 0
                words = []

                while word_index < len(alignments[segment_idx]) and saved_tokens < len(
                    text_tokens_per_segment[segment_idx][subsegment_idx]
                ):
                    timing = alignments[segment_idx][word_index]

                    if timing["word"]:
                        words.append(
                            dict(
                                word=timing["word"],
                                start=round(time_offset + timing["start"], 2),
                                end=round(time_offset + timing["end"], 2),
                                probability=timing["probability"],
                            )
                        )

                    saved_tokens += len(timing["tokens"])
                    word_index += 1

                # hack: truncate long words at segment boundaries.
                # a better segmentation algorithm based on VAD should be able to replace this.
                if len(words) > 0:
                    # ensure the first and second word after a pause is not longer than
                    # twice the median word duration.
                    if words[0][
                        "end"
                    ] - last_speech_timestamp > median_duration * 4 and (
                        words[0]["end"] - words[0]["start"] > max_duration
                        or (
                            len(words) > 1
                            and words[1]["end"] - words[0]["start"] > max_duration * 2
                        )
                    ):
                        if (
                            len(words) > 1
                            and words[1]["end"] - words[1]["start"] > max_duration
                        ):
                            boundary = max(
                                words[1]["end"] / 2, words[1]["end"] - max_duration
                            )
                            words[0]["end"] = words[1]["start"] = boundary
                        words[0]["start"] = max(0, words[0]["end"] - max_duration)

                    # prefer the segment-level start timestamp if the first word is too long.
                    if (
                        subsegment["start"] < words[0]["end"]
                        and subsegment["start"] - 0.5 > words[0]["start"]
                    ):
                        words[0]["start"] = max(
                            0,
                            min(words[0]["end"] - median_duration, subsegment["start"]),
                        )
                    else:
                        subsegment["start"] = words[0]["start"]

                    # prefer the segment-level end timestamp if the last word is too long.
                    if (
                        subsegment["end"] > words[-1]["start"]
                        and subsegment["end"] + 0.5 < words[-1]["end"]
                    ):
                        words[-1]["end"] = max(
                            words[-1]["start"] + median_duration, subsegment["end"]
                        )
                    else:
                        subsegment["end"] = words[-1]["end"]

                    last_speech_timestamp = subsegment["end"]
                segments[segment_idx][subsegment_idx]["words"] = words
        return last_speech_timestamp

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        text_tokens: List[int],
        encoder_output: object,
        num_frames: int,
        median_filter_width: int = 7,
    ) -> List[dict]:
        if len(text_tokens) == 0:
            return []

        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            text_tokens,
            num_frames,
            median_filter_width=median_filter_width,
        )
        return_list = []
        for result, text_token in zip(results, text_tokens):
            text_token_probs = result.text_token_probs
            alignments = result.alignments
            text_indices = [pair[0] for pair in alignments]
            time_indices = [pair[1] for pair in alignments]

            words, word_tokens = tokenizer.split_to_word_tokens(
                text_token + [tokenizer.eot]
            )
            if len(word_tokens) <= 1:
                # return on eot only
                # >>> np.pad([], (1, 0))
                # array([0.])
                # This results in crashes when we lookup jump_times with float, like
                # IndexError: arrays used as indices must be of integer (or boolean) type
                return_list.append([])
                continue
            word_boundaries = [0] + list(itertools.accumulate(len(t) for t in word_tokens[:-1]))
            if len(word_boundaries) <= 1:
                return_list.append([])
                continue

            jumps = [True] + [first != second for first, second in zip(text_indices, text_indices[1:])]
            jump_times = [t / self.tokens_per_second for t, jump in zip(time_indices, jumps) if jump]
            start_times = [jump_times[index] for index in word_boundaries[:-1]]
            end_times = [jump_times[index] for index in word_boundaries[1:]]
            word_probabilities = [
                sum(text_token_probs[i:j]) / (j-i)
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            return_list.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )
        return return_list

    def detect_language(
        self,
        audio: Optional[MelFeatures] = None,
        features: Optional[MelFeatures] = None,
        vad_filter: bool = False,
        vad_parameters: Union[dict, VadOptions] = None,
        language_detection_segments: int = 1,
        language_detection_threshold: float = 0.5,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Use Whisper to detect the language of the input audio or features.

        Arguments:
            audio: Input audio signal, must be a 1D float array sampled at 16khz.
            features: Input Mel spectrogram features, must be a float array with
                shape (n_mels, n_frames), if `audio` is provided, the features will be ignored.
                Either `audio` or `features` must be provided.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.

        Returns:
            language: Detected language.
            language_probability: Probability of the detected language.
            all_language_probs: List of tuples with all language names and probabilities.
        """
        assert (
            audio is not None or features is not None
        ), "Either `audio` or `features` must be provided."

        if audio is not None:
            if vad_filter:
                speech_chunks = get_speech_timestamps(audio, vad_parameters)
                audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
                audio = array('f', itertools.chain.from_iterable(audio_chunks))

            audio = audio[
                : language_detection_segments * self.feature_extractor.n_samples
            ]
            features = self.feature_extractor(audio)

        features = _slice_features(features, 0, language_detection_segments * self.feature_extractor.nb_max_frames)

        detected_language_info = {}
        for i in range(0, features.shape[-1], self.feature_extractor.nb_max_frames):
            encoder_output = self.encode(
                pad_or_trim(_slice_features(features, i, i + self.feature_extractor.nb_max_frames))
            )
            # results is a list of tuple[str, float] with language names and probabilities.
            results = self.model.detect_language(encoder_output)[0]

            # Parse language names to strip out markers
            all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
            # Get top language token and probability
            language, language_probability = all_language_probs[0]
            if language_probability > language_detection_threshold:
                break
            detected_language_info.setdefault(language, []).append(language_probability)
        else:
            # If no language detected for all segments, the majority vote of the highest
            # projected languages for all segments is used to determine the language.
            language = max(
                detected_language_info,
                key=lambda lang: len(detected_language_info[lang]),
            )
            language_probability = max(detected_language_info[language])

        return language, language_probability, all_language_probs
