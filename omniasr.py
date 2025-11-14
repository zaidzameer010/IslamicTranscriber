from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchConfig,
)
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

SAMPLING_RATE = 16000

vad_model = load_silero_vad()


def infer_device(arg: Optional[str]) -> Optional[str]:
    if arg is None or arg == "auto":
        if torch is not None:
            try:
                if torch.cuda.is_available():  # type: ignore[union-attr]
                    return "cuda"
            except Exception:
                pass
        return "cpu"
    return arg


def infer_dtype(arg: Optional[str]) -> Optional["torch.dtype"]:
    if torch is None:
        return None
    if arg is None or arg == "auto":
        try:
            if torch.cuda.is_available():  # type: ignore[union-attr]
                return torch.bfloat16
        except Exception:
            return None
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(arg)


def format_srt_time(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def build_beam_config(args) -> Optional[Wav2Vec2LlamaBeamSearchConfig]:
    use_beam = (
        args.beam_size is not None
        or args.length_norm
        or args.compression_window is not None
        or args.compression_threshold is not None
    )
    if not use_beam:
        return None
    return Wav2Vec2LlamaBeamSearchConfig(
        nbest=args.beam_size or 5,
        length_norm=args.length_norm,
        compression_window=args.compression_window or 100,
        compression_threshold=args.compression_threshold or 4.0,
    )


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="omniasr",
        description="Transcribe audio files with Omnilingual-ASR",
    )
    parser.add_argument(
        "audio",
        nargs="*",
        help="Audio files to transcribe",
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_CTC_1B",
        help="Model card name, e.g. omniASR_LLM_1B, omniASR_CTC_1B",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (auto=CUDA if available, else CPU)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Computation dtype (auto picks a reasonable default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for transcription",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default="urd_Arab",
        help=(
            "Language codes like eng_Latn; pass one code to use for all files "
            "or one code per input file"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write transcripts to (defaults to alongside audio)",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default="srt",
        help="Extension for transcript files (default: srt)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=6,
        help="Beam size (nbest) for LLM decoding",
    )
    parser.add_argument(
        "--length-norm",
        action="store_true",
        help="Enable length normalization in beam search",
    )
    parser.add_argument(
        "--compression-window",
        type=int,
        default=None,
        help="Repetition compression window size",
    )
    parser.add_argument(
        "--compression-threshold",
        type=float,
        default=None,
        help="Repetition compression threshold",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=38.0,
        help="Maximum duration per chunk in seconds",
    )
    return parser.parse_args(argv)


def prepare_langs(
    audio_paths: List[Path], langs: Optional[List[str]]
) -> Optional[List[str]]:
    if not langs:
        return None
    if len(langs) == 1:
        return [langs[0]] * len(audio_paths)
    if len(langs) != len(audio_paths):
        raise SystemExit(
            "Number of --lang codes must be 1 or equal to number of audio files"
        )
    return langs


def find_audio_files() -> List[Path]:
    cwd = Path.cwd()
    files: List[Path] = []
    for pattern in ("*.mp3", "*.MP3", "*.mp4", "*.MP4"):
        files.extend(cwd.glob(pattern))
    return sorted(files, key=lambda p: p.name.lower())


def select_file(console: Console, files: List[Path]) -> Path:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", width=4)
    table.add_column("File", overflow="fold")
    for i, f in enumerate(files, 1):
        table.add_row(str(i), f.name)
    console.print(table)
    choices = [str(i) for i in range(1, len(files) + 1)]
    selected = Prompt.ask("Select a file", choices=choices, default="1")
    return files[int(selected) - 1]


def merge_segments(
    segments: List[Tuple[int, int]],
    max_seconds: float,
    merge_gap_s: float = 0.4,
) -> List[Tuple[int, int]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    max_len_samples = int(max_seconds * SAMPLING_RATE) if max_seconds > 0 else 0
    merge_gap_samples = int(merge_gap_s * SAMPLING_RATE) if merge_gap_s > 0 else 0
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        gap = start - cur_end
        if (
            merge_gap_samples
            and gap <= merge_gap_samples
            and (not max_len_samples or end - cur_start <= max_len_samples)
        ):
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    if not max_len_samples:
        return merged

    final: List[Tuple[int, int]] = []
    for start, end in merged:
        cur = start
        while max_len_samples and end - cur > max_len_samples:
            seg_end = cur + max_len_samples
            final.append((cur, seg_end))
            cur = seg_end
        final.append((cur, end))
    return final


def transcribe_with_segments(
    pipeline: ASRInferencePipeline,
    path: Path,
    lang: Optional[str],
    batch_size: int,
    max_seconds: float,
    console: Console,
) -> str:
    wav = read_audio(str(path), sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        threshold=0.5,
        sampling_rate=SAMPLING_RATE,
        min_speech_duration_ms=150,
        min_silence_duration_ms=600,
        speech_pad_ms=200,
    )

    segments: List[Tuple[int, int]] = []
    if speech_timestamps:
        for ts in speech_timestamps:
            start = int(ts["start"])
            end = int(ts["end"])
            segments.append((start, end))
    else:
        total = wav.shape[-1]
        segments = [(0, int(total))]

    segments = merge_segments(segments, max_seconds=max_seconds, merge_gap_s=0.4)

    inputs = [
        {"waveform": wav[start:end], "sample_rate": SAMPLING_RATE}
        for (start, end) in segments
    ]

    results: List[str] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )
    with progress:
        task = progress.add_task(f"Transcribing {path.name}", total=len(inputs))
        for idx in range(0, len(inputs), batch_size):
            batch_inputs = inputs[idx : idx + batch_size]
            if lang is not None:
                lang_batch = [lang] * len(batch_inputs)
            else:
                lang_batch = None
            texts = pipeline.transcribe(
                batch_inputs,
                batch_size=min(batch_size, len(batch_inputs)),
                lang=lang_batch,
            )
            results.extend(texts)
            progress.update(task, advance=len(batch_inputs))

    lines: List[str] = []
    for i, ((start, end), text) in enumerate(zip(segments, results), start=1):
        start_sec = float(start) / SAMPLING_RATE
        end_sec = float(end) / SAMPLING_RATE
        lines.append(str(i))
        lines.append(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}")
        lines.append((text or "").strip())
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    console = Console()
    args = parse_args(argv)
    if args.audio:
        audio_paths = [Path(p) for p in args.audio]
    else:
        files = find_audio_files()
        if not files:
            console.print(
                "[bold red]No .mp3 or .mp4 files found in the current directory.[/bold red]"
            )
            return
        target = select_file(console, files)
        audio_paths = [target]

    missing = [str(p) for p in audio_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing audio files: {', '.join(missing)}")

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype)
    beam_config = build_beam_config(args)

    pipeline_kwargs = {"model_card": args.model_card}
    if device is not None:
        pipeline_kwargs["device"] = device
    if dtype is not None:
        pipeline_kwargs["dtype"] = dtype
    if beam_config is not None:
        pipeline_kwargs["beam_search_config"] = beam_config

    pipeline = ASRInferencePipeline(**pipeline_kwargs)

    langs = prepare_langs(audio_paths, args.lang)
    max_seconds = args.segment_seconds

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    ext = "." + args.output_ext.lstrip(".")

    for idx, path in enumerate(audio_paths):
        lang = langs[idx] if langs else None
        text = transcribe_with_segments(
            pipeline=pipeline,
            path=path,
            lang=lang,
            batch_size=args.batch_size,
            max_seconds=max_seconds,
            console=console,
        )
        if out_dir is not None:
            out_path = out_dir / (path.stem + ext)
        else:
            out_path = path.with_suffix(ext)
        out_path.write_text(text, encoding="utf-8")
        print(str(out_path))


if __name__ == "__main__":
    main()
