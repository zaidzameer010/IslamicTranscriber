from pathlib import Path
from datetime import timedelta, datetime

try:
    import torch
    _torch_cuda = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    _torch_cuda = False
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt
import srt
import json
import numpy as np
import librosa
import noisereduce as nr
from typing import Optional
import inspect
from scipy.signal import butter, filtfilt, wiener
from types import SimpleNamespace

def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def format_timestamp_vtt(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

def preprocess_audio(input_path: Path, cfg: dict, console: Console) -> Optional[np.ndarray]:
    """Load, denoise, and normalize audio; return mono waveform at 16kHz.
    If loading fails (e.g., some MP4s), return None to enable fallback.
    """
    try:
        y, sr = librosa.load(str(input_path), sr=16000, mono=True)
    except Exception as e:
        console.log(f"[yellow]librosa load failed ({e}); falling back to raw file input")
        return None
    pp = cfg.get("preprocess", {})
    # Optional high-pass / low-pass filters
    hp = pp.get("highpass_hz")
    lp = pp.get("lowpass_hz")
    if isinstance(hp, (int, float)) and hp and hp > 0:
        try:
            b, a = butter(4, hp, btype='highpass', fs=sr)
            y = filtfilt(b, a, y)
        except Exception as e:
            console.log(f"[yellow]High-pass filter skipped: {e}")
    if isinstance(lp, (int, float)) and lp and lp > 0:
        try:
            b, a = butter(4, lp, btype='lowpass', fs=sr)
            y = filtfilt(b, a, y)
        except Exception as e:
            console.log(f"[yellow]Low-pass filter skipped: {e}")
    if cfg.get("preprocess", {}).get("denoise", True):
        try:
            y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=cfg.get("preprocess", {}).get("denoise_prop_decrease", 0.8))
        except Exception as e:
            console.log(f"[yellow]Noise reduction skipped due to error: {e}")
    # Optional basic dereverb via Wiener filter
    if pp.get("dereverb", False):
        try:
            y = wiener(y)
        except Exception as e:
            console.log(f"[yellow]Dereverb (Wiener) skipped: {e}")
    # Optional preemphasis
    if pp.get("preemphasis", False):
        try:
            coef = float(pp.get("preemphasis_coef", 0.97))
            y = librosa.effects.preemphasis(y, coef=coef)
        except Exception as e:
            console.log(f"[yellow]Preemphasis skipped: {e}")
    if cfg.get("preprocess", {}).get("normalize", True):
        max_abs = float(np.max(np.abs(y))) if y.size else 1.0
        if max_abs > 0:
            y = (y / max_abs) * 0.95
    return y.astype(np.float32)

def _split_words_into_chunks(words_ts, seg_cfg: dict):
    """Split a list of word timestamp objects into chunks by limits and punctuation.
    Returns list of dicts: {start, end, text}.
    """
    max_words = int(seg_cfg.get("max_words", 20))
    max_chars = int(seg_cfg.get("max_chars", 42))
    max_dur = float(seg_cfg.get("max_duration", 6.0))
    min_dur = float(seg_cfg.get("min_duration", 1.0))
    # End-of-sentence punctuation only (avoid mid-sentence breaks)
    punct_set = set([".", "!", "?", "؟", "۔", "…"])  # include RTL equivalents

    # Hard limits act as safety net to avoid unbounded segments when punctuation is absent
    hard_factor = float(seg_cfg.get("hard_factor", 1.8))
    hard_max_words = int(max_words * hard_factor)
    hard_max_chars = int(max_chars * hard_factor)
    hard_max_dur = float(max_dur * hard_factor)

    chunks = []
    cur = []
    cur_start = None
    for w in words_ts:
        if cur_start is None:
            cur_start = w.start
        cur.append(w)
        cur_text = ''.join(x.word for x in cur).strip()
        cur_dur = (cur[-1].end - cur_start) if cur else 0.0
        hit_punct = cur[-1].word.strip()[-1:] in punct_set
        # Only break at sentence-ending punctuation (if min duration satisfied)
        # Otherwise keep accumulating until punctuation, unless hard limits are exceeded
        hard_limits_hit = (
            len(cur) >= hard_max_words or len(cur_text) >= hard_max_chars or cur_dur >= hard_max_dur
        )
        if (hit_punct and cur_dur >= min_dur) or hard_limits_hit:
            chunks.append({
                "start": cur_start,
                "end": cur[-1].end,
                "text": cur_text,
            })
            cur = []
            cur_start = None
    if cur:
        chunks.append({
            "start": cur_start,
            "end": cur[-1].end,
            "text": ''.join(x.word for x in cur).strip(),
        })
    return chunks

def _prepare_text_for_language(text: str, lang: Optional[str]) -> str:
    # For Arabic/Urdu, prepend RLM to improve RTL rendering in some players
    if lang in {"ar", "ur"}:
        return "\u200F" + text
    return text

def transcribe_file(model: WhisperModel, input_path: Path, output_path: Path, cfg: dict, console: Console):
    console.rule(f"[bold green]Transcribing {input_path.name}")
    audio = preprocess_audio(input_path, cfg, console)
    if cfg["vad_filter"]:
        vad_params = cfg["vad_parameters"]
    else:
        vad_params = None
    # Build kwargs dynamically to avoid passing unsupported parameters on older versions
    requested_kwargs = {
        "beam_size": cfg["beam_size"],
        "best_of": cfg["best_of"],
        "patience": cfg["patience"],
        "length_penalty": cfg["length_penalty"],
        "repetition_penalty": cfg["repetition_penalty"],
        "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
        "language": cfg["language"],
        "word_timestamps": cfg["word_timestamps"],
        "vad_filter": cfg["vad_filter"],
        "vad_parameters": vad_params,
        "condition_on_previous_text": cfg["condition_on_previous_text"],
        "temperature": cfg["temperature_list"],
        "initial_prompt": cfg["initial_prompt"],
        "no_speech_threshold": cfg["no_speech_threshold"],
        "log_prob_threshold": cfg["log_prob_threshold"],
        "compression_ratio_threshold": cfg["compression_ratio_threshold"],
        "prompt_reset_on_temperature": cfg["prompt_reset_on_temperature"],
        "hotwords": cfg["hotwords"],
        "language_detection_threshold": cfg["language_detection_threshold"],
        "language_detection_segments": cfg["language_detection_segments"],
    }
    sig = inspect.signature(model.transcribe)
    allowed = set(sig.parameters.keys())
    safe_kwargs = {k: v for k, v in requested_kwargs.items() if k in allowed}
    segments_gen, info = model.transcribe(
        (audio if audio is not None else str(input_path)),
        **safe_kwargs,
    )
    dur_after_vad = getattr(info, "duration_after_vad", None)
    dur_total = getattr(info, "duration", None)
    dur_str = f"{dur_after_vad:.2f}s after VAD" if isinstance(dur_after_vad, (int, float)) else (
        f"{dur_total:.2f}s" if isinstance(dur_total, (int, float)) else "unknown duration"
    )
    console.log(
        f"[bold magenta]Detected language: {getattr(info, 'language', '?')} (p={getattr(info, 'language_probability', 0.0):.2f}), duration: {dur_str}"
    )
    # Display progress while receiving segments
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.completed} segments"), console=console, transient=True) as progress:
        task = progress.add_task(f"Transcribing {input_path.name}", total=None)
        segments = []
        json_segments = []
        for segment in segments_gen:
            segments.append(segment)
            # collect JSON-friendly segment with word confidences
            words_json = []
            if getattr(segment, 'words', None):
                for w in segment.words:
                    words_json.append({
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    })
            json_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": getattr(segment, 'avg_logprob', None),
                "no_speech_prob": getattr(segment, 'no_speech_prob', None),
                "words": words_json,
            })
            progress.update(task, advance=1)
    # Optional two-pass re-decode for low-confidence segments
    two_cfg = cfg.get("two_pass", {})
    if two_cfg.get("enabled", False):
        recoded = 0
        # ensure we have waveform for slicing
        y = audio
        sr = 16000
        if y is None:
            try:
                y, sr = librosa.load(str(input_path), sr=16000, mono=True)
            except Exception as e:
                console.log(f"[yellow]Two-pass skipped: unable to load audio for slicing ({e})")
                y = None
        if y is not None and len(y) > 0:
            sig2 = inspect.signature(model.transcribe)
            allowed2 = set(sig2.parameters.keys())
            for i, seg in enumerate(segments):
                words = getattr(seg, 'words', None) or []
                probs = []
                for w in words:
                    p = getattr(w, 'probability', None)
                    if p is not None:
                        probs.append(p)
                mean_p = (sum(probs) / len(probs)) if probs else None
                avg_lp = getattr(seg, 'avg_logprob', None)
                low_conf = False
                min_mean_word_prob = float(two_cfg.get('min_mean_word_prob', 0.6))
                max_avg_logprob = float(two_cfg.get('max_avg_logprob', -1.25))
                if mean_p is not None and mean_p < min_mean_word_prob:
                    low_conf = True
                elif mean_p is None and isinstance(avg_lp, (int, float)) and avg_lp < max_avg_logprob:
                    low_conf = True
                if not low_conf:
                    continue
                start_ts = float(getattr(seg, 'start', 0.0))
                end_ts = float(getattr(seg, 'end', start_ts))
                s_idx = max(0, int(start_ts * sr))
                e_idx = min(len(y), int(end_ts * sr))
                if e_idx <= s_idx:
                    continue
                region = y[s_idx:e_idx]
                second_req = {
                    "language": cfg.get("language"),
                    "word_timestamps": True,
                    "vad_filter": False,
                    "condition_on_previous_text": False,
                    "beam_size": int(two_cfg.get("beam_size", 12)),
                    "best_of": int(two_cfg.get("best_of", 5)),
                    "temperature": list(two_cfg.get("temperature_list", [0.0, 0.2, 0.4, 0.6])),
                }
                second_kwargs = {k: v for k, v in second_req.items() if k in allowed2}
                try:
                    segs2_gen, _ = model.transcribe(region, **second_kwargs)
                    segs2 = list(segs2_gen)
                except Exception as e:
                    console.log(f"[yellow]Second-pass decode failed for segment {i+1}: {e}")
                    continue
                if not segs2:
                    continue
                # Merge results and adjust times with offset
                new_text_parts = []
                new_words_attr = []
                new_words_json = []
                for s2 in segs2:
                    t_text = s2.text.strip()
                    if t_text:
                        new_text_parts.append(t_text)
                    if getattr(s2, 'words', None):
                        for w in s2.words:
                            st = start_ts + float(getattr(w, 'start', 0.0))
                            en = start_ts + float(getattr(w, 'end', 0.0))
                            new_words_attr.append(SimpleNamespace(start=st, end=en, word=w.word, probability=getattr(w, 'probability', None)))
                            new_words_json.append({"start": st, "end": en, "word": w.word, "probability": getattr(w, 'probability', None)})
                new_text = " ".join(new_text_parts).strip() or getattr(seg, 'text', '')
                last_end_rel = float(getattr(segs2[-1], 'end', (end_ts - start_ts)))
                new_end = max(end_ts, start_ts + last_end_rel)
                new_seg = SimpleNamespace(start=start_ts, end=new_end, text=new_text, words=new_words_attr, avg_logprob=getattr(segs2[-1], 'avg_logprob', None), no_speech_prob=getattr(segs2[-1], 'no_speech_prob', None))
                segments[i] = new_seg
                json_segments[i] = {
                    "start": start_ts,
                    "end": new_end,
                    "text": new_text,
                    "avg_logprob": getattr(new_seg, 'avg_logprob', None),
                    "no_speech_prob": getattr(new_seg, 'no_speech_prob', None),
                    "words": new_words_json,
                }
                recoded += 1
        console.log(f"[bold cyan]Two-pass re-decoded {recoded} low-confidence segments")
    subtitles = []
    idx = 1
    seg_cfg = cfg.get("segmentation", {})
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        words_ts = getattr(segment, 'words', None) or []
        if words_ts:
            for ch in _split_words_into_chunks(words_ts, seg_cfg):
                content = _prepare_text_for_language(ch["text"], cfg.get("language"))
                subtitles.append(srt.Subtitle(
                    index=idx,
                    start=timedelta(seconds=ch["start"]),
                    end=timedelta(seconds=ch["end"]),
                    content=content,
                ))
                idx += 1
        else:
            # Fallback single block
            start_ts = segment.start
            end_ts = segment.end
            content = _prepare_text_for_language(text, cfg.get("language"))
            subtitles.append(srt.Subtitle(
                index=idx,
                start=timedelta(seconds=start_ts),
                end=timedelta(seconds=end_ts),
                content=content,
            ))
            idx += 1
    srt_content = srt.compose(subtitles)
    output_path.write_text(srt_content, encoding="utf-8")
    console.log(f"[bold blue]Written SRT to {output_path}")
    # Emit JSON with word confidences
    if cfg.get("emit_json", True):
        json_path = output_path.with_suffix('.json')
        payload = {
            "language": getattr(info, 'language', None),
            "language_probability": getattr(info, 'language_probability', None),
            "segments": json_segments,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.log(f"[bold blue]Written JSON to {json_path}")
    # Emit WebVTT
    if cfg.get("emit_vtt", True):
        vtt_path = output_path.with_suffix('.vtt')
        lines = ["WEBVTT", ""]
        for sub in subtitles:
            start_s = sub.start.total_seconds()
            end_s = sub.end.total_seconds()
            lines.append(f"{format_timestamp_vtt(start_s)} --> {format_timestamp_vtt(end_s)}")
            lines.append(sub.content)
            lines.append("")
        vtt_path.write_text("\n".join(lines), encoding="utf-8")
        console.log(f"[bold blue]Written VTT to {vtt_path}")
    # Emit run config snapshot for reproducibility
    if cfg.get("emit_run_config", True):
        run_path = output_path.with_suffix('.run.json')
        # sanitize VAD parameters for JSON (Infinity -> null)
        vad_params_snap = dict(cfg.get("vad_parameters", {}))
        ms = vad_params_snap.get("max_speech_duration_s")
        if isinstance(ms, float) and (ms == float("inf") or ms == float("-inf")):
            vad_params_snap["max_speech_duration_s"] = None
        snapshot = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "model_size": cfg.get("model_size"),
            "device": cfg.get("device"),
            "language": cfg.get("language"),
            "vad_filter": cfg.get("vad_filter"),
            "vad_parameters": vad_params_snap,
            "decode_parameters": {k: v for k, v in safe_kwargs.items() if k != "vad_parameters"},
            "segmentation": cfg.get("segmentation"),
            "preprocess": cfg.get("preprocess"),
            "input_file": str(input_path.name),
        }
        run_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        console.log(f"[bold blue]Written run config to {run_path}")

# Configuration dictionary for runtime parameters
CONFIG = {
    "input": "",  # input file or directory path
    "output": ".",           # output directory for SRT/JSON/VTT files
    "model_size": "large-v3",
    "device": "cuda" if _torch_cuda else "cpu",
    "compute_type": "float16",
    # Decoding/thresholds
    
    "beam_size": 20,
    "best_of": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "language": None,           # None enables auto-detection
    "word_timestamps": True,
    "vad_filter": True,
    "vad_parameters": {
        "threshold": 0.5,
        "neg_threshold": None,
        "min_speech_duration_ms": 100,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    },
    "condition_on_previous_text": True,
    "prompt_reset_on_temperature": 0.5,
    "temperature_list": [0.0, 0.2, 0.4],
    "initial_prompt":"",
    "hotwords": None,
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "language_detection_threshold": 0.5,
    "language_detection_segments": 1,
    # Preprocess and segmentation
    "preprocess": {
        "denoise": False,
        "denoise_prop_decrease": 0.8,
        "normalize": True,
        "highpass_hz": 50.0,   # remove rumble
        "lowpass_hz": None,    # e.g., 7800.0 to suppress hiss
        "preemphasis": False,
        "preemphasis_coef": 0.97,
        "dereverb": False,     # basic Wiener-based dereverb
    },
    "segmentation": {"max_words": 20, "max_chars": 42, "max_duration": 6.0, "min_duration": 1.0},
    # Two-pass decode for low-confidence segments
    "two_pass": {
        "enabled": True,
        "min_mean_word_prob": 0.6,
        "max_avg_logprob": -1.25,
        "beam_size": 12,
        "best_of": 5,
        "temperature_list": [0.0, 0.2, 0.4, 0.6],
    },
    # Extra outputs
    "emit_json": False,
    "emit_vtt": False,
    "emit_run_config": False,
}

def main():
    console = Console()
    cfg = CONFIG
    output_dir = Path(cfg["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg['device']
    compute_type = cfg['compute_type'] if device != 'cpu' else 'int8'
    console.log(
        f"[bold magenta]Loading Whisper model {cfg['model_size']} on {device} with {compute_type}"
    )
    model = WhisperModel(
        cfg['model_size'], device=device, compute_type=compute_type
    )
    # Interactive input file selection
    supported_exts = [".mp3", ".wav", ".flac", ".mp4", ".m4a", ".aac", ".ogg"]
    files = sorted([f for f in Path.cwd().iterdir() if f.is_file() and f.suffix.lower() in supported_exts])
    if not files:
        console.print("[bold red]No supported audio files found in current directory[/]")
        return
    console.print("[bold cyan]Available audio files:[/]")
    for idx, f in enumerate(files, start=1):
        console.print(f"[green]{idx}[/] {f.name}")
    choice = Prompt.ask("Select input file number", choices=[str(i) for i in range(1, len(files)+1)])
    selected_file = files[int(choice) - 1]
    # Language selection
    lang_map = {"0": None, "1": "ur", "2": "ar", "3": "en"}
    console.print("\n[bold cyan]Select language:[/]")
    console.print("[green]0[/] Auto-detect  [green]1[/] Urdu  [green]2[/] Arabic  [green]3[/] English")
    lang_choice = Prompt.ask("Enter language number", choices=list(lang_map.keys()))
    cfg["language"] = lang_map[lang_choice]
    lang_display = cfg["language"] if cfg["language"] else "auto"
    console.log(f"[bold magenta]Language set to {lang_display}")
    console.log(
        f"[bold cyan]Decoding params -> beam: {cfg['beam_size']}, best_of: {cfg['best_of']}, temps: {cfg['temperature_list']}"
    )
    # Transcribe selected file
    out_srt = output_dir / f"{selected_file.stem}.srt"
    transcribe_file(model, selected_file, out_srt, cfg, console)

if __name__ == "__main__":
    main() 