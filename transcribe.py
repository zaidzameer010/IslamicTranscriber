from pathlib import Path
from datetime import timedelta

import torch
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt
import srt

def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def transcribe_file(model: WhisperModel, input_path: Path, output_path: Path, cfg: dict, console: Console):
    console.rule(f"[bold green]Transcribing {input_path.name}")
    if cfg["vad_filter"]:
        vad_params = cfg["vad_parameters"]
    else:
        vad_params = None
    segments_gen, info = model.transcribe(
        str(input_path),
        beam_size=cfg["beam_size"],
        language=cfg["language"],
        word_timestamps=cfg["word_timestamps"],
        vad_filter=cfg["vad_filter"],
        vad_parameters=vad_params,
        condition_on_previous_text=cfg["condition_on_previous_text"],
        temperature=cfg["temperature_list"],
        initial_prompt=cfg["initial_prompt"],
    )
    # Display progress while receiving segments
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.completed} segments"), console=console, transient=True) as progress:
        task = progress.add_task(f"Transcribing {input_path.name}", total=None)
        segments = []
        for segment in segments_gen:
            segments.append(segment)
            progress.update(task, advance=1)
    subtitles = []
    for i, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        start = segment.start
        end = segment.end
        subtitles.append(srt.Subtitle(
            index=i,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text,
        ))
    srt_content = srt.compose(subtitles)
    output_path.write_text(srt_content, encoding="utf-8")
    console.log(f"[bold blue]Written SRT to {output_path}")

# Configuration dictionary for runtime parameters
CONFIG = {
    "input": "",  # input file or directory path
    "output": ".",           # output directory for SRT files
    "model_size": "large-v3",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "compute_type": "float16",
    "beam_size": 15,
    "language": None,           # will be set via prompt: ur, ar, en
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 500},  # lower threshold for sensitivity
    "condition_on_previous_text": True,
    "temperature_list": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],             # greedy decoding for highest accuracy
    "initial_prompt":"This audio contains both Urdu and Arabic. Please transcribe both languages accurately, preserving Arabic Quranic verses, Hadiths, and prayers in Arabic script, and Urdu explanations in Urdu script."
}

def main():
    console = Console()
    cfg = CONFIG
    output_dir = Path(cfg["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    console.log(
        f"[bold magenta]Loading Whisper model {cfg['model_size']} on {cfg['device']} with {cfg['compute_type']}"
    )
    model = WhisperModel(
        cfg['model_size'], device=cfg['device'], compute_type=cfg['compute_type']
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
    lang_map = {"1": "ur", "2": "ar", "3": "en"}
    console.print("\n[bold cyan]Select language:[/]")
    console.print("[green]1[/] Urdu  [green]2[/] Arabic  [green]3[/] English")
    lang_choice = Prompt.ask("Enter language number", choices=list(lang_map.keys()))
    cfg["language"] = lang_map[lang_choice]
    console.log(f"[bold magenta]Language set to {cfg['language']}" )
    # Transcribe selected file
    out_srt = output_dir / f"{selected_file.stem}.srt"
    transcribe_file(model, selected_file, out_srt, cfg, console)

if __name__ == "__main__":
    main() 