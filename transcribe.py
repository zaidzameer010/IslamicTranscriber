from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from faster_whisper import WhisperModel, BatchedInferencePipeline


def get_device_and_compute():
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


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


def find_audio_files() -> List[Path]:
    cwd = Path.cwd()
    files = []
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


def select_language(console: Console) -> Optional[str]:
    console.print("[bold magenta]Language[/bold magenta]: 1) Auto-detect  2) Urdu (ur)")
    choice = Prompt.ask("Choose language", choices=["1", "2"], default="1")
    return None if choice == "1" else "ur"


def build_initial_prompt() -> str:
    return (
        "یہ ایک مذہبی بیان ہے جس میں عربی آیاتِ قرآنی، احادیث اور اردو تشریحات شامل ہیں۔ "
        "برائے مہربانی مقدس ناموں اور اصطلاحات کو درست اور مکمل لکھیں جیسے: اللہ، محمد ﷺ، قرآن، حدیث، سبحان اللہ، "
        "الحمد للہ، ان شاء اللہ، سورۃ الفاتحہ، صحیح بخاری، امام ابو حنیفہ، تفسیر ابن کثیر۔"
    )


def transcribe_to_srt(
    console: Console,
    path: Path,
    initial_prompt=None,
    vad_parameters=None,
    language: Optional[str] = None,
) -> Path:
    device, compute_type = get_device_and_compute()
    with console.status(f"Loading model large-v3 on {device} ({compute_type})..."):
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    use_batched = device == "cuda"
    transcriber = model.transcribe
    if use_batched:
        batched = BatchedInferencePipeline(model=model)
        transcriber = batched.transcribe

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )

    out_path = path.with_suffix(".srt")
    with progress:
        effective_vad = vad_parameters or {
            "threshold": 0.6,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 2000,
        }
        kwargs = {
            "task": "transcribe",
            "beam_size": 15,
            "vad_filter": True,
            "vad_parameters": effective_vad,
            "word_timestamps": False,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "initial_prompt": initial_prompt if initial_prompt else None,
            "condition_on_previous_text": True,
        }
        if language:
            kwargs["language"] = language
        if use_batched:
            kwargs["batch_size"] = 4
        segments, info = transcriber(str(path), **kwargs)

        total = float(info.duration) if getattr(info, "duration", None) else None
        if total and total > 0:
            task_id = progress.add_task(f"Transcribing {path.name}", total=total)
        else:
            task_id = progress.add_task(f"Transcribing {path.name}", total=None, start=False)
            progress.start_task(task_id)

 
        idx = 1
        last = 0.0
        with out_path.open("w", encoding="utf-8") as srt:
            for segment in segments:
                start_ts = format_srt_time(float(segment.start))
                end_ts = format_srt_time(float(segment.end))
                text = (segment.text or "").strip()
                srt.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")
                idx += 1
                if total:
                    delta = float(segment.end) - last
                    if delta < 0:
                        delta = 0.0
                    progress.update(task_id, advance=delta)
                    last = float(segment.end)
        if total:
            progress.update(task_id, completed=total)
    return out_path
 
 
def main():
    console = Console()
    console.print("[bold green]ASR Transcriber (Faster-Whisper large-v3)[/bold green]")
    files = find_audio_files()
    if not files:
        console.print("[bold red]No .mp3 or .mp4 files found in the current directory.[/bold red]")
        return
    target = select_file(console, files)
    language = select_language(console)
    init_prompt = build_initial_prompt()
    console.print("[bold cyan]Using initial prompt to bias transcription for Urdu/Arabic religious context.[/bold cyan]")
    if language:
        console.print(f"[bold cyan]Forcing language:[/bold cyan] {language}")
    try:
        out_file = transcribe_to_srt(console, target, initial_prompt=init_prompt, language=language)
        console.print(f"[bold green]Saved:[/bold green] {out_file.name}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
 
 
if __name__ == "__main__":
    main()
