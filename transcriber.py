import os
import torch
import re
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from datetime import timedelta
from pathlib import Path
from moviepy.editor import VideoFileClip
import tempfile
import warnings
from faster_whisper import WhisperModel

# Disable warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

console = Console()

SUPPORTED_LANGUAGES = {
    'ar': 'Arabic',
    'ur': 'Urdu',
    'en': 'English',
    'hi': 'Hindi',
    'fa': 'Persian',
    'tr': 'Turkish',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ms': 'Malay',
    'bn': 'Bengali',
    'id': 'Indonesian',
    'ta': 'Tamil'
}

SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv', '.avi', '.mov']

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds // 60) % 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def save_srt(segments, output_file: str):
    """Save transcription segments in SRT format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def extract_audio_from_video(video_path, progress):
    """Extract audio from video file and return path to temporary audio file."""
    task = progress.add_task("[bold bright_magenta]Extracting audio from video...", total=100)
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()
    try:
        video = VideoFileClip(str(video_path))
        progress.update(task, completed=50)
        video.audio.write_audiofile(temp_audio_path, logger=None)
        video.close()
        progress.update(task, completed=100)
        return temp_audio_path
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise e

def get_audio_path():
    """Get audio path from user input."""
    while True:
        console.print("\n[bold cyan]Enter the path to your audio/video file:[/bold cyan] ", end="")
        path = input().strip()
        path = Path(path)
        if path.exists():
            if path.suffix.lower() in SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS:
                return path
            else:
                console.print(f"[red]Error: Unsupported file format. Supported formats:[/red]")
                console.print(f"[yellow]Audio:[/yellow] {', '.join(SUPPORTED_AUDIO_FORMATS)}")
                console.print(f"[yellow]Video:[/yellow] {', '.join(SUPPORTED_VIDEO_FORMATS)}")
        else:
            console.print("[red]Error: File not found. Please enter a valid path.[/red]")

def get_language_selection():
    """Get language selection from user input."""
    console.print("\n[bold cyan]Available Languages:[/bold cyan]")
    for i, (code, name) in enumerate(SUPPORTED_LANGUAGES.items(), 1):
        console.print(f"{i}. {code}: {name}")
    console.print("\nEnter language number or press Enter for auto-detection: ", end="")
    while True:
        choice = input().strip()
        if choice == "":
            return None
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(SUPPORTED_LANGUAGES):
                return list(SUPPORTED_LANGUAGES.keys())[choice_num - 1]
            else:
                console.print("[red]Invalid number. Please try again: [/red]", end="")
        except ValueError:
            console.print("[red]Please enter a valid number: [/red]", end="")

def get_model_selection():
    """Get model selection from user input."""
    models = ['large-v3', 'medium', 'small']
    console.print("\n[bold cyan]Available Models:[/bold cyan]")
    for i, model in enumerate(models, 1):
        console.print(f"{i}. {model}")
    console.print("\nEnter model number (or press Enter for large-v3): ", end="")
    while True:
        choice = input().strip()
        if choice == "":
            return "large-v3"
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(models):
                return models[choice_num - 1]
            else:
                console.print("[red]Invalid number. Please try again: [/red]", end="")
        except ValueError:
            console.print("[red]Please enter a valid number: [/red]", end="")

def transcribe():
    """Transcribe audio file using Whisper with improved instructions and post-processing."""
    try:
        file_path = get_audio_path()
        output_path = file_path.parent / f"{file_path.stem}.srt"
        model_name = get_model_selection()
        language = get_language_selection()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold white]{task.description}"),
            BarColumn(complete_style="bright_yellow", finished_style="bright_green"),
            TaskProgressColumn(),
            console=console,
            expand=True
        ) as progress:
            # If a video is provided, extract its audio.
            audio_path = file_path
            temp_audio_path = None
            if file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                temp_audio_path = extract_audio_from_video(file_path, progress)
                audio_path = Path(temp_audio_path)
            
            try:
                model_task = progress.add_task("[bold bright_blue]Loading Whisper model...", total=100)
                if torch.cuda.is_available():
                    model = WhisperModel(
                        model_name,
                        device="cuda",
                        compute_type="float16",
                        cpu_threads=8,
                        num_workers=2,
                        download_root=None,
                        local_files_only=False
                    )
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    gpu_name = torch.cuda.get_device_name()
                    console.print(f"[bold bright_green]Using GPU:[/bold bright_green] {gpu_name}")
                else:
                    model = WhisperModel(
                        model_name,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=8,
                        num_workers=2
                    )
                progress.update(model_task, completed=100)

                transcribe_task = progress.add_task("[bold bright_yellow]Transcribing audio...", total=100, start=True)

                # --- Enhanced initial prompt ---
                enhanced_prompt = (
                    "Transcribe this Islamic lecture accurately. For any Quranic verses, Hadith, or Islamic terms, "
                    "use the original Arabic script exactly as spoken. For all other content, transcribe in Urdu script. "
                    "Do not mix scripts within the same sentence unless the speaker explicitly switches languages."
                )
                # ---------------------------------

                segments, info = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=15,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=400
                    ),
                    condition_on_previous_text=True,
                    initial_prompt=enhanced_prompt,
                    temperature=0.0,
                    without_timestamps=False,
                    max_initial_timestamp=1.0,
                    suppress_tokens=[],
                    best_of=3,
                    task="transcribe"
                )

                # Convert generator to list while updating progress.
                segments_list = []
                total_duration = 0
                audio_duration = info.duration

                for segment in segments:
                    segments_list.append(segment)
                    total_duration = segment.end
                    progress_percent = min(int((total_duration / audio_duration) * 100), 99)
                    progress.update(transcribe_task, completed=progress_percent)

                segments = segments_list
                progress.update(transcribe_task, completed=100)

                # Save the transcription to SRT file
                save_task = progress.add_task("[bold bright_green]Saving SRT file...", total=100)
                save_srt(segments, str(output_path))
                progress.update(save_task, completed=100)
            
            finally:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except Exception:
                        pass

        console.print(f"\n[bold bright_green]✓[/bold bright_green] Transcription saved to: {output_path}")
        console.print(f"[bold bright_blue]Detected language:[/bold bright_blue] {info.language}")
        console.print(f"[bold bright_yellow]Language probability:[/bold bright_yellow] {info.language_probability:.2%}")
        if file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
            console.print("[bold bright_magenta]Note:[/bold bright_magenta] Video file was processed by extracting its audio")
        
        return output_path

    except Exception as e:
        console.print(f"\n[bold bright_red]Error:[/bold bright_red] {str(e)}")
        return None

if __name__ == '__main__':
    transcribe()
