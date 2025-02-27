#!/usr/bin/env python3
"""
Optimized Audio/Video Transcription Tool
Using Whisper Large-v3 with faster-whisper
Streamlined for mixed Urdu+Arabic+English
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Audio processing
try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub package is required. Install with 'pip install pydub'.")
    print("Note: ffmpeg must also be installed on your system.")
    sys.exit(1)

# Transcription
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper package is required. Install with 'pip install faster-whisper'.")
    sys.exit(1)

# Rich UI components
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich import print as rprint
except ImportError:
    print("Error: rich package is required. Install with 'pip install rich'.")
    sys.exit(1)

# For SRT generation
try:
    import srt
    from datetime import timedelta
except ImportError:
    print("Error: srt package is required. Install with 'pip install srt'.")
    sys.exit(1)

# For file selection
import glob

# Quality/Speed tradeoff options
QUALITY_PRESETS = {
    "accurate": {
        "model_size": "large-v3",
        "beam_size": 10,
        "vad_filter": True, 
        "compute_type": "float16" if True else "int8",  # Will be properly set based on device
    }
}

# Language-specific prompts to guide transcription
LANGUAGE_PROMPTS = {
    "ar": """بسم الله الرحمن الرحيم. هذا نص عربي فصيح. يرجى كتابة النص بدقة مع علامات الترقيم والتشكيل:

النص التالي باللغة العربية الفصحى. يجب مراعاة:
- كتابة الهمزات بشكل صحيح
- وضع علامات الترقيم المناسبة
- مراعاة الفواصل بين الجمل
- كتابة التاء المربوطة والمفتوحة بشكل صحيح
""",
    "ur": "بسم اللہ الرحمٰن الرحیم۔ یہ اردو میں ٹرانسکرپشن ہے:",
    "en": "The following is English speech transcription:",
}

DEFAULT_MULTILINGUAL_PROMPT = """The following audio may contain a mix of Arabic, Urdu, and English.
Please transcribe each language accurately with proper punctuation while maintaining the original language."""

# Common error corrections for each language
CORRECTIONS = {
    "ar": {
        # Add common Arabic transcription errors here
        "ه ذا": "هذا",
        "ف ي": "في",
        "ع ن": "عن",
        "م ن": "من",
        "إ ن": "إن",
        "أ ن": "أن",
        "ا ل": "ال",
        # Common mistakes with hamza
        "اسلام": "إسلام",
        "انسان": "إنسان",
        "امام": "إمام",
        # Fixing common spacing issues
        " ،": "،",
        " .": ".",
        " ؟": "؟",
        "  ": " ",
    },
    "ur": {
        # Add common Urdu transcription errors here
        "ک يا": "کیا",
        "ه ے": "هے",
    },
    "en": {
        # Add common English transcription errors here
        "i s": "is",
        "i t": "it",
    }
}


def get_device_info():
    """Determine the best available device and compute type"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", "float16"
        else:
            return "cpu", "int8"
    except ImportError:
        return "cpu", "int8"


def extract_audio(file_path: str, temp_dir: str, optimize: bool = True) -> str:
    """Extract audio efficiently for transcription"""
    console = Console()
    
    input_path = Path(file_path)
    file_name = input_path.stem
    temp_audio_path = os.path.join(temp_dir, f"{file_name}.wav")
    
    with console.status(f"[bold green]Extracting audio from {input_path.name}..."):
        try:
            # Load audio
            audio = AudioSegment.from_file(file_path)
            
            # Efficient conversion for speech recognition
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz is optimal for Whisper
            
            # For large files, downsample to reduce processing time
            if optimize and len(audio) > 600000:  # If longer than 10 minutes
                audio = audio.set_sample_width(2)  # 16-bit is sufficient
            
            # Export optimized audio
            audio.export(temp_audio_path, format="wav")
            return temp_audio_path
            
        except Exception as e:
            console.print(f"[bold red]Error extracting audio: {str(e)}")
            raise


def transcribe_audio(
    audio_path: str, 
    quality_preset: str = "accurate",  
    language: str = None,  
    device: str = None,
    compute_type: str = None
) -> List[dict]:
    """Single-pass transcription using Whisper large-v3 model with enhanced Arabic support and accurate timestamps"""
    console = Console()
    
    # Get preset configuration
    preset = QUALITY_PRESETS[quality_preset]
    model_size = preset["model_size"]
    beam_size = preset["beam_size"]
    vad_filter = preset["vad_filter"]
    
    # Determine device if not specified
    if not device or not compute_type:
        device, compute_type = get_device_info()
        
    console.print(Panel(
        Text.from_markup(f"[bold]Transcribing with Whisper [cyan]{model_size}[/cyan] model...")
    ))
    console.print(f"Using: [cyan]{device}[/cyan] | Quality: [cyan]{quality_preset}[/cyan]")
    
    # Initialize model with enhanced settings
    model = WhisperModel(
        model_size, 
        device=device, 
        compute_type=compute_type,
        download_root=os.path.expanduser("~/.cache/whisper")
    )
    
    # Enhanced parameters for accurate transcription and timestamps
    transcription_params = {
        "beam_size": max(beam_size, 10),  # Larger beam size for better search
        "language": language if language else "ar",
        "vad_filter": True,  # Enable VAD for better segment detection
        "vad_parameters": dict(
            min_silence_duration_ms=300,  # Reduced for finer segmentation
            speech_pad_ms=100,  # Reduced padding for more precise boundaries
            threshold=0.35  # Balanced threshold for speech detection
        ),
        "word_timestamps": True,
        "initial_prompt": LANGUAGE_PROMPTS.get(language, DEFAULT_MULTILINGUAL_PROMPT),
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # More temperature values for better sampling
        "best_of": 5,  # Increased for better candidate selection
        "condition_on_previous_text": True,
        "no_speech_threshold": 0.3,
        "compression_ratio_threshold": 2.4
    }
    
    # If using large-v3 with sufficient compute, add more options for accuracy
    if model_size == "large-v3" and device in ["cuda", "mps"]:
        transcription_params.update({
            "best_of": 5
        })
    
    # Transcribe audio
    segments, info = model.transcribe(audio_path, **transcription_params)
    
    # Process and format segments
    segments_list = []
    for segment in segments:
        segments_list.append({
            'id': len(segments_list),
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip(),
            'words': segment.words,
            'language': info.language  # Use detected language from Whisper
        })
    
    # Report language detection info
    detected_language = info.language
    language_confidence = round(info.language_probability * 100, 2)
    console.print(f"Detected language: [cyan]{detected_language}[/cyan] (Confidence: {language_confidence}%)")
    
    # Apply post-processing fixes
    processed_segments = post_process_segments(segments_list, detected_language)
    
    return processed_segments


def post_process_segments(segments, detected_language):
    """Apply post-processing to improve transcription quality and timestamp accuracy"""
    processed = []
    
    # Language-specific corrections dictionary
    corrections = {}
    
    # Choose correction set based on detected language
    if detected_language in ["ar", "ur", "en"]:
        corrections = CORRECTIONS[detected_language]
    else:
        # For mixed content, include all corrections
        for lang_corrections in CORRECTIONS.values():
            corrections.update(lang_corrections)
    
    # Process each segment
    for segment in segments:
        text = segment["text"]
        
        # Apply text corrections
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        
        # Update with corrected text
        segment["text"] = text
        processed.append(segment)
    
    # Enhanced segment merging with timestamp adjustments
    if len(processed) > 1:
        merged = [processed[0]]
        
        for current in processed[1:]:
            previous = merged[-1]
            
            # Calculate gap between segments
            gap = current['start'] - previous['end']
            
            # If segments are very close (less than 0.3s gap) and previous is short
            if (gap < 0.3 and len(previous['text'].split()) < 4):
                # Adjust timestamps for smoother transition
                midpoint = (current['start'] + previous['end']) / 2
                previous['end'] = midpoint
                current['start'] = midpoint
                
                # Merge segments
                previous['end'] = current['end']
                previous['text'] += " " + current['text']
                
                # If words info exists, merge it
                if 'words' in previous and 'words' in current:
                    previous['words'].extend(current['words'])
            else:
                # If gap is small but segments shouldn't be merged,
                # adjust timestamps to prevent overlap
                if gap < 0:
                    midpoint = (current['start'] + previous['end']) / 2
                    previous['end'] = midpoint
                    current['start'] = midpoint
                merged.append(current)
        
        return merged
    
    return processed


def generate_srt(segments: List[dict], output_path: str) -> None:
    """Generate SRT file from transcription segments"""
    console = Console()
    
    srt_entries = []
    for i, segment in enumerate(segments, start=1):
        start_time = timedelta(seconds=float(segment['start']))
        end_time = timedelta(seconds=float(segment['end']))
        
        entry = srt.Subtitle(
            index=i,
            start=start_time,
            end=end_time,
            content=segment['text']
        )
        srt_entries.append(entry)
    
    # Write SRT file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(srt_entries))
    
    console.print(f"[bold green]SRT file generated: [cyan]{output_path}")


def select_file(directory: str = ".") -> str:
    """Interactive file selection with Rich"""
    console = Console()
    
    # Find all media files in directory
    media_extensions = [
        # Audio formats
        '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
        # Video formats
        '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'
    ]
    
    media_files = []
    
    for ext in media_extensions:
        media_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        media_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    
    if not media_files:
        console.print("[bold red]No media files found in the current directory.")
        return None
    
    # Sort files alphabetically
    media_files.sort()
    
    # Create a table for better display
    table = Table(title="Available Media Files")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Filename", style="green")
    table.add_column("Size", style="blue", justify="right")
    
    # Maximum files to show at once
    max_files = 15
    current_page = 0
    total_pages = (len(media_files) + max_files - 1) // max_files
    
    while True:
        # Display table of files for current page
        table.rows = []  # Clear existing rows
        
        start_idx = current_page * max_files
        end_idx = min(start_idx + max_files, len(media_files))
        
        for i in range(start_idx, end_idx):
            file_path = media_files[i]
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            # Determine file type
            ext = Path(file_path).suffix.lower()
            file_type = "🎵 Audio" if ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'] else "🎬 Video"
            
            table.add_row(
                str(i + 1),
                file_type,
                file_name,
                f"{file_size:.2f} MB"
            )
        
        # Display the table
        console.print("\n")
        console.print(table)
        
        # Pagination info
        console.print(f"Page {current_page + 1}/{total_pages}")
        
        # Navigation options
        if total_pages > 1:
            console.print("[bold]Navigation:[/bold] (n)ext page, (p)revious page, or enter file number")
        
        # Get user input
        selection = Prompt.ask("Select a file", default="1")
        
        # Handle navigation commands
        if selection.lower() == 'n' and current_page < total_pages - 1:
            current_page += 1
            continue
        elif selection.lower() == 'p' and current_page > 0:
            current_page -= 1
            continue
        
        # Handle file selection
        try:
            index = int(selection) - 1
            if 0 <= index < len(media_files):
                return media_files[index]
            else:
                console.print("[bold red]Invalid selection. Please try again.")
        except ValueError:
            console.print("[bold red]Please enter a valid number or navigation command.")


def main():
    console = Console()
    
    # App header
    console.print(Panel.fit(
        Text.from_markup("[bold cyan]Whisper Large-v3 Transcription Tool[/bold cyan]\n"
                         "[italic]Optimized for mixed Urdu+Arabic+English audio[/italic]"),
        border_style="cyan"
    ))
    
    try:
        # File selection
        file_path = select_file()
        if not file_path:
            console.print("[bold red]No file selected. Exiting.")
            return
        
        # Output path
        input_path = Path(file_path)
        output_dir = input_path.parent
        output_file = output_dir / f"{input_path.stem}.srt"
        
        # Quality preset selection
        quality_options = {
            "1": ("accurate", "Accurate - Most accurate but slower (Large-v3 model)")
        }
        
        console.print("Select quality preset:")
        for key, (preset, desc) in quality_options.items():
            console.print(f"[bold cyan]{key}.[/bold cyan] {desc}")
        
        quality_choice = Prompt.ask("Enter option number", default="1")
        quality_preset = quality_options.get(quality_choice, ("accurate", ""))[0]
        
        # Language selection
        language_options = {
            "1": ("ar", "Arabic"),
            "2": ("ur", "Urdu"),
            "3": ("en", "English"),
            "4": (None, "Auto-detect (let Whisper decide)")
        }
        
        console.print("Select language:")
        for key, (code, name) in language_options.items():
            console.print(f"[bold cyan]{key}.[/bold cyan] {name}")
        
        lang_choice = Prompt.ask("Enter option number", default="4")
        language = language_options.get(lang_choice, (None, ""))[0]
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Single progress bar for the entire process
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Extract audio
                task = progress.add_task("[bold yellow]Extracting audio...", total=None)
                audio_path = extract_audio(file_path, temp_dir, optimize=True)
                progress.update(task, completed=True)
                
                # Transcribe in a single pass
                task = progress.add_task("[bold green]Transcribing audio...", total=None)
                segments = transcribe_audio(
                    audio_path,
                    quality_preset=quality_preset,
                    language=language
                )
                progress.update(task, completed=True)
                
                # Generate SRT
                task = progress.add_task("[bold blue]Generating SRT file...", total=None)
                generate_srt(segments, str(output_file))
                progress.update(task, completed=True)
            
            # Final success message
            rprint(Panel(
                Text.from_markup(
                    f"[bold green]Transcription complete![/bold green]\n"
                    f"Output saved to: [bold cyan]{output_file}[/bold cyan]"
                ),
                title="Success",
                border_style="green"
            ))
    
    except KeyboardInterrupt:
        console.print("[bold yellow]Process interrupted by user.")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")


if __name__ == "__main__":
    main()