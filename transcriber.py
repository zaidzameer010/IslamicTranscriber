#!/usr/bin/env python3
"""
Advanced Audio/Video Transcription Tool
Using Whisper Large-v3 with faster-whisper
Optimized for mixed Urdu+Arabic+English transcription
"""

import os
import sys
import tempfile
import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Audio processing
try:
    from pydub import AudioSegment
    import librosa
    import soundfile as sf
    import noisereduce as nr
except ImportError:
    print("Error: Audio processing packages are required.")
    print("Install with: 'pip install pydub librosa soundfile noisereduce'")
    print("Note: ffmpeg must also be installed on your system.")
    sys.exit(1)

# Transcription
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper package is required. Install with 'pip install faster-whisper'.")
    sys.exit(1)

# Language detection
try:
    import langdetect
    from langdetect import DetectorFactory
    # Make language detection deterministic
    DetectorFactory.seed = 42
except ImportError:
    print("Error: langdetect package is required. Install with 'pip install langdetect'.")
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

# For file selection
import glob

# For SRT generation
try:
    import srt
    from datetime import timedelta
except ImportError:
    print("Error: srt package is required. Install with 'pip install srt'.")
    sys.exit(1)


class LanguageDetector:
    """Advanced language detection for mixed language audio"""
    
    # Script patterns for identifying languages
    SCRIPT_PATTERNS = {
        "arabic": re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+'),
        "latin": re.compile(r'[a-zA-Z\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+'),
        "devanagari": re.compile(r'[\u0900-\u097F]+')
    }
    
    # High-frequency words to distinguish languages with similar scripts
    FREQUENT_WORDS = {
        "ar": ["في", "من", "على", "إلى", "عن", "مع", "هذا", "أن", "لا", "ما", "الله", "هو"],
        "ur": ["میں", "کے", "کا", "اور", "کو", "سے", "نے", "ہے", "کہ", "پر", "کی", "ہیں"],
        "en": ["the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "with", "as"]
    }
    
    def __init__(self, primary_languages=None, confidence_threshold=0.6):
        self.primary_languages = primary_languages or ["ur", "ar", "en"]
        self.confidence_threshold = confidence_threshold
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Advanced language detection using multiple techniques"""
        if not text or len(text.strip()) < 5:
            return "unknown", 0.0
        
        # Script analysis
        script_counts = {script: len(pattern.findall(text)) 
                        for script, pattern in self.SCRIPT_PATTERNS.items()}
        total_chars = sum(script_counts.values())
        
        if total_chars == 0:
            return "unknown", 0.0
            
        script_ratios = {script: count/total_chars for script, count in script_counts.items()}
        
        # Primary script detection
        primary_script = max(script_ratios.items(), key=lambda x: x[1])[0]
        
        # Script-to-language mapping with validation
        if primary_script == "arabic" and script_ratios["arabic"] > 0.5:
            # Differentiate Urdu from Arabic using frequency analysis
            ur_matches = sum(1 for word in self.FREQUENT_WORDS["ur"] if word in text)
            ar_matches = sum(1 for word in self.FREQUENT_WORDS["ar"] if word in text)
            
            # Calculate weighted scores
            ur_score = ur_matches / max(len(self.FREQUENT_WORDS["ur"]), 1)
            ar_score = ar_matches / max(len(self.FREQUENT_WORDS["ar"]), 1)
            
            # Add stronger differentiation for ambiguous cases
            if "ہے" in text or "کا" in text or "کی" in text:
                ur_score += 0.3
            if "في" in text or "من" in text or "على" in text:
                ar_score += 0.3
                
            if ur_score > ar_score:
                return "ur", max(0.6, ur_score)
            else:
                return "ar", max(0.6, ar_score)
                
        elif primary_script == "latin" and script_ratios["latin"] > 0.5:
            # Validate it's English using frequency analysis
            en_matches = sum(1 for word in self.FREQUENT_WORDS["en"] if word.lower() in text.lower())
            en_score = en_matches / max(len(self.FREQUENT_WORDS["en"]), 1)
            return "en", max(0.7, en_score)
        
        # Try langdetect as a fallback
        try:
            detected = langdetect.detect(text)
            confidence = 0.6  # langdetect doesn't provide confidence
            return detected, confidence
        except:
            # Unable to detect language
            return "unknown", 0.0


class AudioProcessor:
    """Enhanced audio processing for better transcription quality"""
    
    def __init__(self, normalize=True, noise_reduction=True):
        self.normalize = normalize
        self.noise_reduction = noise_reduction
    
    def extract_audio(self, file_path: str, temp_dir: str) -> str:
        """Extract and enhance audio for better transcription"""
        console = Console()
        
        input_path = Path(file_path)
        file_name = input_path.stem
        temp_audio_path = os.path.join(temp_dir, f"{file_name}_raw.wav")
        enhanced_path = os.path.join(temp_dir, f"{file_name}_enhanced.wav")
        
        with console.status(f"[bold green]Processing audio from {input_path.name}..."):
            try:
                # Extract audio
                ext = input_path.suffix.lower()
                audio = AudioSegment.from_file(file_path)
                
                # Convert to mono and set appropriate sample rate for speech
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio.export(temp_audio_path, format="wav")
                
                # Audio enhancement
                if self.normalize or self.noise_reduction:
                    # Load with librosa for advanced processing
                    y, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
                    
                    # Normalize audio levels
                    if self.normalize:
                        y = librosa.util.normalize(y)
                    
                    # Apply noise reduction
                    if self.noise_reduction:
                        # Estimate noise from first 2 seconds
                        noise_sample = y[:min(len(y), int(2 * sr))]
                        y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.8)
                    
                    # Apply a gentle high-pass filter to reduce low-frequency noise
                    y = librosa.effects.preemphasis(y, coef=0.97)
                    
                    # Trim silence
                    y, _ = librosa.effects.trim(y, top_db=30)
                    
                    # Save enhanced audio
                    sf.write(enhanced_path, y, sr)
                    return enhanced_path
                
                return temp_audio_path
                
            except Exception as e:
                console.print(f"[bold red]Error processing audio: {str(e)}")
                raise


class TranscriptionOptimizer:
    """Optimizes transcription parameters for mixed language audio"""
    
    def __init__(self, model_size="large-v3"):
        self.model_size = model_size
        self.device = self._get_optimal_device()
        self.language_detector = LanguageDetector()
        self.console = Console()
        
    def _get_optimal_device(self):
        """Determine the best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return ("cuda", "float16")
            # Check for Apple Silicon
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return ("mps", "float16")
            else:
                return ("cpu", "int8")
        except:
            return ("cpu", "int8")
            
    def get_language_prompt(self, language: str) -> str:
        """Get language-specific initial prompt"""
        prompts = {
            "ar": "بسم الله الرحمن الرحيم. فيما يلي نص باللغة العربية:",
            "ur": "بسم اللہ الرحمٰن الرحیم۔ درج ذیل اردو میں متن ہے:",
            "en": "The following is an English language transcription:",
        }
        return prompts.get(language, "")
    
    def analyze_sample(self, model, audio_path):
        """Analyze a sample of audio to detect languages and optimize parameters"""
        # Sample transcription to detect languages
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=None,
            vad_filter=True,
            max_initial_timestamp=30.0,  # Only process the first 30 seconds
        )
        
        # Collect texts
        texts = [segment.text for segment in segments]
        full_text = " ".join(texts)
        
        # Detect languages in sample
        languages = {}
        
        # Overall language
        overall_lang, _ = self.language_detector.detect_language(full_text)
        languages[overall_lang] = 1.0
        
        # Check individual segments for other languages
        for text in texts:
            if len(text.strip()) < 10:
                continue
            lang, conf = self.language_detector.detect_language(text)
            if lang != "unknown" and conf > 0.6:
                languages[lang] = languages.get(lang, 0) + 1
        
        # Convert to percentages
        total = sum(languages.values())
        if total > 0:
            languages = {k: v/total for k, v in languages.items()}
        
        return languages, info.language
    
    def get_optimal_parameters(self, detected_languages, main_language=None):
        """Get optimal parameters based on detected languages"""
        # Default parameters
        params = {
            'beam_size': 10,
            'language': main_language,
            'vad_filter': True,
            'vad_parameters': {
                'min_silence_duration_ms': 500,
                'speech_pad_ms': 500
            },
            'word_timestamps': True,
            'condition_on_previous_text': True,
            'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'patience': 1.0,
            'best_of': 5,
            'suppress_tokens': [-1],
            'initial_prompt': None,
        }
        
        # Check if mixed language (multiple languages with significant presence)
        is_mixed = len([lang for lang, pct in detected_languages.items() 
                      if pct > 0.2 and lang != "unknown"]) > 1
        
        # For mixed language content, optimize parameters
        if is_mixed:
            params['beam_size'] = 15  # Higher beam size for mixed content
            params['best_of'] = 5  # More candidates
            
            # Generate mixed language prompt
            prompts = []
            for lang, pct in sorted(detected_languages.items(), key=lambda x: x[1], reverse=True):
                if pct > 0.2 and lang in ["ar", "ur", "en"]:
                    prompts.append(self.get_language_prompt(lang))
            
            if prompts:
                params['initial_prompt'] = " ".join(prompts)
        else:
            # Single language optimization
            primary_lang = max(detected_languages.items(), key=lambda x: x[1])[0]
            if primary_lang in ["ar", "ur", "en"]:
                params['language'] = primary_lang
                params['initial_prompt'] = self.get_language_prompt(primary_lang)
                
                # Language-specific optimizations
                if primary_lang in ["ar", "ur"]:
                    params['beam_size'] = 12  # Higher beam size for Arabic script
        
        return params
    
    def transcribe_with_optimal_settings(self, audio_path, user_language=None):
        """Transcribe audio with optimal settings for language mix"""
        self.console.print(Panel.fit(
            Text.from_markup(f"[bold]Loading Whisper [cyan]{self.model_size}[/cyan] model...")
        ))
        
        # Initialize model
        device, compute_type = self.device
        self.console.print(f"Using device: [cyan]{device}[/cyan] with compute type: [cyan]{compute_type}[/cyan]")
        
        model = WhisperModel(
            self.model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=os.path.expanduser("~/.cache/whisper")
        )
        
        # Step 1: Analyze language distribution
        self.console.print(f"[bold yellow]Analyzing language distribution...")
        detected_languages, detected_lang = self.analyze_sample(model, audio_path)
        
        # Display language distribution
        lang_table = Table(title="Detected Language Distribution")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Percentage", style="green")
        
        for lang, pct in sorted(detected_languages.items(), key=lambda x: x[1], reverse=True):
            if lang != "unknown":
                lang_table.add_row(lang, f"{pct*100:.1f}%")
        
        self.console.print(lang_table)
        
        # Step 2: Get optimal parameters
        params = self.get_optimal_parameters(detected_languages, user_language)
        
        # Step 3: Transcribe with optimal parameters
        self.console.print(f"[bold green]Transcribing with optimized parameters...")
        segments, info = model.transcribe(
            audio_path,
            **params
        )
        
        # Step 4: Post-process segments
        processed_segments = self.post_process_segments(segments)
        
        return processed_segments, info
    
    def post_process_segments(self, segments):
        """Post-process segments for better quality"""
        processed = []
        
        # Convert generator to list
        segment_list = list(segments)
        
        # Process each segment
        for i, segment in enumerate(segment_list):
            text = segment.text.strip()
            
            # Skip empty segments
            if not text:
                continue
                
            # Detect language of segment
            lang, conf = self.language_detector.detect_language(text)
            
            # Apply language-specific text normalization
            text = self.normalize_text(text, lang)
            
            # Store processed segment
            processed.append({
                'id': len(processed),
                'start': segment.start,
                'end': segment.end,
                'text': text,
                'words': segment.words,
                'language': lang
            })
            
        # Merge very short segments with same language
        merged = []
        if processed:
            current = processed[0]
            
            for next_seg in processed[1:]:
                # If segments are close in time, same language, and current is short
                if (next_seg['start'] - current['end'] < 0.5 and
                    current['language'] == next_seg['language'] and
                    len(current['text'].split()) < 5):
                    # Merge segments
                    current['end'] = next_seg['end']
                    current['text'] += " " + next_seg['text']
                    # Merge words if available
                    if current['words'] and next_seg['words']:
                        current['words'].extend(next_seg['words'])
                else:
                    merged.append(current)
                    current = next_seg
            
            merged.append(current)
            return merged
        
        return processed
    
    def normalize_text(self, text, language):
        """Apply language-specific text normalization"""
        if not text:
            return text
            
        # Common fixes
        text = re.sub(r'\s+', ' ', text).strip()
        
        if language == "ar":
            # Arabic specific normalization
            arabic_norm = {
                'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alif forms
                'ة': 'ه',  # Ta marbuta
                'ى': 'ي',  # Alif maksura
                'ﻷ': 'لا', 'ﻹ': 'لا', 'ﻵ': 'لا',  # Ligatures
            }
            for orig, repl in arabic_norm.items():
                text = text.replace(orig, repl)
                
        elif language == "ur":
            # Urdu specific normalization
            urdu_norm = {
                # Add Urdu-specific replacements
                '۔': '.',  # Convert Urdu full stop to period
            }
            for orig, repl in urdu_norm.items():
                text = text.replace(orig, repl)
        
        elif language == "en":
            # English specific fixes
            # Capitalize first letter of sentences
            text = '. '.join(s.capitalize() for s in text.split('. '))
            # Fix spacing around punctuation
            text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        
        return text


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
        
        # Determine output path (same directory as input)
        input_path = Path(file_path)
        output_dir = input_path.parent
        output_file = output_dir / f"{input_path.stem}.srt"
        
        # Language selection
        language_options = {
            "1": ("ar", "Arabic"),
            "2": ("ur", "Urdu"),
            "3": ("en", "English"),
            "4": (None, "Auto-detect (recommended for mixed content)")
        }
        
        console.print("Select language preference:")
        for key, (code, name) in language_options.items():
            console.print(f"[bold cyan]{key}.[/bold cyan] {name}")
        
        lang_choice = Prompt.ask("Enter option number", default="4")
        language = language_options.get(lang_choice, (None, ""))[0]
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process audio
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Initialize processors
                audio_processor = AudioProcessor(normalize=True, noise_reduction=True)
                transcription_optimizer = TranscriptionOptimizer(model_size="large-v3")
                
                # Extract and enhance audio
                task = progress.add_task("[bold yellow]Processing audio...", total=None)
                audio_path = audio_processor.extract_audio(file_path, temp_dir)
                progress.update(task, completed=True)
                
                # Transcribe with optimal settings
                task = progress.add_task("[bold green]Transcribing audio...", total=None)
                segments, info = transcription_optimizer.transcribe_with_optimal_settings(
                    audio_path, 
                    user_language=language
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