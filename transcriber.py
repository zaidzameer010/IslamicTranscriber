import os
import torch
import re
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from datetime import timedelta
from pathlib import Path
from pydub import AudioSegment
import tempfile
import warnings
from faster_whisper import WhisperModel
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import unicodedata

# Disable warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

console = Console()

# Language-specific character sets
URDU_CHARS = set('ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھءیےآأؤإئ')
ARABIC_CHARS = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ')

# Enhanced language support with language-specific models
LANGUAGE_MODELS = {
    'ar': {
        'name': 'Arabic',
        'wav2vec_model': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
        'char_set': ARABIC_CHARS,
        'initial_prompt': 'هذا نص باللغة العربية الفصحى.',  # "This is Modern Standard Arabic text."
    },
    'ur': {
        'name': 'Urdu',
        'whisper_model': 'large-v3',
        'char_set': URDU_CHARS,
        'initial_prompt': 'یہ اردو زبان میں متن ہے۔',  # "This is text in Urdu language."
    }
}

# Update supported languages
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

def save_srt(segments, output_file: str, language=None):
    """Save transcription segments in SRT format with language-specific processing."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            
            # Process text based on language
            text = process_text_by_language(segment.text.strip(), language)
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def extract_audio_from_video(video_path, output_path=None):
    """Extract audio from video file using pydub."""
    try:
        # If no output path specified, create temp file
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "temp_audio.wav")

        # Load video file and extract audio
        video = AudioSegment.from_file(video_path)
        
        # Export as WAV for better compatibility with Whisper
        video.export(output_path, format="wav")
        
        return output_path
    except Exception as e:
        console.print(f"[red]Error extracting audio from video: {str(e)}")
        return None

def process_media_file(file_path):
    """Process media file and return path to audio file."""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # If it's already an audio file, return the path
        if file_ext in SUPPORTED_AUDIO_FORMATS:
            return str(file_path)
        
        # If it's a video file, extract the audio
        elif file_ext in SUPPORTED_VIDEO_FORMATS:
            console.print("[yellow]Extracting audio from video...[/yellow]")
            audio_path = extract_audio_from_video(file_path)
            if audio_path:
                console.print("[green]Audio extracted successfully![/green]")
                return audio_path
            else:
                console.print("[red]Failed to extract audio from video.[/red]")
                return None
        else:
            console.print(f"[red]Unsupported file format: {file_ext}[/red]")
            return None
            
    except Exception as e:
        console.print(f"[red]Error processing media file: {str(e)}")
        return None

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

class LanguageDetector:
    def __init__(self):
        self.models = {}
        self.processors = {}
        
    def load_models(self, languages=['ar', 'ur']):
        """Load language-specific models."""
        for lang in languages:
            if lang in LANGUAGE_MODELS:
                if 'wav2vec_model' in LANGUAGE_MODELS[lang]:
                    try:
                        model_name = LANGUAGE_MODELS[lang]['wav2vec_model']
                        self.models[lang] = Wav2Vec2ForCTC.from_pretrained(model_name)
                        self.processors[lang] = Wav2Vec2Processor.from_pretrained(model_name)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not load wav2vec model for {lang}: {str(e)}[/yellow]")
                
    def detect_language_segment(self, audio_segment, sample_rate=16000):
        """Detect the dominant language in an audio segment."""
        if not self.models:  # If no models were loaded successfully
            return None
            
        confidences = {}
        
        for lang, model in self.models.items():
            try:
                inputs = self.processors[lang](
                    audio_segment, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    logits = model(inputs.input_values).logits
                    confidences[lang] = torch.max(logits).item()
            except Exception as e:
                console.print(f"[yellow]Warning: Error processing {lang}: {str(e)}[/yellow]")
                
        if confidences:
            return max(confidences.items(), key=lambda x: x[1])[0]
        return None

def load_model(model_size="large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16"):
    """Enhanced model loading with multilingual support."""
    try:
        # Initialize language detector
        lang_detector = LanguageDetector()
        lang_detector.load_models()
        
        # Load main whisper model with optimized settings
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=None,
            cpu_threads=4,
            num_workers=1,
        )
        
        if model is None:
            raise Exception("Failed to load Whisper model")
            
        return model, lang_detector
    except Exception as e:
        console.print(f"[red]Error loading models: {str(e)}")
        return None, None

def segment_audio(audio_path, segment_duration=30):
    """Segment audio into smaller chunks for analysis."""
    audio, sr = librosa.load(audio_path, sr=16000)
    segment_length = int(segment_duration * sr)
    segments = []
    
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        if len(segment) > 0.5 * segment_length:  # Only keep segments > 50% of desired length
            segments.append((segment, i / sr, (i + len(segment)) / sr))
    
    return segments, sr

def detect_script_type(text):
    """
    Detect the dominant script in the text (Urdu, Arabic, or Other).
    Returns a tuple of (script_type, confidence).
    """
    if not text or len(text.strip()) == 0:
        return 'unknown', 0.0
    
    # Count characters by script
    char_counts = {
        'ur': 0,
        'ar': 0,
        'other': 0
    }
    
    total_chars = 0
    for char in text:
        if char.isspace() or not char.isalpha():
            continue
        total_chars += 1
        if char in URDU_CHARS:
            char_counts['ur'] += 1
        elif char in ARABIC_CHARS:
            char_counts['ar'] += 1
        else:
            char_counts['other'] += 1
    
    if total_chars == 0:
        return 'unknown', 0.0
    
    # Calculate percentages
    percentages = {
        script: count / total_chars 
        for script, count in char_counts.items()
    }
    
    # Get dominant script
    dominant_script = max(percentages.items(), key=lambda x: x[1])
    
    # Only return if confidence is high enough
    if dominant_script[1] > 0.6:  # 60% threshold
        return dominant_script[0], dominant_script[1]
    return 'unknown', dominant_script[1]

def clean_arabic_text(text):
    """Clean and normalize Arabic text."""
    # Normalize Arabic text
    text = unicodedata.normalize('NFKC', text)
    
    # Fix common Arabic character issues
    replacements = {
        'ي': 'ي',  # Replace Farsi Ya with Arabic Ya
        'ك': 'ك',  # Replace Farsi Kaf with Arabic Kaf
        '‍': '',   # Remove zero-width joiner
        'ة': 'ة',  # Fix Tah Marbuta
        'ه': 'ه',  # Fix Ha
        'أ': 'أ',  # Fix Hamza
        'إ': 'إ',  # Fix Hamza
        'آ': 'آ',  # Fix Alef with Madda
        'ؤ': 'ؤ',  # Fix Waw with Hamza
        'ئ': 'ئ'   # Fix Ya with Hamza
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def clean_urdu_text(text):
    """Clean and normalize Urdu text."""
    # Normalize Urdu text
    text = unicodedata.normalize('NFKC', text)
    
    # Fix common Urdu character issues
    replacements = {
        'ی': 'ی',  # Fix Ya
        'ہ': 'ہ',  # Fix Ha
        'ھ': 'ھ',  # Fix Do-chashmi Ha
        'ے': 'ے',  # Fix Barri Ya
        'آ': 'آ',  # Fix Alef Madda
        'ں': 'ں',  # Fix Noon Ghunna
        '‍': '',   # Remove zero-width joiner
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def process_text_by_language(text, language):
    """Process text based on detected or specified language."""
    if not text:
        return text
        
    # First detect script if language not specified
    if language is None:
        script_type, confidence = detect_script_type(text)
        if script_type in ['ur', 'ar']:
            language = script_type
    
    # Apply language-specific processing
    if language == 'ar':
        return clean_arabic_text(text)
    elif language == 'ur':
        return clean_urdu_text(text)
    
    return text

def transcribe_audio(model, lang_detector, audio_path, language=None, task="transcribe",
                    beam_size=15, best_of=5, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """Enhanced transcription with multilingual support."""
    try:
        if model is None:
            raise Exception("Model not properly initialized")
            
        # For Urdu or Arabic, use direct Whisper transcription with language-specific settings
        if language in ['ur', 'ar']:
            # Set language-specific prompts
            initial_prompt = LANGUAGE_MODELS[language].get('initial_prompt') if language in LANGUAGE_MODELS else None
            
            # Get transcription
            segments_generator, info = model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                    threshold=0.5
                ),
                condition_on_previous_text=True,
                initial_prompt=initial_prompt
            )
            # Convert generator to list
            segments_list = list(segments_generator)
            return segments_list, info
            
        # For other languages or auto-detection
        if language is None and lang_detector and lang_detector.models:
            segments_list = []
            audio_segments, sr = segment_audio(audio_path)
            
            for audio_segment, start_time, end_time in audio_segments:
                detected_lang = lang_detector.detect_language_segment(audio_segment, sr)
                if detected_lang is None:
                    detected_lang = language  # Fall back to user-selected language or None
                    
                # Transcribe with detected language
                segment_results, _ = model.transcribe(
                    audio_path,
                    language=detected_lang,
                    task=task,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=temperature,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=400,
                        threshold=0.5
                    ),
                    condition_on_previous_text=True,
                    start_time=start_time,
                    end_time=end_time
                )
                segments_list.extend(list(segment_results))
        else:
            # Default transcription with specified language
            segments_generator, info = model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                    threshold=0.5
                ),
                condition_on_previous_text=True
            )
            # Convert generator to list
            segments_list = list(segments_generator)
            return segments_list, info
        
        return segments_list, None
    except Exception as e:
        console.print(f"[red]Error during transcription: {str(e)}")
        return None, None

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
            # Process media file and get audio path
            audio_path = process_media_file(file_path)
            if audio_path is None:
                return None

            try:
                model_task = progress.add_task("[bold bright_blue]Loading Whisper model...", total=100)
                model, lang_detector = load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
                if model is None:
                    raise Exception("Failed to initialize the model")
                progress.update(model_task, completed=100)

                transcribe_task = progress.add_task("[bold bright_yellow]Transcribing audio...", total=100, start=True)

                segments_list, info = transcribe_audio(model, lang_detector, audio_path, language=language, task="transcribe", 
                    beam_size=15, best_of=5, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                
                if segments_list is None:
                    raise Exception("Transcription failed")
                
                progress.update(transcribe_task, completed=100)

                # Save the transcription to SRT file with language-specific processing
                save_task = progress.add_task("[bold bright_green]Saving SRT file...", total=100)
                save_srt(segments_list, str(output_path), language)
                progress.update(save_task, completed=100)
            
            finally:
                if os.path.exists(audio_path) and os.path.basename(audio_path).startswith("temp_audio"):
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass

        console.print(f"\n[bold bright_green]✓[/bold bright_green] Transcription saved to: {output_path}")
        
        # Display language info with script detection results
        if info is not None and hasattr(info, 'language'):
            console.print(f"[bold bright_blue]Detected language:[/bold bright_blue] {info.language}")
            if hasattr(info, 'language_probability'):
                console.print(f"[bold bright_yellow]Language probability:[/bold bright_yellow] {info.language_probability:.2%}")
        elif language:
            console.print(f"[bold bright_blue]Selected language:[/bold bright_blue] {SUPPORTED_LANGUAGES.get(language, language)}")
        
        # Show script detection results for the first few segments
        if segments_list:  # Changed from checking len() to just checking if not None
            sample_text = segments_list[0].text.strip()
            script_type, confidence = detect_script_type(sample_text)
            console.print(f"[bold bright_cyan]Detected script:[/bold bright_cyan] {script_type.upper()}")
            console.print(f"[bold bright_yellow]Script confidence:[/bold bright_yellow] {confidence:.2%}")
            
        if file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
            console.print("[bold bright_magenta]Note:[/bold bright_magenta] Video file was processed by extracting its audio")
        
        return output_path

    except Exception as e:
        console.print(f"\n[bold bright_red]Error:[/bold bright_red] {str(e)}")
        return None

if __name__ == '__main__':
    transcribe()
