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
        "beam_size": 15,
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

# Islamic terminology dictionary for context-aware corrections
ISLAMIC_TERMINOLOGY = {
    "ar": {
        "الله": {"correct": "اللَّه", "context": ["بسم", "رسول", "صلى"]},
        "محمد": {"correct": "مُحَمَّد", "context": ["النبي", "رسول", "صلى"]},
        "القرآن": {"correct": "القُرْآن", "context": ["تلاوة", "آيات", "سورة"]},
        "الحديث": {"correct": "الحَدِيث", "context": ["صحيح", "رواه", "في"]},
        "الصلاة": {"correct": "الصَّلاة", "context": ["أقام", "فرض", "وقت"]},
        "المسجد": {"correct": "المَسْجِد", "context": ["في", "إلى", "من"]},
        "الإسلام": {"correct": "الإِسْلام", "context": ["دين", "أركان", "شريعة"]},
        "الإيمان": {"correct": "الإِيمان", "context": ["أركان", "شعب", "زيادة"]}
    },
    "ur": {
        "اللہ": {"correct": "اللہ تعالیٰ", "context": ["بسم", "رسول", "صلی"]},
        "محمد": {"correct": "محمد ﷺ", "context": ["حضرت", "رسول", "نبی"]},
        "قرآن": {"correct": "قرآن پاک", "context": ["تلاوت", "آیات", "سورہ"]},
        "حدیث": {"correct": "حدیث شریف", "context": ["صحیح", "روایت", "میں"]},
        "نماز": {"correct": "نماز", "context": ["فرض", "وقت", "پڑھنا"]},
        "مسجد": {"correct": "مسجد", "context": ["میں", "سے", "کو"]},
        "اسلام": {"correct": "اسلام", "context": ["دین", "ارکان", "شریعت"]},
        "ایمان": {"correct": "ایمان", "context": ["ارکان", "لانا", "پر"]}
    },
    "en": {
        "allah": {"correct": "Allah", "context": ["praise", "god", "worship"]},
        "muhammad": {"correct": "Muhammad (peace be upon him)", "context": ["prophet", "messenger", "said"]},
        "quran": {"correct": "Qur'an", "context": ["holy", "verse", "surah"]},
        "hadith": {"correct": "Hadith", "context": ["narrated", "reported", "authentic"]},
        "prayer": {"correct": "prayer", "context": ["five", "daily", "time"]},
        "mosque": {"correct": "mosque", "context": ["in", "to", "from"]},
        "islam": {"correct": "Islam", "context": ["religion", "pillars", "faith"]},
        "iman": {"correct": "Iman", "context": ["faith", "belief", "pillars"]}
    }
}

# Common error corrections for each language
CORRECTIONS = {
    "ar": {
        # Common word spacing errors
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
        # Islamic terms with proper diacritics
        "سبحان الله": "سُبْحَانَ اللَّه",
        "الحمد لله": "الحَمْدُ لِلَّه",
        "ان شاء الله": "إِنْ شَاءَ اللَّه",
        "بسم الله": "بِسْمِ اللَّه",
        # Common phonetic confusions
        "ظ/ض": "ظ",
        "س/ص": "ص",
        "ح/ه": "ح",
        # Tashkeel corrections
        "اللة": "اللَّه",
        "رحمة": "رَحْمَة",
        "بركة": "بَرَكَة"
    },
    "ur": {
        # Common word spacing errors
        "ک يا": "کیا",
        "ه ے": "هے",
        # Common Islamic terms
        "سبحان اللہ": "سبحان اللہ",
        "الحمد للہ": "الحمد للہ",
        "ان شاء اللہ": "ان شاء اللہ",
        "بسم اللہ": "بسم اللہ الرحمٰن الرحیم",
        # Phonetic corrections
        "ذ/ز": "ذ",
        "ث/س": "ث",
        "ح/ہ": "ح",
        # Common mistakes
        "انشااللہ": "ان شاء اللہ",
        "ماشااللہ": "ما شاء اللہ",
        "جزاکاللہ": "جزاک اللہ"
    },
    "en": {
        # Common word spacing errors
        "i s": "is",
        "i t": "it",
        # Common Islamic terms
        "inshallah": "In sha Allah",
        "mashallah": "Ma sha Allah",
        "alhamdulillah": "Alhamdulillah",
        "subhanallah": "Subhan Allah",
        # Common capitalization mistakes
        "ramadan": "Ramadan",
        "eid": "Eid",
        "surah": "Surah",
        "ayah": "Ayah",
        "jummah": "Jummah",
        "hajj": "Hajj",
        "umrah": "Umrah",
        "salah": "Salah"
    }
}

# Arabic-specific character sets and patterns
ARABIC_CHARS = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْ')
ARABIC_PATTERNS = [
    r'\b(قال|قالت|يقول|تقول|قلت)\b',  # Speech indicators
    r'\b(الله|الرحمن|الرحيم)\b',  # Allah's names
    r'\b(آية|سورة|القرآن)\b',  # Quran references
    r'\b(حديث|صحيح|روى)\b',  # Hadith references
]

# Urdu-specific character sets and patterns
URDU_CHARS = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيےٹڈڑژڈھہ')
URDU_PATTERNS = [
    r'\b(کہا|کہتے|کہتی|بولے)\b',  # Speech indicators
    r'\b(اللہ|رحمٰن|رحیم)\b',  # Allah's names
    r'\b(آیت|سورہ|قرآن)\b',  # Quran references
    r'\b(حدیث|صحیح|روایت)\b',  # Hadith references
    r'\b(ہے|ہیں|تھا|تھے|گا|گے)\b',  # Urdu-specific verbs
]

# English character set
ENGLISH_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


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


def extract_audio(file_path: str, temp_dir: str, optimize: bool = True, progress_callback=None) -> str:
    """Extract audio efficiently for transcription"""
    console = Console()
    
    input_path = Path(file_path)
    file_name = input_path.stem
    temp_audio_path = os.path.join(temp_dir, f"{file_name}.wav")
    
    with console.status(f"[bold green]Extracting audio from {input_path.name}...") if not progress_callback else nullcontext():
        try:
            # Load audio
            if progress_callback:
                progress_callback(0, "Loading audio file...")
            audio = AudioSegment.from_file(file_path)
            if progress_callback:
                progress_callback(25, "Converting to mono...")
            
            # Efficient conversion for speech recognition
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz is optimal for Whisper
            if progress_callback:
                progress_callback(50, "Optimizing audio...")
            
            # For large files, downsample to reduce processing time
            if optimize and len(audio) > 600000:  # If longer than 10 minutes
                audio = audio.set_sample_width(2)  # 16-bit is sufficient
            if progress_callback:
                progress_callback(75, "Exporting audio...")
            
            # Export optimized audio
            audio.export(temp_audio_path, format="wav")
            if progress_callback:
                progress_callback(100, "Audio extraction complete")
            return temp_audio_path
            
        except Exception as e:
            console.print(f"[bold red]Error extracting audio: {str(e)}")
            raise


def transcribe_audio(
    audio_path: str, 
    quality_preset: str = "accurate",  
    language: str = None,  
    device: str = None,
    compute_type: str = None,
    progress_callback=None
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
    
    if progress_callback:
        progress_callback(0, f"Initializing Whisper {model_size} model...")
        
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
    
    if progress_callback:
        progress_callback(20, "Setting up transcription parameters...")
    
    # Enhanced parameters for accurate transcription and timestamps
    transcription_params = {
        "beam_size": max(beam_size, 15),  # Larger beam size for better search
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
    
    if progress_callback:
        progress_callback(30, "Starting transcription process...")
    
    # Transcribe audio
    segments, info = model.transcribe(audio_path, **transcription_params)
    
    if progress_callback:
        progress_callback(70, "Processing transcription segments...")
    
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
    
    if progress_callback:
        progress_callback(90, "Applying post-processing...")
    
    # Apply post-processing fixes
    processed_segments = post_process_segments(segments_list, detected_language, language)
    
    if progress_callback:
        progress_callback(100, "Transcription complete")
    
    return processed_segments


def is_arabic_text(text: str) -> bool:
    """Determine if text is primarily Arabic based on character distribution and patterns"""
    if not text.strip():
        return False
        
    # Count Arabic characters
    arabic_char_count = sum(1 for char in text if char in ARABIC_CHARS)
    total_chars = len(text.strip())
    
    # Check for Arabic patterns
    pattern_matches = sum(1 for pattern in ARABIC_PATTERNS if re.search(pattern, text))
    
    # Calculate Arabic score
    arabic_score = (arabic_char_count / total_chars if total_chars > 0 else 0) + (0.2 * pattern_matches)
    
    # Check for specific Arabic-only characters that don't appear in Urdu
    arabic_specific_chars = set('ةًٌٍَُِّْ')
    has_arabic_specific = any(char in arabic_specific_chars for char in text)
    
    return arabic_score > 0.7 or has_arabic_specific


def is_urdu_text(text: str) -> bool:
    """Determine if text is primarily Urdu based on character distribution and patterns"""
    if not text.strip():
        return False
        
    # Count Urdu characters
    urdu_char_count = sum(1 for char in text if char in URDU_CHARS)
    total_chars = len(text.strip())
    
    # Check for Urdu patterns
    pattern_matches = sum(1 for pattern in URDU_PATTERNS if re.search(pattern, text))
    
    # Calculate Urdu score
    urdu_score = (urdu_char_count / total_chars if total_chars > 0 else 0) + (0.2 * pattern_matches)
    
    # Check for specific Urdu-only characters that don't appear in Arabic
    urdu_specific_chars = set('ےٹڈڑژڈھہ')
    has_urdu_specific = any(char in urdu_specific_chars for char in text)
    
    return urdu_score > 0.7 or has_urdu_specific


def detect_language_context(text: str, window_size: int = 5) -> str:
    """Enhanced language detection using context windows and statistical analysis"""
    # Split text into words
    words = text.split()
    if not words:
        return "en"  # Default to English for empty text
    
    # Initialize language scores
    scores = {"ar": 0, "ur": 0, "en": 0}
    
    # Check if entire text is clearly Arabic or Urdu
    if is_arabic_text(text):
        return "ar"
    if is_urdu_text(text):
        return "ur"
    
    # Analyze each word in context
    for i in range(len(words)):
        word = words[i]
        
        # Get context window
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        context = words[start:end]
        context_text = ' '.join(context)
        
        # Score based on character sets
        word_chars = set(word)
        ar_score = len(word_chars & ARABIC_CHARS) / max(len(word_chars), 1)
        ur_score = len(word_chars & URDU_CHARS) / max(len(word_chars), 1)
        en_score = len(word_chars & ENGLISH_CHARS) / max(len(word_chars), 1)
        
        # Check for language-specific characters
        if any(char in set('ةًٌٍَُِّْ') for char in word):
            ar_score += 0.5  # Boost Arabic score for Arabic-specific chars
        if any(char in set('ےٹڈڑژڈھہ') for char in word):
            ur_score += 0.5  # Boost Urdu score for Urdu-specific chars
        
        # Add context-based scoring
        for ctx_word in context:
            if any(term in ctx_word.lower() for term in ISLAMIC_TERMINOLOGY.get("ar", {})):
                ar_score += 0.2
            if any(term in ctx_word.lower() for term in ISLAMIC_TERMINOLOGY.get("ur", {})):
                ur_score += 0.2
            if any(term in ctx_word.lower() for term in ISLAMIC_TERMINOLOGY.get("en", {})):
                en_score += 0.2
        
        # Check for language-specific patterns in context
        for pattern in ARABIC_PATTERNS:
            if re.search(pattern, context_text):
                ar_score += 0.3
        for pattern in URDU_PATTERNS:
            if re.search(pattern, context_text):
                ur_score += 0.3
        
        # Update overall scores
        scores["ar"] += ar_score
        scores["ur"] += ur_score
        scores["en"] += en_score
    
    # Return the language with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def apply_context_aware_corrections(text: str, language: str) -> str:
    """Apply context-aware corrections using surrounding words"""
    words = text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        # Get context window
        start = max(0, i - 2)
        end = min(len(words), i + 3)
        context = ' '.join(words[start:end]).lower()
        
        # Check Islamic terminology dictionary
        word_lower = word.lower()
        if word_lower in ISLAMIC_TERMINOLOGY.get(language, {}):
            term_info = ISLAMIC_TERMINOLOGY[language][word_lower]
            # Check if context matches
            if any(ctx in context for ctx in term_info["context"]):
                word = term_info["correct"]
        
        # Apply regular corrections
        if language in CORRECTIONS:
            for error, correction in CORRECTIONS[language].items():
                if word == error:
                    word = correction
        
        corrected_words.append(word)
    
    return ' '.join(corrected_words)


def format_mixed_language_text(text: str, primary_language: str) -> str:
    """Format text with mixed languages, adding appropriate markers for clarity"""
    words = text.split()
    formatted_words = []
    current_language = None
    language_segment = []
    
    # Process each word to identify language segments
    for word in words:
        # Skip empty words
        if not word.strip():
            continue
            
        # Determine word language
        word_chars = set(word)
        ar_chars = len(word_chars & ARABIC_CHARS)
        ur_chars = len(word_chars & URDU_CHARS)
        en_chars = len(word_chars & ENGLISH_CHARS)
        
        # Determine language based on character distribution
        if ar_chars > ur_chars and ar_chars > en_chars:
            word_language = "ar"
        elif ur_chars > ar_chars and ur_chars > en_chars:
            word_language = "ur"
        elif en_chars > ar_chars and en_chars > ur_chars:
            word_language = "en"
        else:
            # Default to primary language if unclear
            word_language = primary_language
        
        # Handle language transitions
        if current_language is None:
            current_language = word_language
            language_segment.append(word)
        elif word_language == current_language:
            language_segment.append(word)
        else:
            # Process completed language segment
            segment_text = ' '.join(language_segment)
            
            # Apply language-specific formatting
            if current_language != primary_language and len(language_segment) > 1:
                # Add formatting for non-primary language segments
                if current_language == "ar" and primary_language == "ur":
                    segment_text = f"[{segment_text}]"  # Arabic in brackets when primary is Urdu
                elif current_language == "ur" and primary_language == "ar":
                    segment_text = f"<{segment_text}>"  # Urdu in angle brackets when primary is Arabic
            
            formatted_words.append(segment_text)
            
            # Start new segment
            current_language = word_language
            language_segment = [word]
    
    # Process the last segment
    if language_segment:
        segment_text = ' '.join(language_segment)
        if current_language != primary_language and len(language_segment) > 1:
            if current_language == "ar" and primary_language == "ur":
                segment_text = f"[{segment_text}]"  # Arabic in brackets when primary is Urdu
            elif current_language == "ur" and primary_language == "ar":
                segment_text = f"<{segment_text}>"  # Urdu in angle brackets when primary is Arabic
        formatted_words.append(segment_text)
    
    return ' '.join(formatted_words)


def post_process_segments(segments, detected_language, selected_language=None):
    """Apply post-processing to improve transcription quality and timestamp accuracy"""
    processed = []
    
    # Determine primary language (user selected or detected)
    primary_language = selected_language if selected_language else detected_language
    
    # Process each segment
    for segment in segments:
        text = segment["text"]
        
        # Detect language for this segment (may differ from overall detected language)
        segment_language = detect_language_context(text) 
        
        # Apply basic corrections first
        corrections = {}
        if segment_language in ["ar", "ur", "en"]:
            corrections = CORRECTIONS[segment_language]
        else:
            # For mixed content, include all corrections
            for lang_corrections in CORRECTIONS.values():
                corrections.update(lang_corrections)
        
        # Apply basic text corrections
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        
        # Apply context-aware corrections
        text = apply_context_aware_corrections(text, segment_language)
        
        # Update with corrected text
        segment["text"] = text
        segment["detected_language"] = segment_language  # Store detected language for reference
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
                
                # Only merge if they are the same language
                if previous.get('detected_language') == current.get('detected_language'):
                    # Merge segments
                    previous['end'] = current['end']
                    previous['text'] += " " + current['text']
                    
                    # If words info exists, merge it
                    if 'words' in previous and 'words' in current:
                        previous['words'].extend(current['words'])
                else:
                    merged.append(current)
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
        
        # Set quality preset directly instead of asking user
        quality_preset = "accurate"
        
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
        
        # Update Urdu prompt for better language separation when Urdu is selected
        if language == "ur":
            LANGUAGE_PROMPTS["ur"] = """بسم اللہ الرحمٰن الرحیم۔ یہ اردو میں ٹرانسکرپشن ہے:

اس آڈیو میں اردو اور عربی دونوں زبانیں ہو سکتی ہیں۔ براہ کرم:
- اردو متن کو اردو میں لکھیں
- عربی اقتباسات کو عربی میں لکھیں
- دونوں زبانوں کے درمیان واضح فرق رکھیں
- قرآنی آیات اور احادیث کو درست عربی میں لکھیں"""
        
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