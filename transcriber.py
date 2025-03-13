import os
import sys
import tempfile
import re
from pathlib import Path
from typing import List, Optional
from collections import Counter
import difflib

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

# Add a new preset optimized for Apple Silicon
APPLE_SILICON_PRESET = {
    "model_size": "large-v3",
    "beam_size": 10,  # Slightly reduced beam size for better performance
    "vad_filter": True,
    "compute_type": "float16",  # Apple Silicon works well with float16
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
    "ur": """بسم اللہ الرحمٰن الرحیم۔ یہ اردو میں ٹرانسکرپشن ہے:

اردو زبان میں درج ذیل باتوں کا خیال رکھیں:
- حروف کی درست ادائیگی اور املا
- مناسب رموز اوقاف کا استعمال (۔، ؛، ؟)
- جملوں کے درمیان مناسب وقفے
- اردو کے خاص حروف جیسے ڑ، ٹ، ڈ، ھ، ں کی درست تحریر
- اسلامی اصطلاحات کی درست املا
""",
    "en": "The following is English speech transcription:",

    # Enhanced Urdu prompt
    "ur_enhanced": """بسم اللہ الرحمٰن الرحیم۔ یہ اردو میں ٹرانسکرپشن ہے:

اردو زبان میں درج ذیل باتوں کا خیال رکھیں:
- حروف کی درست ادائیگی اور املا
- مناسب رموز اوقاف کا استعمال (۔، ؛، ؟)
- جملوں کے درمیان مناسب وقفے
- اردو کے خاص حروف جیسے ڑ، ٹ، ڈ، ھ، ں کی درست تحریر
- مذہبی اصطلاحات جیسے "اللہ"، "رسول"، "نبی"، "سنت" وغیرہ کو درست املا میں لکھیں
- عربی اقتباسات کو ان کے اصل عربی رسم الخط میں لکھیں
- انگریزی الفاظ کو ان کی درست انگریزی املا میں لکھیں
"""
}

# Mixed language prompts for better handling of code-switching
MIXED_LANGUAGE_PROMPTS = {
    "ur_ar": """
    ﺑﺴﻢ اﷲ اﻟﺮﺣﻤﻦ اﻟﺮﺣﻴﻢ۔ یہ اردو اور عربی میں ملی جلی ٹرانسکرپشن ہے:

    ملاحظات:
    - ہر زبان کو اس کے اصل رسم الخط میں لکھیں
    - عربی الفاظ عربی املا میں اور اردو الفاظ اردو املا میں لکھیں
    - مذہبی اصطلاحات کی درست املا کا خیال رکھیں
    - قرآنی آیات کو عربی میں صحیح طور پر نقل کریں
    - حدیث کے الفاظ کو ان کے اصل عربی میں لکھیں جب بھی انہیں عربی میں پڑھا جائے
    - اردو، عربی، اور انگریزی کوڈ سوئچنگ کا خیال رکھیں
    """,
}

DEFAULT_MULTILINGUAL_PROMPT = """The following audio may contain a mix of Arabic, Urdu, and English.
Please transcribe each language accurately with proper punctuation while maintaining the original language."""

# Islamic terminology database for proper transcription
ISLAMIC_TERMS = {
    # Common mistranscribed Islamic terms with corrections (Arabic/Urdu)
    "alah": "اللہ",
    "allah": "اللہ",
    "rasool": "رسول",
    "rasul": "رسول",
    "nabi": "نبی",
    "quran": "قرآن",
    "koran": "قرآن",
    "salat": "صلاة",
    "namaz": "نماز",
    "sunnah": "سنت",
    "sunnat": "سنت",
    "hadith": "حدیث",
    "hadis": "حدیث",
    "bismillah": "بسم اللہ",
    "bismilla": "بسم اللہ",
    "subhanallah": "سبحان اللہ",
    "alhamdulillah": "الحمد للہ",
    "allahu akbar": "اللہ اکبر",
    "inshallah": "ان شاء اللہ",
    "mashallah": "ما شاء اللہ",
    "astaghfirullah": "استغفر اللہ",
    "jazakallah": "جزاک اللہ",
    "imam": "امام",
    "masjid": "مسجد",
    "ramadan": "رمضان",
    "eid": "عید",
    "hajj": "حج",
    "umrah": "عمرہ",
    "dua": "دعا",
    "zakat": "زکاۃ",
    "sadaqah": "صدقہ",
    "tawhid": "توحید",
    "shirk": "شرک",
    "jannah": "جنّہ",
    "jahannam": "جہنّم",
    "akhirah": "آخرت",
    "deen": "دین"
}

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
        # Common Urdu transcription errors
        "ک يا": "کیا",
        "ه ے": "هے",
        "ن ه": "نہ",
        "ک ه": "کہ",
        "ج ه": "جہ",
        "ت ه": "تہ",
        "پ ه": "پہ",
        "ب ه": "بہ",
        # Common spacing issues
        " ۔": "۔",
        " ،": "،",
        "  ": " ",
        # Common Urdu word corrections
        "ہے ں": "ہیں",
        "کر تے": "کرتے",
        "کر تا": "کرتا",
        "ہو تا": "ہوتا",
        "ہو تے": "ہوتے",
        "جا تا": "جاتا",
        "جا تے": "جاتے",
    },
    "en": {
        # Add common English transcription errors here
        "i s": "is",
        "i t": "it",
    }
}

# Expanded corrections for Urdu
URDU_CORRECTIONS = {
    # Common spacing issues
    "ک ے": "کے",
    "ہ و": "ہو",
    "ن ے": "نے",
    "م ی ں": "میں",
    "ہ ے": "ہے",
    "ک ی": "کی",
    "ک ر": "کر",
    "ک و": "کو",
    "ا س": "اس",
    "ا ن": "ان",
    "ت و": "تو",
    "ج و": "جو",
    "س ے": "سے",

    # Common transcription errors for Urdu
    "ھے": "ہے",
    "مین": "میں",
    "ھی": "ہی",
    "ھو": "ہو",
    "ھم": "ہم",
    "کھا": "کہا",
    "یھ": "یہ",
    "وھ": "وہ",

    # Additional corrections
    "انھوں": "انہوں",
    "انھی": "انہی",
    "انھیں": "انہیں",
    "جاۓ": "جائے",
    "آۓ": "آئے",
    "گۓ": "گئے",
    "ہوۓ": "ہوئے",
    "ہے، ": "ہے، ",
    "ہیں، ": "ہیں، ",
    "ل ی ے": "لیے",
    "گ ی ا": "گیا",
    "د ی ا": "دیا",
    "ر ہ ا": "رہا",
    "ر ہ ے": "رہے",
    "چ ا ہ ی ے": "چاہیے",
    "چ ا ہ ت ا": "چاہتا",
    "چ ا ہ ت ے": "چاہتے",
    "س ک ت ا": "سکتا",
    "س ک ت ے": "سکتے",
    "ب و ل ت ا": "بولتا",
    "ب و ل ت ے": "بولتے",
    "م ل ت ا": "ملتا",
    "م ل ت ے": "ملتے"
}

# Update Urdu corrections by merging dictionaries
CORRECTIONS["ur"] = {**CORRECTIONS["ur"], **URDU_CORRECTIONS}


def get_device_info():
    """Determine the best available device and compute type with enhanced Apple Silicon support"""
    try:
        import torch
        import platform
        
        # Check if running on macOS
        is_macos = platform.system() == "Darwin"
        is_apple_silicon = False
        
        # Check for Apple Silicon specifically
        if is_macos:
            try:
                # Check processor type on macOS
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                       capture_output=True, text=True)
                processor_info = result.stdout.strip()
                is_apple_silicon = "Apple" in processor_info
            except:
                # Fallback detection method
                is_apple_silicon = platform.processor() == 'arm'
        
        # Device selection logic with enhanced Apple Silicon support
        if torch.cuda.is_available():
            return "cuda", "float16", False
        elif is_macos and is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Return that we're on Apple Silicon for special optimizations
            return "mps", "float16", True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Regular MPS (not Apple Silicon)
            return "mps", "float16", False
        else:
            return "cpu", "int8", False
    except ImportError:
        return "cpu", "int8", False


def detect_apple_silicon_model():
    """Detect the specific Apple Silicon model (M1, M2, M3, etc.) and return optimization settings"""
    import platform
    import subprocess
    import re
    
    # Default optimization settings
    optimization_settings = {
        "model": "unknown",
        "cores": 8,
        "neural_engine_cores": 16,
        "memory_bandwidth": "standard",
        "recommended_batch_size": 8,
        "recommended_threads": 4,
        "memory_optimization_level": "medium"
    }
    
    # Only proceed if we're on macOS
    if platform.system() != "Darwin":
        return optimization_settings
    
    try:
        # Check if we're on Apple Silicon
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                               capture_output=True, text=True)
        processor_info = result.stdout.strip()
        
        # Not Apple Silicon
        if "Apple" not in processor_info:
            return optimization_settings
            
        # Try to identify the specific model
        m1_pattern = re.compile(r'M1(\s+)?(Pro|Max|Ultra)?', re.IGNORECASE)
        m2_pattern = re.compile(r'M2(\s+)?(Pro|Max|Ultra)?', re.IGNORECASE)
        m3_pattern = re.compile(r'M3(\s+)?(Pro|Max|Ultra)?', re.IGNORECASE)
        
        # Check for M1 variants
        m1_match = m1_pattern.search(processor_info)
        if m1_match:
            optimization_settings["model"] = "M1"
            variant = m1_match.group(2).lower() if m1_match.group(2) else "base"
            
            if variant == "base":
                optimization_settings.update({
                    "cores": 8,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 6,
                    "recommended_threads": 4,
                    "memory_optimization_level": "high"  # More conservative with base M1
                })
            elif variant == "pro":
                optimization_settings.update({
                    "cores": 10,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 8,
                    "recommended_threads": 6,
                    "memory_optimization_level": "medium"
                })
            elif variant == "max":
                optimization_settings.update({
                    "cores": 10,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 10,
                    "recommended_threads": 8,
                    "memory_optimization_level": "low"  # More memory available
                })
            elif variant == "ultra":
                optimization_settings.update({
                    "cores": 20,
                    "neural_engine_cores": 32,
                    "recommended_batch_size": 16,
                    "recommended_threads": 12,
                    "memory_optimization_level": "low"
                })
                
        # Check for M2 variants
        m2_match = m2_pattern.search(processor_info)
        if m2_match:
            optimization_settings["model"] = "M2"
            variant = m2_match.group(2).lower() if m2_match.group(2) else "base"
            
            if variant == "base":
                optimization_settings.update({
                    "cores": 8,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 8,
                    "recommended_threads": 6,
                    "memory_optimization_level": "medium"
                })
            elif variant == "pro":
                optimization_settings.update({
                    "cores": 12,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 10,
                    "recommended_threads": 8,
                    "memory_optimization_level": "medium"
                })
            elif variant == "max":
                optimization_settings.update({
                    "cores": 12,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 12,
                    "recommended_threads": 10,
                    "memory_optimization_level": "low"
                })
            elif variant == "ultra":
                optimization_settings.update({
                    "cores": 24,
                    "neural_engine_cores": 32,
                    "recommended_batch_size": 16,
                    "recommended_threads": 16,
                    "memory_optimization_level": "low"
                })
                
        # Check for M3 variants
        m3_match = m3_pattern.search(processor_info)
        if m3_match:
            optimization_settings["model"] = "M3"
            variant = m3_match.group(2).lower() if m3_match.group(2) else "base"
            
            if variant == "base":
                optimization_settings.update({
                    "cores": 8,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 10,
                    "recommended_threads": 6,
                    "memory_optimization_level": "medium"
                })
            elif variant == "pro":
                optimization_settings.update({
                    "cores": 12,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 12,
                    "recommended_threads": 8,
                    "memory_optimization_level": "medium"
                })
            elif variant == "max":
                optimization_settings.update({
                    "cores": 16,
                    "neural_engine_cores": 16,
                    "recommended_batch_size": 16,
                    "recommended_threads": 12,
                    "memory_optimization_level": "low"
                })
            elif variant == "ultra":
                optimization_settings.update({
                    "cores": 32,
                    "neural_engine_cores": 32,
                    "recommended_batch_size": 20,
                    "recommended_threads": 16,
                    "memory_optimization_level": "low"
                })
        
        # Try to get actual core count if available
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], 
                                   capture_output=True, text=True)
            core_count = int(result.stdout.strip())
            optimization_settings["actual_cores"] = core_count
        except:
            pass
            
        return optimization_settings
        
    except Exception as e:
        print(f"Note: Could not detect Apple Silicon model: {e}")
        return optimization_settings


def optimize_for_apple_silicon():
    """Apply system-level optimizations for Apple Silicon"""
    import platform
    
    # Only proceed if we're on macOS
    if platform.system() != "Darwin":
        return False
    
    try:
        # Check if we're on Apple Silicon
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.optional.arm64'], 
                               capture_output=True, text=True)
        is_arm64 = result.stdout.strip() == '1'
        
        if not is_arm64:
            return False
            
        # Detect specific Apple Silicon model and get optimized settings
        apple_silicon_info = detect_apple_silicon_model()
        
        # Apply Apple Silicon specific optimizations
        
        # 1. Set environment variables for better Metal performance
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # 2. Configure thread settings for Apple Silicon based on detected model
        os.environ['MPS_USE_NEURAL_ENGINE'] = '1'
        
        # Set optimal thread count based on detected model
        recommended_threads = apple_silicon_info.get("recommended_threads", 4)
        os.environ['OMP_NUM_THREADS'] = str(recommended_threads)
        os.environ['MKL_NUM_THREADS'] = str(recommended_threads)
        
        # 3. Set memory management for better performance
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
        # 4. Enable Metal Performance Shaders
        os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '0'
        
        # 5. Apply memory optimization with level based on detected model
        memory_level = apple_silicon_info.get("memory_optimization_level", "medium")
        optimize_memory_for_apple_silicon(memory_level)
        
        return True
    except Exception as e:
        print(f"Note: Could not apply Apple Silicon optimizations: {e}")
        return False


def optimize_memory_for_apple_silicon(optimization_level="medium"):
    """Apply memory optimizations specifically for Apple Silicon devices"""
    try:
        import torch
        import gc
        import platform
        import subprocess
        
        # Only proceed if we're on macOS with Apple Silicon
        if platform.system() != "Darwin":
            return False
            
        # Check for Apple Silicon
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                   capture_output=True, text=True)
            total_memory_bytes = int(result.stdout.strip())
            total_memory_gb = total_memory_bytes / (1024**3)
            
            # Get available memory
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            vm_stat_output = result.stdout
            
            # Parse vm_stat output to get page size and free pages
            page_size_line = [line for line in vm_stat_output.split('\n') if 'page size of' in line]
            if page_size_line:
                page_size = int(page_size_line[0].split('page size of')[1].strip().split(' ')[0])
            else:
                page_size = 4096  # Default page size
                
            free_pages_line = [line for line in vm_stat_output.split('\n') if 'Pages free' in line]
            if free_pages_line:
                free_pages = int(free_pages_line[0].split(':')[1].strip().replace('.', ''))
            else:
                free_pages = 0
                
            free_memory_bytes = free_pages * page_size
            free_memory_gb = free_memory_bytes / (1024**3)
            
            # Configure memory optimizations based on available memory and optimization level
            # For Apple Silicon with unified memory, we need to be more conservative
            
            # 1. Force garbage collection to free up memory
            gc.collect()
            
            # 2. Set PyTorch memory fraction based on available memory and optimization level
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Memory fraction based on optimization level
                memory_fractions = {
                    "low": 0.8,     # Less conservative (for high-end models with lots of RAM)
                    "medium": 0.7,  # Balanced approach
                    "high": 0.6     # More conservative (for base models with limited RAM)
                }
                memory_fraction = memory_fractions.get(optimization_level, 0.7)
                
                # Adjust based on total memory
                if total_memory_gb <= 8:
                    memory_fraction = min(memory_fraction, 0.7)  # More conservative for 8GB models
                
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # 3. Empty PyTorch CUDA cache (works for MPS too)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            # 4. Set MPS cache size based on available memory and optimization level
            if hasattr(torch.backends.mps, 'set_cache_size_mb'):
                # Cache size percentages based on optimization level
                cache_percentages = {
                    "low": 0.3,     # Can use more cache (30% of free memory)
                    "medium": 0.25, # Balanced (25% of free memory)
                    "high": 0.2     # Conservative (20% of free memory)
                }
                cache_percentage = cache_percentages.get(optimization_level, 0.25)
                
                # Calculate cache size with caps based on total system memory
                if total_memory_gb <= 8:
                    max_cache_mb = 1536  # 1.5GB max for 8GB models
                elif total_memory_gb <= 16:
                    max_cache_mb = 2048  # 2GB max for 16GB models
                else:
                    max_cache_mb = 3072  # 3GB max for 32GB+ models
                
                cache_size_mb = min(int(free_memory_gb * 1024 * cache_percentage), max_cache_mb)
                torch.backends.mps.set_cache_size_mb(cache_size_mb)
                
            # 5. Configure PyTorch to be more memory efficient
            if hasattr(torch, 'set_grad_enabled'):
                # Disable gradient calculation since we're only doing inference
                torch.set_grad_enabled(False)
                
            return True
            
        except Exception as e:
            print(f"Note: Could not determine memory configuration: {e}")
            return False
            
    except Exception as e:
        print(f"Note: Could not apply memory optimizations: {e}")
        return False


def configure_torch_for_mps():
    """Configure PyTorch specifically for optimal MPS performance"""
    try:
        import torch
        
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return False
            
        # Apply MPS-specific PyTorch configurations
        
        # 1. Set default dtype for better performance with MPS
        torch.set_default_dtype(torch.float32)
        
        # 2. Configure autocast for MPS
        # This helps with mixed precision operations
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        
        # 3. Set MPS cache size for better memory management
        if hasattr(torch.backends.mps, 'set_cache_size_mb'):
            # Allocate a reasonable cache size (adjust based on available memory)
            torch.backends.mps.set_cache_size_mb(8192)
        
        return True
    except Exception as e:
        print(f"Note: Could not configure PyTorch for MPS: {e}")
        return False


def identify_script_segments(text):
    """Identify segments of text as Arabic, Urdu, or English"""
    # Character ranges
    arabic_specific = re.compile(r'[ص-ڪةءآأؤإئ]')
    urdu_specific = re.compile(r'[ٹڈڑژگڳڱںھہۂۃۄۆۇۈۉۊۋیےێېۍې]')
    english = re.compile(r'[a-zA-Z]')

    # Count character types
    ar_count = len(arabic_specific.findall(text))
    ur_count = len(urdu_specific.findall(text))
    en_count = len(english.findall(text))

    # Determine dominant script
    if ar_count > ur_count and ar_count > en_count:
        return "ar"
    elif ur_count > ar_count and ur_count > en_count:
        return "ur"
    elif en_count > ar_count and en_count > ur_count:
        return "en"
    else:
        # Default to mixed if no clear winner
        return "mixed"


def phonetic_corrector(text, language):
    """Apply phonetic-based corrections for common confused sounds between Urdu and Arabic"""
    if language == "ur":
        # Handle common Arabic letters mistakenly used in Urdu
        replacements = [
            # Arabic ص/ض/ث to Urdu equivalents س
            (r'ص(?![ء-ي])', 'س'),
            (r'ض(?![ء-ي])', 'ز'),
            (r'ث(?![ء-ي])', 'س'),

            # Arabic ح to Urdu ہ when appropriate
            (r'(?<![ء-ي])ح(?![ء-ي])', 'ہ'),

            # Arabic ذ to Urdu ز when appropriate
            (r'(?<![ء-ي])ذ(?![ء-ي])', 'ز'),

            # Fix Urdu-specific characters
            (r'ه', 'ہ'),  # Use correct Urdu heh
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

    return text


def extract_audio(file_path: str, temp_dir: str, optimize: bool = True) -> str:
    """Extract audio efficiently for transcription with enhanced preprocessing and Apple Silicon optimization"""
    console = Console()

    input_path = Path(file_path)
    file_name = input_path.stem
    temp_audio_path = os.path.join(temp_dir, f"{file_name}.wav")

    # Check if we're on Apple Silicon
    _, _, is_apple_silicon = get_device_info()
    if is_apple_silicon:
        console.print(f"[bold cyan]Optimizing audio extraction for Apple Silicon...[/bold cyan]")

    with console.status(f"[bold green]Extracting and optimizing audio from {input_path.name}..."):
        try:
            # Load audio
            audio = AudioSegment.from_file(file_path)

            # Enhanced audio processing pipeline
            # 1. Convert to mono and set appropriate sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            # 2. Normalize audio levels for better speech recognition
            audio = audio.normalize(headroom=1.0)

            # 3. Apply Apple Silicon specific optimizations if detected
            if is_apple_silicon:
                audio = optimize_audio_for_apple_silicon(audio, optimize)
            # Otherwise apply standard optimizations
            elif optimize and len(audio) > 10000:  # Only for clips > 10 seconds when optimization is enabled
                # Split audio into chunks and process each chunk
                chunk_size = 10000  # 10 seconds
                chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

                processed_chunks = []
                for chunk in chunks:
                    # Simple noise reduction: remove very quiet parts that might be background noise
                    # Using pydub's detect_silence and remove silence capabilities
                    from pydub.silence import detect_silence, split_on_silence

                    # Detect silent parts
                    silent_ranges = detect_silence(chunk, min_silence_len=50, silence_thresh=-40)

                    # If silent ranges found, split and remove them
                    if silent_ranges:
                        # Split on silence and keep only non-silent parts
                        non_silent_parts = split_on_silence(chunk, min_silence_len=50, silence_thresh=-40, keep_silence=100)
                        if non_silent_parts:
                            # Combine non-silent parts
                            chunk = sum(non_silent_parts, AudioSegment.empty())

                    processed_chunks.append(chunk)

                # Recombine chunks
                audio = sum(processed_chunks)

                # 4. Add slight compression to reduce dynamic range
                # This helps with whisper's recognition of quieter speech
                audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)

            # 5. Export with 16-bit precision
            audio = audio.set_sample_width(2)

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
    """Enhanced transcription with better mixed language support and Apple Silicon optimization"""
    console = Console()

    # Get preset configuration
    preset = QUALITY_PRESETS[quality_preset]
    model_size = preset["model_size"]
    beam_size = preset["beam_size"]
    vad_filter = preset["vad_filter"]

    # Determine device if not specified
    if not device or not compute_type:
        device, compute_type, is_apple_silicon = get_device_info()
        
        # Apply Apple Silicon optimizations if detected
        if is_apple_silicon:
            console.print("[bold cyan]Apple Silicon detected! Applying optimizations...[/bold cyan]")
            optimize_for_apple_silicon()
            configure_torch_for_mps()
            
            # Get specific Apple Silicon model information
            apple_silicon_info = detect_apple_silicon_model()
            model_name = apple_silicon_info.get("model", "unknown")
            
            # Display detected model information
            if model_name != "unknown":
                console.print(f"[bold green]Detected {model_name} chip[/bold green]")
                
                # Show more detailed info if available
                if "cores" in apple_silicon_info:
                    console.print(f"[green]CPU cores: {apple_silicon_info['cores']} | Neural Engine cores: {apple_silicon_info['neural_engine_cores']}[/green]")
            
            # Use Apple Silicon specific preset
            if quality_preset == "accurate":
                preset = APPLE_SILICON_PRESET
                model_size = preset["model_size"]
                beam_size = preset["beam_size"]
                console.print("[bold cyan]Using optimized settings for Apple Silicon[/bold cyan]")
    else:
        # Check if manually specified device is MPS
        is_apple_silicon = device == "mps"
        if is_apple_silicon:
            optimize_for_apple_silicon()
            configure_torch_for_mps()
            apple_silicon_info = detect_apple_silicon_model()
        else:
            apple_silicon_info = {}

    # Enhanced configuration for Urdu/Arabic mixed content
    is_mixed_urdu_arabic = language == "ur"

    # Select appropriate prompt
    initial_prompt = DEFAULT_MULTILINGUAL_PROMPT
    if language:
        if is_mixed_urdu_arabic:
            # Use enhanced Urdu prompt or mixed Urdu-Arabic prompt
            initial_prompt = LANGUAGE_PROMPTS.get("ur_enhanced", LANGUAGE_PROMPTS["ur"])
        else:
            initial_prompt = LANGUAGE_PROMPTS.get(language, DEFAULT_MULTILINGUAL_PROMPT)

    console.print(Panel(
        Text.from_markup(f"[bold]Transcribing with Whisper [cyan]{model_size}[/cyan] model...")
    ))
    console.print(f"Using: [cyan]{device}[/cyan] | Quality: [cyan]{quality_preset}[/cyan]")
    
    if is_apple_silicon:
        console.print(f"[bold green]Apple Silicon acceleration enabled[/bold green]")

    if language:
        if is_mixed_urdu_arabic:
            console.print(f"Language: [cyan]Urdu[/cyan] (with mixed Arabic detection enabled)")
        else:
            console.print(f"Language: [cyan]{language}[/cyan] (User selected)")
    else:
        console.print(f"Language: [cyan]Auto-detect[/cyan]")

    # Apply MPS-specific model initialization optimizations
    model_kwargs = {
        "model_size": model_size,
        "device": device,
        "compute_type": compute_type,
        "download_root": os.path.expanduser("~/.cache/whisper")
    }
    
    # Add MPS-specific optimizations
    if is_apple_silicon:
        # Get recommended thread count from Apple Silicon model detection
        recommended_threads = apple_silicon_info.get("recommended_threads", 4)
        
        # Configure model for optimal MPS performance based on detected chip
        model_kwargs.update({
            "cpu_threads": recommended_threads,
            "num_workers": max(1, recommended_threads // 2),  # Scale workers based on threads
        })
        
        # Apply MPS-specific environment variables right before model initialization
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Initialize model with enhanced settings
    model = WhisperModel(**model_kwargs)

    # Optimized parameters for Urdu and Arabic mixed content
    transcription_params = {
        "beam_size": max(beam_size, 15) if is_mixed_urdu_arabic else max(beam_size, 10),
        "language": language if language else None,
        "vad_filter": False,  # Disabled VAD filter to capture full transcription
        "word_timestamps": True,
        "initial_prompt": initial_prompt,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "best_of": 5,
        "condition_on_previous_text": True,
        "no_speech_threshold": 0.05,  # Lowered threshold to capture quieter speech
        "compression_ratio_threshold": 2.4,
        "max_initial_timestamp": 0.8,
        "hallucination_silence_threshold": 0.2
    }

    # Enhanced parameters specifically for Urdu/Arabic mixed content
    if is_mixed_urdu_arabic:
        transcription_params.update({
            "vad_parameters": dict(
                min_silence_duration_ms=50,  # Further reduced for better word boundary detection
                speech_pad_ms=60,  # Increased for better context
                threshold=0.25  # Lower threshold to catch quieter speech
            ),
            "best_of": 7,  # Increased for better candidate selection in mixed content
            "temperature": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Finer temperature sampling
        })
    else:
        # Default VAD parameters for other languages
        transcription_params["vad_parameters"] = dict(
            min_silence_duration_ms=80,
            speech_pad_ms=40,
            threshold=0.35
        )
        
    # Apple Silicon specific transcription optimizations
    if is_apple_silicon:
        # Get recommended batch size from Apple Silicon model detection
        recommended_batch_size = apple_silicon_info.get("recommended_batch_size", 8)
        
        # Adjust parameters for better performance on Apple Silicon based on detected model
        transcription_params.update({
            "beam_size": min(transcription_params["beam_size"], recommended_batch_size),
            "best_of": min(transcription_params.get("best_of", 5), recommended_batch_size - 2),
        })
        
        # Adjust temperature range based on model
        model_name = apple_silicon_info.get("model", "unknown")
        if model_name in ["M2", "M3"]:
            # Newer chips can handle more temperature samples
            transcription_params["temperature"] = [0.0, 0.2, 0.4, 0.6, 0.8]
        else:
            # More conservative for M1
            transcription_params["temperature"] = [0.0, 0.2, 0.4, 0.6]

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
            'language': language or info.language
        })

    # Report language detection info
    detected_language = language or info.language
    language_confidence = round(info.language_probability * 100, 2) if not language else 100.0
    console.print(f"Detected language: [cyan]{detected_language}[/cyan] (Confidence: {language_confidence}%)")

    # Apply enhanced post-processing
    processed_segments = post_process_segments(segments_list, detected_language)

    return processed_segments


def post_process_segments(segments, detected_language):
    """Apply enhanced post-processing to improve transcription quality"""
    processed = []

    # Choose appropriate correction dictionary based on language
    corrections = {}
    if detected_language in ["ar", "ur", "en"]:
        corrections = CORRECTIONS[detected_language]
    else:
        # For mixed content, include all corrections
        for lang_corrections in CORRECTIONS.values():
            corrections.update(lang_corrections)

    # Process each segment
    for segment in segments:
        text = segment["text"].strip()
        original_text = text

        # 1. Apply basic text corrections
        for error, correction in corrections.items():
            text = text.replace(error, correction)

        # 2. Apply phonetic-based corrections
        text = phonetic_corrector(text, detected_language)

        # 3. Islamic terminology corrections
        # First tokenize to handle words properly
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word.lower() in ISLAMIC_TERMS:
                # Replace the word with its correct form while preserving case
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                text = pattern.sub(ISLAMIC_TERMS[word.lower()], text)

        # 4. Script identification and segment-specific processing
        # Split text into sentences for more granular processing
        sentences = re.split(r'([.۔،؟!])', text)
        processed_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                processed_sentences.append(sentence)
                continue

            script = identify_script_segments(sentence)

            # Apply script-specific corrections
            if script == "ar":
                # Arabic-specific corrections
                for ar_error, ar_correction in CORRECTIONS["ar"].items():
                    sentence = sentence.replace(ar_error, ar_correction)
            elif script == "ur":
                # Urdu-specific corrections
                for ur_error, ur_correction in CORRECTIONS["ur"].items():
                    sentence = sentence.replace(ur_error, ur_correction)

            processed_sentences.append(sentence)

        # Reassemble the text
        text = ''.join(processed_sentences)

        # 5. Fix number transcription errors (common in Urdu)
        number_pattern = re.compile(r'\b\d+\b')
        if detected_language == "ur" and number_pattern.search(text):
            # Convert digits to Urdu words where appropriate
            for match in number_pattern.finditer(text):
                num = match.group(0)
                # Only convert small numbers that are likely to be fully pronounced
                if len(num) <= 2 and int(num) <= 30:
                    try:
                        # This would require a proper Urdu number-to-word function
                        # For demonstration, we're using a placeholder
                        text = text.replace(num, f"[{num}]")
                    except:
                        pass

        # Update segment with corrected text
        segment["text"] = text
        # Add original text for reference
        segment["original_text"] = original_text
        processed.append(segment)

    # Enhanced segment merging with timestamp adjustments
    if len(processed) > 1:
        merged = [processed[0]]

        for current in processed[1:]:
            previous = merged[-1]

            # Calculate gap between segments
            gap = current['start'] - previous['end']

            # Get text lengths to prevent overly long segments
            prev_text_length = len(previous['text'])
            curr_text_length = len(current['text'])
            prev_word_count = len(previous['text'].split())

            # Improved merging criteria with better timestamp handling
            # 1. Fix any potential timestamp overlap
            if gap < 0:
                # Calculate middle point between segments as a natural boundary
                midpoint = (previous['end'] + current['start']) / 2

                # Ensure minimum span for comprehension
                min_segment_duration = 0.3  # Minimum segment duration in seconds

                # Adjust previous end time, ensuring minimum duration
                new_prev_end = min(midpoint, current['start'] - 0.01)
                if new_prev_end - previous['start'] >= min_segment_duration:
                    previous['end'] = new_prev_end

                # Adjust current start time, ensuring it's after previous end
                current['start'] = max(previous['end'] + 0.01, midpoint)

            # 2. Only attempt to merge segments if they're very close and meet strict criteria
            if (gap < 0.1 and
                prev_word_count <= 2 and
                prev_text_length < 30 and
                prev_text_length + curr_text_length < 80):

                # Preserve current start time to maintain synchronization
                original_current_start = current['start']

                # Merge segments
                previous['end'] = current['end']
                previous['text'] += " " + current['text']

                # If words info exists, merge it with timestamp adjustments
                if 'words' in previous and 'words' in current:
                    # Adjust timestamps of words in current segment to account for any gap
                    adjustment = original_current_start - previous['end']

                    for word in current['words']:
                        if hasattr(word, 'start'):
                            word.start = max(previous['end'], word.start - adjustment)
                            word.end = max(word.start + 0.01, word.end - adjustment)
                        elif isinstance(word, dict) and 'start' in word:
                            word['start'] = max(previous['end'], word['start'] - adjustment)
                            word['end'] = max(word['start'] + 0.01, word['end'] - adjustment)

                    previous['words'].extend(current['words'])
            else:
                # Process current segment for splitting if needed
                segments_to_add = split_long_segment(current)
                merged.extend(segments_to_add)

        return merged

    return processed


def split_long_segment(segment):
    """Split a segment if it exceeds the maximum word count"""
    MAX_WORDS_PER_SEGMENT = 15  # Reduced from 20 for better readability

    words = segment['text'].split()
    word_count = len(words)

    # If segment is short enough, return it as is
    if word_count <= MAX_WORDS_PER_SEGMENT:
        return [segment]

    # For segments over the limit, split at the limit or nearest punctuation
    target_idx = min(MAX_WORDS_PER_SEGMENT, word_count // 2)

    # Find a good split point near the target index, preferably at punctuation
    split_idx = target_idx
    # Look for punctuation near the target to make a natural break
    for i in range(target_idx - 3, target_idx + 4):
        if i > 0 and i < word_count and any(words[i-1].endswith(p) for p in ['.', '،', '؟', '!', '?', ':', ';']):
            split_idx = i
            break

    # Create two segments from the split
    first_half = ' '.join(words[:split_idx])
    second_half = ' '.join(words[split_idx:])

    # Calculate time distribution more precisely using word timestamps when available
    time_span = segment['end'] - segment['start']

    split_time = None

    # If we have word-level timestamps, use them for more accurate splitting
    if 'words' in segment and segment['words'] and len(segment['words']) > 0:
        # Try to find the exact timestamp of the word at split point
        try:
            # Handle split_idx being at or beyond the available word timestamps
            word_idx = min(split_idx, len(segment['words'])-1)

            if hasattr(segment['words'][word_idx], 'start'):
                split_time = segment['words'][word_idx].start
            elif isinstance(segment['words'][word_idx], dict) and 'start' in segment['words'][word_idx]:
                split_time = segment['words'][word_idx]['start']

            # Ensure split_time is within segment boundaries
            if split_time is not None:
                # Ensure split_time is not too close to start or end
                min_time = segment['start'] + (time_span * 0.1)  # At least 10% into the segment
                max_time = segment['end'] - (time_span * 0.1)    # At most 90% into the segment
                split_time = max(min_time, min(split_time, max_time))

        except (IndexError, AttributeError, KeyError):
            split_time = None

    # Fall back to proportional calculation if no valid split_time was determined
    if split_time is None:
        # More accurate proportional calculation based on character count
        char_ratio = len(' '.join(words[:split_idx])) / len(' '.join(words))
        split_time = segment['start'] + (time_span * char_ratio)

    # Create first segment
    first_segment = segment.copy()
    first_segment['text'] = first_half
    first_segment['end'] = split_time

    # Create second segment
    second_segment = segment.copy()
    second_segment['text'] = second_half
    second_segment['start'] = split_time

    # If words info exists, split it too
    if 'words' in segment and segment['words']:
        # Find word that corresponds to split time
        word_split_idx = 0
        for i, word in enumerate(segment['words']):
            word_start = None

            if hasattr(word, 'start'):
                word_start = word.start
            elif isinstance(word, dict) and 'start' in word:
                word_start = word['start']

            if word_start is not None and word_start >= split_time:
                word_split_idx = i
                break

        # Ensure we don't get an empty words list for either segment
        if word_split_idx == 0:
            word_split_idx = 1
        elif word_split_idx >= len(segment['words']):
            word_split_idx = len(segment['words']) - 1

        first_segment['words'] = segment['words'][:word_split_idx]
        second_segment['words'] = segment['words'][word_split_idx:]

    # Recursively split the second segment if it's still too long
    result = [first_segment]
    result.extend(split_long_segment(second_segment))

    return result


def generate_srt(segments: List[dict], output_path: str) -> None:
    """Generate SRT file from transcription segments"""
    console = Console()

    # Validate and correct timestamps before generating SRT
    validated_segments = []
    for i, segment in enumerate(segments):
        # Ensure start is before end
        if segment['start'] >= segment['end']:
            # Add a small duration if timestamps are equal or inverted
            segment['end'] = segment['start'] + 0.5

        # Ensure no overlap with previous segment
        if i > 0 and segment['start'] < validated_segments[-1]['end']:
            # Adjust start time to match previous end time
            segment['start'] = validated_segments[-1]['end'] + 0.01

            # If this adjustment causes start >= end, fix end time too
            if segment['start'] >= segment['end']:
                segment['end'] = segment['start'] + 0.5

        # Ensure minimum duration for readability
        min_duration = 0.7  # seconds
        if segment['end'] - segment['start'] < min_duration:
            segment['end'] = segment['start'] + min_duration

        validated_segments.append(segment)

    # Generate SRT entries from validated segments
    srt_entries = []
    for i, segment in enumerate(validated_segments, start=1):
        # Convert to proper timedelta objects, ensuring they're valid numbers
        try:
            start_seconds = max(0, float(segment['start']))
            end_seconds = max(start_seconds + 0.1, float(segment['end']))

            start_time = timedelta(seconds=start_seconds)
            end_time = timedelta(seconds=end_seconds)

            entry = srt.Subtitle(
                index=i,
                start=start_time,
                end=end_time,
                content=segment['text']
            )
            srt_entries.append(entry)
        except (ValueError, TypeError) as e:
            console.print(f"[yellow]Warning: Invalid timestamp in segment {i}. Skipping. Error: {e}")
            continue

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

    # Check for Apple Silicon
    _, _, is_apple_silicon = get_device_info()

    # App header with Apple Silicon detection
    header_text = "[bold cyan]Enhanced Whisper Large-v3 Transcription Tool[/bold cyan]\n"
    if is_apple_silicon:
        header_text += "[italic]Optimized for Apple Silicon with MPS acceleration[/italic]\n"
    header_text += "[italic]Optimized for mixed Urdu+Arabic+English audio[/italic]"
    
    console.print(Panel.fit(
        Text.from_markup(header_text),
        border_style="cyan"
    ))

    # If Apple Silicon is detected, show info about it
    if is_apple_silicon:
        console.print("[bold green]Apple Silicon detected![/bold green] Using Metal Performance Shaders (MPS) for acceleration.")
        
        # Apply optimizations early
        if optimize_for_apple_silicon():
            console.print("[bold green]✓[/bold green] Applied system-level optimizations for Apple Silicon")
        if configure_torch_for_mps():
            console.print("[bold green]✓[/bold green] Configured PyTorch for optimal MPS performance")

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
        
        # Add Apple Silicon optimized preset if on Apple Silicon
        if is_apple_silicon:
            quality_options["2"] = ("apple_silicon", "Apple Silicon Optimized - Balanced for M-series chips")

        console.print("Select quality preset:")
        for key, (preset, desc) in quality_options.items():
            console.print(f"[bold cyan]{key}.[/bold cyan] {desc}")

        quality_choice = Prompt.ask("Enter option number", default="1" if not is_apple_silicon else "2")
        quality_preset = quality_options.get(quality_choice, ("accurate", ""))[0]
        
        # If Apple Silicon preset is selected, use the APPLE_SILICON_PRESET
        use_apple_silicon_preset = quality_preset == "apple_silicon"
        if use_apple_silicon_preset:
            quality_preset = "accurate"  # We'll apply the Apple Silicon preset in transcribe_audio

        # Enhanced language selection with mixed option
        language_options = {
            "1": ("ar", "Arabic"),
            "2": ("ur", "Urdu (with Arabic words detection)"),
            "3": ("en", "English"),
            "4": (None, "Auto-detect (let Whisper decide)")
        }

        console.print("Select language:")
        for key, (code, name) in language_options.items():
            console.print(f"[bold cyan]{key}.[/bold cyan] {name}")

        lang_choice = Prompt.ask("Enter option number", default="4")
        language = language_options.get(lang_choice, (None, ""))[0]
        
        # Apple Silicon specific audio optimization option
        optimize_audio = True
        if is_apple_silicon:
            optimize_audio = Confirm.ask(
                "Apply enhanced audio preprocessing for Apple Silicon?", 
                default=True
            )

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
                task = progress.add_task("[bold yellow]Extracting and optimizing audio...", total=None)
                audio_path = extract_audio(file_path, temp_dir, optimize=optimize_audio)
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

            # Final success message with Apple Silicon info if applicable
            success_text = f"[bold green]Transcription complete![/bold green]\n"
            success_text += f"Output saved to: [bold cyan]{output_file}[/bold cyan]"
            
            if is_apple_silicon:
                success_text += f"\n[italic]Processed with Apple Silicon optimizations[/italic]"
                
            rprint(Panel(
                Text.from_markup(success_text),
                title="Success",
                border_style="green"
            ))

    except KeyboardInterrupt:
        console.print("[bold yellow]Process interrupted by user.")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")
        # Print more detailed error info in debug mode
        import traceback
        console.print("[bold yellow]Detailed error information:[/bold yellow]")
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()