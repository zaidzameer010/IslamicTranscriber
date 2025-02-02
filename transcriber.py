import torch
import os
from tqdm import tqdm
from faster_whisper import WhisperModel
from pydub import AudioSegment
import warnings
from dataclasses import dataclass
warnings.filterwarnings("ignore")

# Set CUDA memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define comprehensive character sets
ARABIC_DIACRITICS = {
    # Harakat (Short vowels)
    'َ',   # Fatha (zabar) - U+064E
    'ِ',   # Kasra (zer) - U+0650
    'ُ',   # Damma (pesh) - U+064F

    # Tanween (Double vowels)
    'ً',   # Fathatan (double zabar) - U+064B
    'ٍ',   # Kasratan (double zer) - U+064D
    'ٌ',   # Dammatan (double pesh) - U+064C

    # Other diacritical marks
    'ّ',   # Shadda (tashdid) - U+0651
    'ْ',   # Sukun - U+0652
    'ٰ',   # Superscript Alif (Khanjareeya) - U+0670
    'ٓ',   # Maddah - U+0653
    'ٔ',   # Hamza Above - U+0654
    'ٕ',   # Hamza Below - U+0655

    # Additional forms
    'ؘ',   # Small Fatha - U+0618
    'ؙ',   # Small Damma - U+0619
    'ؚ',   # Small Kasra - U+061A
    'ﹶ',   # Fatha Isolated Form - U+FE76
    'ﹷ',   # Fatha Medial Form - U+FE77
    'ﹸ',   # Damma Isolated Form - U+FE78
    'ﹹ',   # Damma Medial Form - U+FE79
    'ﹺ',   # Kasra Isolated Form - U+FE7A
    'ﹻ',   # Kasra Medial Form - U+FE7B
}

URDU_CHARS = {
    # Basic Urdu Letters
    'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ',
    'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص',
    'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل',
    'م', 'ن', 'ں', 'و', 'ہ', 'ھ', 'ء', 'ی', 'ے',

    # Urdu Digits
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',

    # Urdu-specific Characters
    'ۂ', 'ۃ', 'ئ', 'ی', 'ۓ', 'ؤ',

    # Urdu Punctuation and Symbols
    '؟', '،', '۔',

    # Additional Urdu Characters
    'ؐ', 'ؑ', 'ؒ', 'ؓ', 'ؔ', 'ؕ', 'ٖ', 'ٗ', 'ٰ',
}

ARABIC_CHARS = {
    # Basic Arabic Letters
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء',

    # Arabic Numbers
    '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',

    # Arabic Punctuation
    '؟', '،', '؛',

    # Additional Arabic Characters
    'آ', 'إ', 'أ', 'ؤ', 'ئ', 'ة', 'ى',

    # Hamza Variations
    'أ', 'إ', 'ؤ', 'ئ',
}.union(ARABIC_DIACRITICS)  # Add all diacritical marks to Arabic character set

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str

class AudioPreprocessor:
    """Audio preprocessing class for noise reduction and enhancement"""

    @staticmethod
    def extract_audio_from_video(video_path):
        """Extract audio from video file"""
        print("Extracting audio from video...")
        output_path = os.path.splitext(video_path)[0] + '_audio.wav'
        video = AudioSegment.from_file(video_path, format="mp4")
        video.export(output_path, format="wav")
        return output_path

    @staticmethod
    def convert_to_wav(audio_path):
        """Convert audio to WAV format if needed"""
        # Check if it's a video file
        if audio_path.lower().endswith('.mp4'):
            return AudioPreprocessor.extract_audio_from_video(audio_path)

        # Handle audio files
        if not audio_path.lower().endswith('.wav'):
            output_path = os.path.splitext(audio_path)[0] + '.wav'
            audio = AudioSegment.from_file(audio_path)
            audio.export(output_path, format='wav')
            return output_path
        return audio_path

    @staticmethod
    def process_audio(input_file, output_file=None):
        """
        Process audio file and return the path to the processed file
        """
        try:
            # If no output file specified, create one
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + "_processed.wav"

            # Load audio file
            audio = AudioSegment.from_file(input_file)

            # Convert to WAV
            audio.export(output_file, format="wav")

            print(f"Audio processed and saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return input_file

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def create_srt_segment(index, start, end, text):
    """Create an SRT segment with proper formatting"""
    return f"{index}\n{start} --> {end}\n{text}\n\n"

def get_supported_languages():
    """Return a dictionary of supported languages and their codes"""
    return {
        "auto": None,
        "english": "en",
        "chinese": "zh",
        "german": "de",
        "spanish": "es",
        "russian": "ru",
        "korean": "ko",
        "french": "fr",
        "japanese": "ja",
        "portuguese": "pt",
        "turkish": "tr",
        "polish": "pl",
        "arabic": "ar",
        "italian": "it",
        "hindi": "hi",
        "dutch": "nl",
        "ukrainian": "uk",
        "greek": "el",
        "czech": "cs",
        "vietnamese": "vi",
        "urdu": "ur"
    }

def get_available_models():
    """Return a dictionary of available models and their approximate VRAM requirements"""
    return {
        "tiny": "~1GB VRAM",
        "base": "~1GB VRAM",
        "small": "~2GB VRAM",
        "medium": "~5GB VRAM",
        "large-v3": "~10GB VRAM"
    }

def get_compute_type(device):
    """Determine the best compute type based on device capabilities"""
    if device == "cuda":
        if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer
            return "float16"  # Use FP16 for modern GPUs
        return "int8"  # Use INT8 for older GPUs
    return "int8"  # Default to INT8 for CPU

def is_arabic_text(text):
    """Check if text contains Arabic characters but not Urdu-specific characters"""
    # Arabic Unicode ranges - more specific to Arabic
    arabic_ranges = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]

    # Urdu-specific characters and combinations
    urdu_specific = {
        'ٹ', 'ڈ', 'ڑ', 'ں', 'ہ', 'ھ', 'ے', 'ۓ',  # Unique to Urdu
        'پ', 'چ', 'ژ', 'گ',  # Persian characters used in Urdu
        'ی', 'ک',  # Different forms in Urdu
        'ؤ', 'ئ',  # Additional Urdu characters
    }

    # Check if the text contains Arabic script from any of the Arabic ranges
    has_arabic = False
    for start, end in arabic_ranges:
        if any(start <= ord(c) <= end for c in text):
            has_arabic = True
            break

    # Check if the text contains Urdu-specific characters
    has_urdu = any(char in text for char in urdu_specific)

    # Return True only if it has Arabic script but no Urdu-specific characters
    return has_arabic and not has_urdu

def is_urdu_text(text):
    """Check if text contains Urdu characters"""
    # Urdu-specific characters and combinations
    urdu_specific = {
        'ٹ', 'ڈ', 'ڑ', 'ں', 'ہ', 'ھ', 'ے', 'ۓ',  # Unique to Urdu
        'پ', 'چ', 'ژ', 'گ',  # Persian characters used in Urdu
        'ی', 'ک',  # Different forms in Urdu
        'ؤ', 'ئ',  # Additional Urdu characters
    }

    # Urdu Unicode ranges
    urdu_ranges = [
        (0x0600, 0x06FF),  # Arabic/Persian/Urdu basic range
        (0x0750, 0x077F),  # Arabic Supplement (some Urdu use)
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]

    # Check for Urdu-specific characters
    has_urdu_chars = any(char in text for char in urdu_specific)

    # Check for characters in Urdu ranges
    has_urdu_range = False
    for start, end in urdu_ranges:
        if any(start <= ord(c) <= end for c in text):
            has_urdu_range = True
            break

    # Consider it Urdu if it has Urdu-specific chars or uses the Urdu ranges
    # and doesn't match pure Arabic patterns
    return has_urdu_chars or (has_urdu_range and not is_arabic_text(text))

def is_arabic_word(word):
    """Check if a word contains Arabic characters"""
    # Count Arabic-only characters (excluding shared characters with Urdu)
    arabic_only_count = sum(1 for char in word if char in ARABIC_CHARS and char not in URDU_CHARS)
    urdu_count = sum(1 for char in word if char in URDU_CHARS)

    # Check for Arabic diacritics
    has_diacritics = has_arabic_diacritics(word)

    # Consider it Arabic if:
    # 1. It has more Arabic-specific characters than Urdu characters, or
    # 2. It has Arabic diacritical marks
    return arabic_only_count > urdu_count or has_diacritics

def is_urdu_word(word):
    """Check if a word contains Urdu characters"""
    # Count Urdu and Arabic characters
    urdu_count = sum(1 for char in word if char in URDU_CHARS)
    arabic_count = sum(1 for char in word if char in ARABIC_CHARS and char not in URDU_CHARS)

    # Consider it Urdu if it has more Urdu-specific characters
    return urdu_count > arabic_count

def has_arabic_diacritics(word):
    """Check if a word contains Arabic diacritical marks"""
    return any(char in ARABIC_DIACRITICS for char in word)

def initialize_nltk():
    """
    Initialize NLTK by downloading required data
    """
    import nltk
    try:
        # Try punkt_tab first (for newer NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
    except:
        # Fall back to punkt for older versions
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocess text before translation to improve quality
    """
    try:
        import nltk
        import spacy
        from nltk.tokenize import sent_tokenize

        # Initialize NLTK
        initialize_nltk()

        # Load spaCy model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            nlp = spacy.load('en_core_web_sm')

        # Split into sentences for better context
        sentences = sent_tokenize(text)

        processed_sentences = []
        for sentence in sentences:
            # Process with spaCy
            doc = nlp(sentence)

            # Fix common issues
            processed = []
            for token in doc:
                # Handle contractions
                if token.text.lower() in ["'m", "'s", "'re", "'ve", "'ll", "'d"]:
                    if processed:
                        processed[-1] = processed[-1] + token.text
                else:
                    processed.append(token.text)

            # Join tokens back together
            processed_sentence = ' '.join(processed)
            processed_sentences.append(processed_sentence)

        return ' '.join(processed_sentences)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return text

def postprocess_text(text):
    """
    Clean up and improve translated text
    """
    try:
        import re
        import nltk
        from nltk.tokenize import sent_tokenize

        # Initialize NLTK
        initialize_nltk()

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?)])', r'\1', text)
        text = re.sub(r'(\()\s+', r'\1', text)

        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Proper sentence capitalization
        try:
            sentences = sent_tokenize(text)
            sentences = [s.capitalize() for s in sentences]
            text = ' '.join(sentences)
        except Exception as e:
            print(f"Error in sentence tokenization: {str(e)}")
            # Fallback: simple capitalization
            text = '. '.join(s.strip().capitalize() for s in text.split('.'))

        # Fix common translation artifacts
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        text = text.replace('( ', '(')
        text = text.replace(' )', ')')

        # Fix quotes
        text = re.sub(r'"([^"]*)"', r'"\1"', text)

        # Fix ellipsis
        text = text.replace('...', '…')
        text = text.replace('. . .', '…')

        # Fix dashes
        text = text.replace(' - ', ' – ')

        return text.strip()
    except Exception as e:
        print(f"Error in postprocessing: {str(e)}")
        return text

def translate_text(text, source_lang=None):
    """
    Translate text to English while preserving Arabic text using enhanced NLP processing
    """
    try:
        from deep_translator import GoogleTranslator

        # Preprocess non-Arabic text
        words = text.split()
        final_words = []
        current_phrase = []

        # Process each word
        for word in words:
            if is_arabic_word(word):
                # If we have accumulated non-Arabic words, translate them
                if current_phrase:
                    # Preprocess the phrase
                    phrase_text = ' '.join(current_phrase)
                    preprocessed_text = preprocess_text(phrase_text)

                    # Translate
                    translator = GoogleTranslator(source=source_lang, target='en')
                    translated_text = translator.translate(preprocessed_text)

                    if translated_text:
                        # Postprocess the translation
                        translated_text = postprocess_text(translated_text)

                        # Handle sentence case
                        if len(final_words) == 0 or final_words[-1].endswith(('.', '!', '?')):
                            translated_text = translated_text.capitalize()

                        final_words.extend(translated_text.split())
                    current_phrase = []

                # Add the Arabic word directly
                final_words.append(word)
                print(f"Preserving Arabic: {word}")
            else:
                # Accumulate non-Arabic words
                current_phrase.append(word)

        # Handle any remaining phrase
        if current_phrase:
            # Preprocess the phrase
            phrase_text = ' '.join(current_phrase)
            preprocessed_text = preprocess_text(phrase_text)

            # Translate
            translator = GoogleTranslator(source=source_lang, target='en')
            translated_text = translator.translate(preprocessed_text)

            if translated_text:
                # Postprocess the translation
                translated_text = postprocess_text(translated_text)

                # Handle sentence case
                if len(final_words) == 0 or final_words[-1].endswith(('.', '!', '?')):
                    translated_text = translated_text.capitalize()

                final_words.extend(translated_text.split())

        # Join all words and do final cleanup
        result = ' '.join(final_words)
        result = postprocess_text(result)

        return result

    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return text  # Return original text if translation fails

def translate_srt_file(input_srt, output_srt, model, language):
    """
    Translate an SRT file while preserving Arabic text
    """
    print(f"\nTranslating {input_srt} to English (preserving Arabic text)...")

    try:
        # Read the original SRT file
        with open(input_srt, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse SRT content
        segments = []
        current_segment = {"id": "", "time": "", "text": []}
        lines = content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.isdigit():
                if current_segment["text"]:
                    segments.append(current_segment.copy())
                current_segment = {"id": line, "time": "", "text": []}
            elif '-->' in line:
                current_segment["time"] = line
            elif line:
                current_segment["text"].append(line)

        if current_segment["text"]:
            segments.append(current_segment)

        # Translate each segment
        translated_segments = []
        for segment in segments:
            # Join text lines
            text = ' '.join(segment["text"])

            # Translate the text while preserving Arabic
            translated_text = translate_text(text, language)

            # Create translated segment
            translated_segment = segment.copy()
            translated_segment["text"] = [translated_text]
            translated_segments.append(translated_segment)

            # Debug output
            print(f"\nOriginal: {text}")
            print(f"Translated: {translated_text}\n")

        # Write translated SRT
        with open(output_srt, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(translated_segments):
                if i > 0:
                    f.write('\n')
                f.write(f"{segment['id']}\n")
                f.write(f"{segment['time']}\n")
                f.write('\n'.join(segment['text']) + '\n')

        print(f"Translation saved to: {output_srt}")
        return output_srt

    except Exception as e:
        print(f"Error during translation: {str(e)}")
        raise

def load_whisper_model(model_name="large-v3"):
    """
    Load Whisper model with optimized settings for better accuracy
    """
    try:
        # Check CUDA availability
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"

        # Compute type based on available memory
        if use_cuda:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            compute_type = "float16" if free_memory > 6e9 else "int8_float16"
        else:
            compute_type = "int8"  # Use int8 for CPU

        # Load model with optimized settings
        model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type
        )

        print(f"Loaded Whisper model: {model_name} ({compute_type}) on {device}")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {str(e)}")
        raise

def detect_language(model, audio_path):
    """Detects the language of the provided audio file using the Whisper model."""
    try:
        # Run transcription with minimal settings to retrieve language info
        _, info = model.transcribe(
            audio_path,
            language=None,
            initial_prompt="",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False
        )
        if info and hasattr(info, "language"):
            return info.language
        return "en"
    except Exception as e:
        print("Error detecting language:", e)
        return "en"

def build_custom_vocabulary(text):
    """Build a custom vocabulary from previously transcribed text to improve accuracy."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        # Initialize NLTK resources
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Tokenize and get parts of speech
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract likely domain-specific terms (nouns and proper nouns)
        stop_words = set(stopwords.words('english'))
        custom_terms = set()
        
        for word, pos in pos_tags:
            # Focus on nouns (NN*) and unknown words (FW)
            if (pos.startswith('NN') or pos == 'FW') and word.lower() not in stop_words:
                custom_terms.add(word)
        
        return list(custom_terms)
    except Exception as e:
        print(f"Error building custom vocabulary: {str(e)}")
        return []

def apply_context_correction(segments, custom_vocab=None):
    """Apply context-aware corrections to transcribed segments."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        from nltk.corpus import wordnet
        nltk.download('wordnet', quiet=True)
        
        corrected_segments = []
        context_window = []  # Keep track of recent segments for context
        
        for segment in segments:
            # Get the text from the segment
            text = segment.text.strip()
            
            # Apply custom vocabulary corrections
            if custom_vocab:
                for term in custom_vocab:
                    # Use string similarity to catch near-matches
                    words = text.split()
                    for i, word in enumerate(words):
                        if (word.lower() != term.lower() and 
                            nltk.edit_distance(word.lower(), term.lower()) <= 2):
                            words[i] = term
                    text = ' '.join(words)
            
            # Apply context-based corrections
            sentences = sent_tokenize(text)
            corrected_sentences = []
            
            for sentence in sentences:
                # Check for semantic consistency with recent context
                if context_window:
                    # Simple context check - ensure proper sentence connections
                    if not sentence[0].isupper():
                        sentence = sentence[0].upper() + sentence[1:]
                    
                    # Ensure proper punctuation between context
                    if not context_window[-1].endswith(('.', '!', '?')):
                        context_window[-1] += '.'
                
                corrected_sentences.append(sentence)
                context_window.append(sentence)
                if len(context_window) > 3:  # Keep last 3 sentences for context
                    context_window.pop(0)
            
            # Create new segment with corrected text
            corrected_text = ' '.join(corrected_sentences)
            corrected_segments.append(type(segment)(
                start=segment.start,
                end=segment.end,
                text=corrected_text
            ))
        
        return corrected_segments
    except Exception as e:
        print(f"Error in context correction: {str(e)}")
        return segments

def transcribe_with_whisper(model, audio_path, language=None):
    """
    Transcribe audio with optimized Whisper settings using sentence-level processing
    """
    try:
        # Create an initial prompt to guide transcription
        initial_prompt = (
            "The following is a high-quality transcription with proper punctuation and capitalization. "
            "Each sentence is complete and grammatically correct. "
            "Numbers and proper nouns are accurately transcribed. "
            "Technical terms and domain-specific vocabulary are preserved."
        )

        # Optimize transcription settings for sentence-level processing
        segments, info = model.transcribe(
            audio_path,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=15,  # Increased for better accuracy
            temperature=0.0,  # Reduce randomness
            condition_on_previous_text=True,  # Enable context awareness
            word_timestamps=True,  # Enable for better alignment
            vad_filter=True,  # Enable VAD for better sentence segmentation
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
                threshold=0.45,
            ),
            # Advanced parameters for better quality
            compression_ratio_threshold=2.0,
            log_prob_threshold=-0.7,
            no_speech_threshold=0.5,
        )

        # Convert generator to list
        segments_list = list(segments)

        # Build custom vocabulary from initial transcription
        all_text = " ".join(segment.text for segment in segments_list)
        custom_vocab = build_custom_vocabulary(all_text)

        # Apply context-aware corrections
        corrected_segments = apply_context_correction(segments_list, custom_vocab)

        # Print detected language info
        if info and hasattr(info, 'language'):
            print(f"\nDetected language: {info.language}")
            if hasattr(info, 'language_probability'):
                print(f"Language probability: {info.language_probability:.2f}")

        return corrected_segments
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise

def write_srt(segments, output_path):
    """
    Write segments to SRT file with sentence-level formatting
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                # Convert timestamps to SRT format
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)

                # Write segment with proper sentence formatting
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")

                # Clean and format the text as a proper sentence
                text = segment.text.strip()
                if text:
                    # Ensure sentence starts with capital letter
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

                    # Ensure sentence ends with proper punctuation
                    if not text[-1] in ['.', '!', '?']:
                        text += '.'

                f.write(f"{text}\n\n")

        print(f"Transcript saved to: {output_path}")
    except Exception as e:
        print(f"Error writing SRT file: {str(e)}")
        raise

def transcribe_audio(audio_path, output_path=None, model_name="large-v3", language=None, translate_to_english=False):
    try:
        print("Preprocessing audio...")
        preprocessor = AudioPreprocessor()

        # Convert to WAV if needed
        wav_path = preprocessor.convert_to_wav(audio_path)

        # Process audio
        processed_path = preprocessor.process_audio(wav_path)

        # Load optimized Whisper model
        print("\nLoading Whisper model...")
        model = load_whisper_model(model_name)

        # Determine language if not specified
        if not language:
            print("Detecting language...")
            language = detect_language(model, processed_path)
            print(f"Detected language: {language}")

        # Transcribe with optimized settings
        print("\nTranscribing audio...")
        segments = transcribe_with_whisper(model, processed_path, language)

        # Write SRT file
        print("\nWriting transcript to SRT file...")
        write_srt(segments, output_path)

        # Translate if requested
        if translate_to_english and language != "en":
            print("\nTranslating to English...")
            translate_srt_file(output_path,
                             output_path.replace(".srt", "_en.srt"),
                             model,
                             language)

        print("\nTranscription completed!")

        # Cleanup temporary files
        if wav_path != audio_path:
            os.remove(wav_path)

        return output_path

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise

def debug_text_type(text):
    """Helper function to debug text type detection"""
    is_arabic = is_arabic_text(text)
    is_urdu = is_urdu_text(text)
    chars = [f"{c} (U+{ord(c):04X})" for c in text]
    return f"Text: {text}\nChars: {', '.join(chars)}\nIs Arabic: {is_arabic}\nIs Urdu: {is_urdu}"

def transcribe_segment(model, audio_path, segment_text, start_time, end_time, language=None, task="transcribe"):
    """Transcribe or translate a specific segment of audio"""
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)

        # Extract the segment (convert times from seconds to milliseconds)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        segment_audio = audio[start_ms:end_ms]

        # Export segment to temporary file
        temp_path = f"temp_segment_{start_ms}_{end_ms}.wav"
        segment_audio.export(temp_path, format="wav")

        try:
            # Transcribe or translate the segment
            segments, _ = model.transcribe(
                temp_path,
                language=language,
                initial_prompt="The following is a high-quality transcription with proper punctuation and capitalization.",
                beam_size=10,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True
            )

            # Get the transcribed/translated text
            segments_list = list(segments)
            if segments_list:
                return segments_list[0].text

        finally:
            # Always clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return segment_text  # Return original if translation failed

    except Exception as e:
        print(f"Warning: Failed to process segment: {str(e)}")
        return segment_text  # Return original text if processing failed

def save_srt(segments_list, output_path, diarization=None):
    """Save segments to SRT file with optional speaker diarization"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments_list, 1):
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                text = segment.text.strip()

                # Add speaker information if diarization is enabled
                if diarization:
                    speaker = get_speaker_for_segment(diarization, segment.start, segment.end)
                    if speaker:
                        text = f"[{speaker}] {text}"

                f.write(create_srt_segment(i, start_time, end_time, text))

        return output_path

    except Exception as e:
        print(f"Error saving SRT file: {str(e)}")
        raise

def contextual_error_correction(text):
    """
    Perform a second pass on the text to correct ambiguous phrases and common transcription errors.
    """
    try:
        # Basic normalization and punctuation fix
        corrected = " ".join(text.split())
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]
        if corrected and corrected[-1] not in '.!?':
            corrected += '.'
        return corrected
    except Exception as e:
        print("Error in contextual_error_correction:", e)
        return text

def segment_audio_file(audio_file_path, min_silence_len=500, silence_thresh=-40, keep_silence=200):
    """
    Segment the audio file into manageable parts using silence detection.
    Returns a list of dictionaries with 'start' (in seconds), 'end' (in seconds) and 'audio_segment' (AudioSegment object).
    """
    try:
        from pydub import AudioSegment, silence
        audio = AudioSegment.from_file(audio_file_path)
        silence_ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=1)
        segments = []
        if not silence_ranges:
            segments.append({'start': 0, 'end': len(audio) / 1000.0, 'audio_segment': audio})
        else:
            if silence_ranges[0][0] > 0:
                seg = audio[0:silence_ranges[0][0]]
                segments.append({'start': 0, 'end': silence_ranges[0][0] / 1000.0, 'audio_segment': seg})
            for i in range(len(silence_ranges) - 1):
                start = silence_ranges[i][1]
                end = silence_ranges[i+1][0]
                seg = audio[start:end]
                segments.append({'start': start / 1000.0, 'end': end / 1000.0, 'audio_segment': seg})
            last_end = silence_ranges[-1][1]
            if last_end < len(audio):
                seg = audio[last_end:]
                segments.append({'start': last_end / 1000.0, 'end': len(audio) / 1000.0, 'audio_segment': seg})
        return segments
    except Exception as e:
        print("Error during audio segmentation:", e)
        return []

if __name__ == "__main__":
    print("Advanced Audio Transcriber with Whisper v3")
    print("-----------------------------------------")

    # Get audio path from user input
    audio_path = input("Enter the path to your audio file: ").strip('"').strip("'")

    # Validate that the file exists
    if not os.path.exists(audio_path):
        print(f"Error: The file {audio_path} does not exist!")
        exit(1)

    # Show available models
    models = get_available_models()
    print("\nAvailable models (with VRAM requirements):")
    print("----------------------------------------")
    for model, vram in models.items():
        print(f"- {model}: {vram}")

    # Get model selection
    model_input = input("\nEnter model name (default: large-v3): ").lower().strip()
    if not model_input:
        model_input = "large-v3"
    elif model_input not in models:
        print(f"Error: Invalid model '{model_input}'. Using large-v3 instead.")
        model_input = "large-v3"

    # Show language options
    languages = get_supported_languages()
    print("\nSupported languages:")
    print("------------------")
    for lang in languages.keys():
        print(f"- {lang}")

    # Get language selection
    lang_input = input("\nEnter language (or 'auto' for automatic detection): ").lower().strip()
    selected_lang = languages.get(lang_input)

    if lang_input != "auto" and selected_lang is None:
        print(f"Error: Unsupported language '{lang_input}'. Using auto-detection instead.")
        selected_lang = None

    # Get translation preference
    translate = input("\nTranslate to English (preserving Arabic text)? (y/n, default: n): ").lower().strip() == 'y'

    # Generate output path in the same location as the source
    base_name = os.path.splitext(audio_path)[0]
    suffix = "_translated" if translate else ""
    output_path = f"{base_name}{suffix}.srt"

    print(f"\nTranscribing: {audio_path}")
    print(f"Language: {lang_input}")
    print(f"Model: {model_input}")
    print(f"Translation to English: {'enabled' if translate else 'disabled'}")
    print(f"Output will be saved to: {output_path}\n")

    try:
        transcribe_audio(
            audio_path,
            output_path,
            model_input,
            selected_lang,
            translate
        )
    except Exception as e:
        print(f"\nTranscription failed: {str(e)}")
        exit(1)