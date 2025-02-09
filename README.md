# Advanced Audio Transcriber

An advanced audio transcription tool using OpenAI's Whisper v3 model with support for multiple languages and accurate timestamps.

## Features

- Uses Whisper v3 large model for state-of-the-art transcription
- Supports 20 languages including Arabic and Urdu
- Generates SRT format output with accurate timestamps
- Interactive CLI interface with progress indicators
- GPU acceleration (if available)
- Saves output in the same directory as the source file

## Supported Languages

- Arabic (ar)
- Urdu (ur)
- English (en)
- Hindi (hi)
- Persian (fa)
- Turkish (tr)
- French (fr)
- German (de)
- Spanish (es)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Malay (ms)
- Bengali (bn)
- Indonesian (id)
- Tamil (ta)

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python transcriber.py "path/to/your/audio.mp3"
```

Specify language (optional):
```bash
python transcriber.py "path/to/your/audio.mp3" --language ar
```

Choose model size (optional):
```bash
python transcriber.py "path/to/your/audio.mp3" --model large-v3
```

## Output

The script will create an SRT file in the same directory as the input audio file. The output file will have the same name as the input file but with a `.srt` extension.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Minimum 8GB RAM
- Disk space for model files (~3GB for large-v3)
