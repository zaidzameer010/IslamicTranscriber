# Advanced Audio Transcriber with Whisper v3

This is an advanced audio transcriber that uses OpenAI's Whisper v3 model with CUDA acceleration to transcribe audio files and generate SRT subtitles with accurate timestamps.

## Features

- Uses Whisper Large v3 model for high accuracy
- CUDA acceleration for faster processing
- Generates SRT format output with precise timestamps
- Supports multiple languages (auto-detection)
- Progress bar for transcription status

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- Required Python packages (install using requirements.txt)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python transcriber.py path/to/your/audio.mp3
```

With custom output path:
```bash
python transcriber.py path/to/your/audio.mp3 --output path/to/output.srt
```

With different model:
```bash
python transcriber.py path/to/your/audio.mp3 --model medium
```

## Supported Audio Formats

- MP3
- WAV
- M4A
- and other formats supported by ffmpeg

## Output

The script will generate an SRT file containing:
- Sequential segment numbers
- Timestamp ranges (HH:MM:SS,mmm)
- Transcribed text
- Proper SRT formatting
