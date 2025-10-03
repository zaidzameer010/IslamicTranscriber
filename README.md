# Advanced Audio Transcriber

An advanced audio transcription tool powered by Faster-Whisper (OpenAI Whisper v3 via CTranslate2) with multi-language support, accurate timestamps, and enhanced post-processing.

## Features

- Faster-Whisper large-v3 model for state-of-the-art transcription (CTranslate2 backend)
- Interactive CLI: pick audio from current folder and choose language (or auto-detect)
- Robust audio pre-processing: denoise + normalization (librosa + noisereduce) with optional HPF/LPF, preemphasis, and basic dereverb (Wiener)
- Silero VAD filtering with tunable parameters for cleaner segments
- Word-level timestamps with punctuation-aware segmentation and RTL-friendly SRT (Urdu/Arabic)
- Outputs: SRT, WebVTT (.vtt), and JSON with per-word confidences
- Domain biasing: initial prompts and hotwords support
- Auto language detection option with thresholds
- GPU acceleration (float16 on CUDA). CPU uses int8 quantization for memory/speed
- Optional grammar enhancement step using Gemini AI to fix punctuation/grammar in SRT
- Optional two-pass re-decode for low-confidence segments to improve accuracy
- Run config snapshot (.run.json) for reproducibility of parameters per run

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
3. (Optional) For GPU: install the matching CUDA PyTorch wheel (see comment in requirements.txt)

## Usage

Basic usage (interactive):
```bash
python transcribe.py
```

During the run you'll be prompted to:
- Select an audio/video file from the current directory (mp3, wav, flac, mp4, m4a, aac, ogg)
- Choose language (Urdu, Arabic, English) or Auto-detect

Outputs are written next to the selected file as `.srt`, `.vtt`, and `.json`.

Grammar enhancement (optional):
```bash
python grammar.py
```
This reads a selected `.srt`, sends just the text to Gemini for correction, and writes `<name>_enhanced.srt`.

## Advanced Pre-processing (optional)

The pipeline always loads mono 16 kHz audio, applies optional noise reduction and normalization. You can further tune pre-processing in `CONFIG["preprocess"]` inside `transcribe.py`:

```python
"preprocess": {
  "denoise": True,
  "denoise_prop_decrease": 0.8,
  "normalize": True,
  "highpass_hz": 50.0,   # remove low-frequency rumble
  "lowpass_hz": null,    # e.g., 7800.0 to suppress hiss (null means disabled)
  "preemphasis": False,
  "preemphasis_coef": 0.97,
  "dereverb": False,     # basic Wiener-based dereverb
}
```

Notes:
- HPF/LPF and dereverb rely on SciPy (installed via `requirements.txt`).
- If `librosa` cannot load an input (some MP4s), the tool falls back to raw file input for Faster-Whisper.

## Two-pass Low-confidence Re-decode (optional)

For segments with low confidence, you can enable a second pass with higher beam size and a temperature schedule to improve accuracy:

```python
"two_pass": {
  "enabled": true,
  "min_mean_word_prob": 0.6,
  "max_avg_logprob": -1.25,
  "beam_size": 12,
  "best_of": 5,
  "temperature_list": [0.0, 0.2, 0.4, 0.6],
}
```

How it works:
- Computes mean per-word probability if available; otherwise uses `avg_logprob`.
- If below the threshold, the segment is sliced from waveform and re-decoded with more exhaustive settings.

## Reproducibility Snapshot

Each run writes a companion `<name>.run.json` next to outputs (controlled by `CONFIG["emit_run_config"]`). It captures:
- Model/device/compute_type
- VAD and decoding parameters (as actually passed to Faster-Whisper)
- Preprocess and segmentation configs
- Language selection and input file name

## Output

- SRT subtitles with accurate, word-informed timestamps (punctuation-aware splitting)
- WebVTT (.vtt) for web players
- JSON with segments and word confidences
- Enhanced SRT (`*_enhanced.srt`) if you run the grammar step

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing). On CPU, int8 quantization is used
- Minimum 8GB RAM
- Disk space for model files (~3GB for large-v3)

### Environment for Grammar Enhancement
Create a `.env` with:
```
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-pro
