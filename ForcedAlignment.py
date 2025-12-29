import stable_whisper
from pathlib import Path

def align_llm_text_to_audio(audio_path, text_path, output_format="srt"):
    print("1. Loading lightweight alignment model...")
    model = stable_whisper.load_model('base')

    print("2. Reading LLM transcript...")
    with open(text_path, 'r', encoding='utf-8') as f:
        # Split by lines and remove empty lines to provide better segment hints
        llm_lines = [line.strip() for line in f if line.strip()]
    # Pass a single string to align() to avoid token/string decoding errors
    llm_text = "\n".join(llm_lines)

    print("3. Aligning text to audio (Forced Alignment)...")
    # Optimized settings for best timestamp accuracy:
    # - token_step=500: Better stability for long segments
    # - suppress_silence=True: Clips timestamps to actual speech
    # - max_word_dur=3.0: Prevents 'stretching' of words over silent gaps
    # - fast_mode=True: Speeds up alignment without sacrificing quality
    result = model.align(
        audio_path, 
        llm_text, 
        language='ur', 
        vad=True, 
        original_split=True, 
        token_step=400,
        suppress_silence=True,
        max_word_dur=3.0,
        fast_mode=True
    )
    
    # Refine the gaps to snap to non-speech boundaries
    result.adjust_gaps()
    
    # OUTPUT OPTIONS
    audio_path = Path(audio_path)

    if output_format == "srt":
        output_file = audio_path.with_suffix(".srt")
        result.to_srt_vtt(str(output_file), word_level=False)
        print(f"Success! Saved to {output_file}")
        
    elif output_format == "json":
        # If you want the raw timestamps in Python to use in code
        data = result.to_dict()
        import json
        output_file = audio_path.with_suffix(".json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Success! Saved to {output_file}")
        
    # Example of how to access timestamps in code:
    if result.segments:
        first_segment = result.segments[0]
        print(f"\nSample Check -> Start: {first_segment.start}, End: {first_segment.end}, Text: {first_segment.text}")
    else:
        print("\nWarning: No segments were produced by alignment.")

# --- RUN THE CODE ---
if __name__ == "__main__":
    # Replace with your actual file names
    AUDIO_FILE = input("Audio path: ") 
    TRANSCRIPT_FILE = input("Transcript txt path: ")
    
    align_llm_text_to_audio(AUDIO_FILE, TRANSCRIPT_FILE, output_format="srt")