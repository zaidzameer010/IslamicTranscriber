import stable_whisper
from pathlib import Path


def align_long_audio(audio_path, text_path, output_format="srt"):
    print("1. Loading Model...")
    # 'large-v3' is best for Urdu, but 'medium' sometimes hallucinates less timestamps.
    # Stick with large-v3 if your hardware supports it.
    model = stable_whisper.load_model('large-v3')

    print("2. Processing LLM Transcript...")
    with open(text_path, 'r', encoding='utf-8') as f:
        llm_text_raw = f.read()

    # Pre-process text to remove difficult chars just for the "matching" phase
    # stable-ts handles the mapping back to original text internally usually,
    # but passing clean text generally reduces drift.
    # If you want to KEEP the original formatting 100%, pass llm_text_raw directly.
    # However, if drift is the issue, ensure the text matches what is actually said.
    
    print("3. Aligning (Deep Mode)...")

    # KEY PARAMETER EXPLANATION FOR LONG FILES:
    # fast_mode=False:    Calculates exact attention weights. Crucial for accuracy.
    # demucs=True:        Runs a separate AI to isolate vocals from noise before aligning.
    #                     This is the #1 fix for "drifting" timestamps.
    # vad=True:           detects silence to reset the timestamp counter.
    # regroup=False:      We disable regrouping initially to get raw precision.
    
    result = model.align(
        audio_path, 
        llm_text_raw, 
        language='ur', 
        token_step=350,
        
        # --- ACCURACY SETTINGS ---
        fast_mode=False,          # DISABLES shortcuts. Slower, but precise.
        vad=True,                 # Voice Activity Detection
        vad_threshold=0.5,        # Sensitivity of silence detection
        
        # --- TIMING SETTINGS ---
        original_split=False,     # Allow model to find natural breaks first (safer than forcing lines)
        min_word_dur=0.1,         # Prevents glitchy micro-timestamps
        
        # --- DRIFT PROTECTION ---
        # If the LLM text is slightly different from audio, these help skip the mismatch
        # rather than crashing the whole timeline.
        gap_padding=" ... ",      # Visual filler for large gaps
    )
    
    # 4. Post-Processing: Re-snap to sentences
    # Since we turned original_split=False for safety, we now regroup the sentences
    # based on punctuation or max duration.
    print("4. Refining Timestamps...")
    result.regroup(min_word_dur=0.3) 
    
    # OUTPUT
    audio_path = Path(audio_path)
    
    output_file = audio_path.with_suffix(".srt")
    result.to_srt_vtt(str(output_file), word_level=False)
    print(f"Success! Saved to {output_file}")
        

if __name__ == "__main__":
    AUDIO = input("Audio file path: ").strip('"') # strips quotes if dragged-dropped
    TRANSCRIPT = input("Transcript file path: ").strip('"')
    
    align_long_audio(AUDIO, TRANSCRIPT)