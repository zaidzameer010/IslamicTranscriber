from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.prompt import Prompt
from google.genai import types

# Load environment variables
load_dotenv(override=True)

# Initialize Gemini AI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# System prompt for grammar correction
SYSTEM_PROMPT = (
    "You are a grammar corrector. "
    "Fix the grammar, add proper punctuation (in RTL order for urdu and arabic), remove word slips and misspellings. "
    "Do not include any timestamps, indices, or file formatting in your output. "
    "Return only the corrected text segments with same block structure as the input in the same order, separated by blank lines."
)

def main():
    console = Console()
    # Discover .srt files in current directory
    srt_files = sorted([f for f in Path.cwd().iterdir() if f.is_file() and f.suffix.lower() == ".srt"])
    if not srt_files:
        console.print("[bold red]No .srt files found in current directory[/]")
        return
    console.print("[bold cyan]Available SRT files:[/]")
    for idx, f in enumerate(srt_files, start=1):
        console.print(f"[green]{idx}[/] {f.name}")
    choice = Prompt.ask("Select SRT file number", choices=[str(i) for i in range(1, len(srt_files)+1)])
    input_path = srt_files[int(choice) - 1]
    console.log(f"[bold magenta]Selected {input_path.name}")

    # Read input SRT content
    srt_text = input_path.read_text(encoding="utf-8")

    # Parse SRT blocks to separate text and timestamps
    blocks = [b for b in srt_text.strip().split("\n\n") if b.strip()]
    entries = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        idx = lines[0]
        timestamp = lines[1]
        text = "\n".join(lines[2:])
        entries.append({"index": idx, "timestamp": timestamp, "text": text})

    # Prepare text-only prompt for model
    texts = [entry["text"] for entry in entries]
    text_prompt = "\n\n".join(texts)
    prompt = SYSTEM_PROMPT + "\n\n" + text_prompt

    console.rule("[bold green]Sending text segments to Gemini for grammar correction")
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"),
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    corrected_segments = response.text.strip().split("\n\n")

    # Reconstruct SRT with corrected text
    new_blocks = []
    for i, entry in enumerate(entries):
        corrected = corrected_segments[i] if i < len(corrected_segments) else entry["text"]
        new_block = f"{entry['index']}\n{entry['timestamp']}\n{corrected}"
        new_blocks.append(new_block)
    new_srt = "\n\n".join(new_blocks) + "\n"

    # Write output file with suffix '_enhanced'
    output_path = input_path.with_name(input_path.stem + "_enhanced.srt")
    output_path.write_text(new_srt, encoding="utf-8")
    console.log(f"[bold blue]Enhanced SRT written to {output_path}")

if __name__ == "__main__":
    main()