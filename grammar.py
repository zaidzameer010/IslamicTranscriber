import os
import json
import httpx
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Generator, Tuple, Optional
from rich.console import Console
from rich.prompt import Prompt
from rich import print as rprint

# Language detection constants
ARABIC_CHARS = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي')
URDU_CHARS = set('ٹپچڈڑژکگںھہےۓ')
ENGLISH_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Corrections dictionary for common errors
CORRECTIONS = {
    "ar": {
        # Arabic corrections
        "،،": "،",
        "...": "…",
        "؟؟": "؟"
    },
    "ur": {
        # Urdu corrections
        "،،": "،",
        "...": "…",
        "؟؟": "؟"
    },
    "en": {
        # English corrections
        ",,": ",",
        "??": "?",
        "!!": "!"
    }
}

# Initialize Rich console for better UI
console = Console()

def detect_language_context(text: str) -> str:
    """Detect the primary language of a text segment based on character frequency."""
    if not text or text.isspace():
        return "unknown"
        
    # Count characters by language
    ar_count = sum(1 for char in text if char in ARABIC_CHARS)
    ur_count = sum(1 for char in text if char in URDU_CHARS)
    en_count = sum(1 for char in text if char in ENGLISH_CHARS)
    
    # Determine dominant language
    total_chars = len([c for c in text if not c.isspace()])
    if total_chars == 0:
        return "unknown"
        
    ar_percent = ar_count / total_chars
    ur_percent = ur_count / total_chars
    en_percent = en_count / total_chars
    
    # Apply thresholds for language detection
    if ar_percent > 0.5:
        return "ar"
    elif ur_percent > 0.3:  # Lower threshold for Urdu as it may mix with Arabic
        return "ur"
    elif en_percent > 0.5:
        return "en"
    elif ar_percent + ur_percent > 0.5:  # Mixed Arabic/Urdu
        return "ar" if ar_percent > ur_percent else "ur"
    else:
        return "unknown"

def apply_context_aware_corrections(text: str, language: str) -> str:
    """Apply language-specific corrections based on context."""
    if language == "unknown":
        return text
        
    # Apply language-specific punctuation fixes
    if language == "ar" or language == "ur":
        # Convert English punctuation to Arabic/Urdu style
        text = text.replace("?", "؟")
        text = text.replace(";", "؛")
        text = text.replace(",", "،")
    elif language == "en":
        # Ensure English punctuation for English text
        text = text.replace("؟", "?")
        text = text.replace("؛", ";")
        text = text.replace("،", ",")
    
    return text

class SRTGrammarCorrector:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY","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")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/zaidzameer010/IslamicTranscriber",
            "Content-Type": "application/json"
        }
        
        # Enhanced system prompt for the grammar correction with emphasis on context
        self.system_prompt = """You are an expert in Islamic content transcription correction. Your task is to review and correct the ENTIRE SRT file text as a cohesive document, focusing on proper spelling, grammar, and formatting for Islamic content in Arabic and English.

IMPORTANT: First read and understand the ENTIRE transcript to grasp the full context, topic, and terminology before making any corrections. Many blocks are part of the same sentence or thought that spans across multiple timestamps.

Focus on:
1. Maintaining accuracy of Islamic terms, names, and concepts in both languages
2. Preserving diacritical marks in Arabic text (tashkeel/harakat)
3. Correcting severely misspelled words, nonsensical punctuation, and grammar errors while respecting language-specific rules
4. Ensuring proper transliteration of Arabic/Islamic terms in English text
5. Maintaining technical terminology if present
6. Ensuring consistent formatting and directionality (RTL for Arabic, LTR for English)
7. Preserving all SRT timestamp lines exactly as they appear - DO NOT modify timestamp lines in any way
8. Proper capitalization of Islamic terms and names in English text
9. Maintaining consistency of terminology, names, and concepts across the entire transcript
10. If you detect missing words that result in incomplete or fragmented sentences, insert the appropriate words to complete the sentence while preserving the intended meaning.

Language-specific guidelines:
Arabic: 
- Preserve all diacritical marks, ensure proper letter connections and native Arabic script
- Use Arabic punctuation marks (like ، and ؛) for Arabic text
- Maintain right-to-left formatting for Arabic text

English: 
- Use standard Islamic English transliteration conventions
- Use English punctuation marks for English text
- Maintain left-to-right formatting for English text

Mixed language content:
- Apply the appropriate punctuation style based on the primary language of each sentence
- For sentences with mixed languages, use punctuation that matches the dominant language

CRITICAL: Understand that many blocks are part of continuous speech. A sentence may start in one block and continue in the next. Always check surrounding blocks for context before correcting.

Note: SRT timestamp lines (e.g. "00:00:20,000 --> 00:00:23,000") must remain COMPLETELY UNTOUCHED"""

    def find_srt_files(self, directory: str = ".") -> List[Path]:
        """Find all .srt files in the given directory."""
        return list(Path(directory).glob("**/*.srt"))

    def read_srt_file(self, file_path: Path) -> str:
        """Read the contents of an SRT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            encodings = ['cp1252', 'iso-8859-1', 'utf-16']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not read file {file_path} with any supported encoding")

    def write_srt_file(self, file_path: Path, content: str) -> None:
        """Write the corrected content back to the SRT file."""
        backup_path = file_path.with_suffix('.srt.backup')
        # Create backup of original file
        if not backup_path.exists():
            file_path.rename(backup_path)
        
        # Write new content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def parse_srt_content(self, text: str) -> List[Dict]:
        """Parse SRT content into blocks with timestamps and text."""
        # Regex pattern to match SRT blocks (index, timestamp, and subtitle text)
        pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=\n\s*\n\s*\d+\s+\d{2}:|$)'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        srt_blocks = []
        for index, timestamp, content in matches:
            srt_blocks.append({
                'index': index.strip(),
                'timestamp': timestamp.strip(),
                'content': content.strip()
            })
        return srt_blocks
    
    def reconstruct_srt(self, blocks: List[Dict]) -> str:
        """Reconstruct SRT file from parsed blocks."""
        result = []
        for block in blocks:
            result.append(f"{block['index']}\n{block['timestamp']}\n{block['content']}\n")
        return "\n".join(result)
    
    def analyze_transcript_context(self, blocks: List[Dict]) -> Dict:
        """Analyze the transcript to extract context information."""
        # Extract all text content for language detection
        all_text = " ".join([block['content'] for block in blocks])
        primary_language = detect_language_context(all_text)
        
        # Extract potential topics and key terms
        # This is a simple implementation - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top frequent words as potential key terms
        key_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "primary_language": primary_language,
            "total_blocks": len(blocks),
            "key_terms": [term for term, _ in key_terms],
            "text_sample": all_text[:500]  # First 500 chars as sample
        }
    
    def group_blocks_by_context(self, blocks: List[Dict], max_group_size: int = 10) -> List[List[Dict]]:
        """Group blocks into contextual chunks for better processing."""
        # This helps maintain context across blocks while keeping chunks manageable
        grouped_blocks = []
        current_group = []
        
        for i, block in enumerate(blocks):
            current_group.append(block)
            
            # Start a new group when max size reached or at natural breaks in content
            if (len(current_group) >= max_group_size or 
                i == len(blocks) - 1 or  # Last block
                (i > 0 and i < len(blocks) - 1 and 
                 blocks[i]['content'].endswith(('.', '?', '!', '。', '؟', '！')) and
                 not blocks[i+1]['content'][0].islower())):  # Natural sentence break
                
                grouped_blocks.append(current_group)
                current_group = []
        
        # Add any remaining blocks
        if current_group:
            grouped_blocks.append(current_group)
            
        return grouped_blocks
    
    def prepare_context_for_api(self, blocks: List[Dict], transcript_context: Dict) -> str:
        """Prepare the full context for the API with metadata."""
        # Extract all subtitle content with block numbers
        subtitle_texts = []
        for block in blocks:
            subtitle_texts.append(f"BLOCK {block['index']}:\n{block['content']}")
        
        # Join subtitle texts with clear separators
        subtitle_content = "\n\n---\n\n".join(subtitle_texts)
        
        # Add context information
        context_info = f"""
TRANSCRIPT CONTEXT:
Primary Language: {transcript_context['primary_language']}
Total Blocks: {transcript_context['total_blocks']}
Key Terms: {', '.join(transcript_context['key_terms'])}
Content Sample: {transcript_context['text_sample']}

FULL TRANSCRIPT CONTENT:
{subtitle_content}
"""
        return context_info
    
    async def correct_grammar_with_context(self, blocks: List[Dict], transcript_context: Dict) -> Generator[str, None, None]:
        """Send text to OpenRouter API for grammar correction with full context awareness."""
        # Prepare the full context for the API
        context_info = self.prepare_context_for_api(blocks, transcript_context)
        
        # Send to API with enhanced context
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "google/gemini-2.0-pro-exp-02-05:free",  # You can change the model as needed
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Please correct the subtitle text in the following transcript. First read and understand the ENTIRE transcript to grasp the full context before making corrections. Many blocks are part of continuous speech that spans multiple timestamps. DO NOT modify block numbers or timestamps:\n\n{context_info}"}
                    ],
                    "stream": True,
                    "temperature": 0.2,  # Lower temperature for more consistent corrections
                    "max_tokens": 507904   # Adjust based on your needs
                },
                timeout=None
            )
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"]:
                            content = data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    def parse_corrected_content(self, corrected_text: str, original_blocks: List[Dict]) -> List[Dict]:
        """Parse the corrected content and update only the subtitle text in original blocks."""
        # Pattern to extract corrected blocks
        pattern = r'BLOCK (\d+):\s*([\s\S]*?)(?=\n\nBLOCK \d+:|$)'
        corrected_blocks = re.findall(pattern, corrected_text, re.MULTILINE)
        
        # Create a mapping of block index to corrected content
        corrections = {}
        for index, content in corrected_blocks:
            corrections[index.strip()] = content.strip()
        
        # Update original blocks with corrected content
        for block in original_blocks:
            if block['index'] in corrections:
                block['content'] = corrections[block['index']]
        
        return original_blocks
    
    async def process_file(self, file_path: Path) -> None:
        """Process a single SRT file with enhanced context awareness."""
        console.print(f"\n[bold blue]Processing file:[/bold blue] {file_path}")
        
        # Read the file
        original_text = self.read_srt_file(file_path)
        
        # Parse SRT content into blocks
        original_blocks = self.parse_srt_content(original_text)
        
        if not original_blocks:
            console.print("[bold red]No valid SRT blocks found in the file![/bold red]")
            return
            
        # Analyze transcript for context
        console.print("[bold green]Analyzing transcript context...[/bold green]")
        transcript_context = self.analyze_transcript_context(original_blocks)
        
        # Display detected context
        console.print(f"[bold]Detected primary language:[/bold] {transcript_context['primary_language']}")
        console.print(f"[bold]Total blocks:[/bold] {transcript_context['total_blocks']}")
        console.print(f"[bold]Key terms detected:[/bold] {', '.join(transcript_context['key_terms'])}")
        
        # Process the text
        console.print("[bold green]Correcting grammar and formatting with full context awareness...[/bold green]")
        corrected_content = ""
        
        # Create a temporary file for streaming updates
        temp_file = file_path.with_suffix('.srt.temp')
        backup_path = file_path.with_suffix('.srt.backup')
        
        # Create backup if it doesn't exist
        if not backup_path.exists():
            file_path.rename(backup_path)
            
        try:
            # Collect all corrected content
            async for chunk in self.correct_grammar_with_context(original_blocks, transcript_context):
                corrected_content += chunk
                console.print(chunk, end="", style="cyan")
            
            # Parse corrected content and update original blocks
            updated_blocks = self.parse_corrected_content(corrected_content, original_blocks)
            
            # Reconstruct SRT file
            final_srt = self.reconstruct_srt(updated_blocks)
            
            # Write the final SRT file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(final_srt)
            
            # Move the temporary file to the actual file
            temp_file.replace(file_path)
            
        except Exception as e:
            console.print(f"\n[bold red]Error during processing:[/bold red] {str(e)}")
            # Restore from backup if there's an error
            if backup_path.exists():
                backup_path.replace(file_path)
            raise
        finally:
            # Clean up temporary file if it exists
            if temp_file.exists():
                temp_file.unlink()
        
        console.print(f"\n[bold green]✓[/bold green] File processed and saved: {file_path}")
        console.print(f"[dim]Original file backed up as: {backup_path}[/dim]")

async def main():
    try:
        corrector = SRTGrammarCorrector()
        
        # Find all SRT files
        srt_files = corrector.find_srt_files()
        if not srt_files:
            console.print("[bold red]No SRT files found in the current directory![/bold red]")
            return

        # Display available files
        console.print("\n[bold]Available SRT files:[/bold]")
        for i, file in enumerate(srt_files, 1):
            console.print(f"{i}. {file}")

        # Get user selection
        selection = Prompt.ask(
            "\nSelect a file to process (enter number)",
            choices=[str(i) for i in range(1, len(srt_files) + 1)]
        )
        
        selected_file = srt_files[int(selection) - 1]
        await corrector.process_file(selected_file)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())