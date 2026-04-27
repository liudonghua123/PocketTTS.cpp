#!/usr/bin/env python3
"""
Download built-in voices for PocketTTS.

This script downloads the predefined voices from Hugging Face.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

_ORIGINS_OF_PREDEFINED_VOICES = {
    "cosette": "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "marius": "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    "javert": "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "jean": "hf://kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav",
    "anna": "hf://kyutai/tts-voices/vctk/p228_023_enhanced.wav",
    "vera": "hf://kyutai/tts-voices/vctk/p229_023_enhanced.wav",
    "fantine": "hf://kyutai/tts-voices/vctk/p244_023_enhanced.wav",
    "charles": "hf://kyutai/tts-voices/vctk/p254_023_enhanced.wav",
    "paul": "hf://kyutai/tts-voices/vctk/p259_023_enhanced.wav",
    "eponine": "hf://kyutai/tts-voices/vctk/p262_023_enhanced.wav",
    "azelma": "hf://kyutai/tts-voices/vctk/p303_023_enhanced.wav",
    "george": "hf://kyutai/tts-voices/vctk/p315_023_enhanced.wav",
    "mary": "hf://kyutai/tts-voices/vctk/p333_023_enhanced.wav",
    "jane": "hf://kyutai/tts-voices/vctk/p339_023_enhanced.wav",
    "michael": "hf://kyutai/tts-voices/vctk/p360_023_enhanced.wav",
    "eve": "hf://kyutai/tts-voices/vctk/p361_023_enhanced.wav",
    "bill_boerst": "hf://kyutai/tts-voices/voice-zero/bill_boerst.wav",
    "peter_yearsley": "hf://kyutai/tts-voices/voice-zero/peter_yearsley.wav",
    "stuart_bell": "hf://kyutai/tts-voices/voice-zero/stuart_bell.wav",
    "caro_davy": "hf://kyutai/tts-voices/voice-zero/caro_davy.wav",
}

def download_voice(name: str, url: str, output_dir: Path):
    """Download a single voice file."""
    if not url.startswith("hf://kyutai/tts-voices/"):
        raise ValueError(f"Unexpected URL format: {url}")
    
    filename = url[len("hf://kyutai/tts-voices/"):]
    repo_id = "kyutai/tts-voices"
    
    local_path = output_dir / f"{name}.wav"
    if local_path.exists():
        print(f"  {name}: already exists at {local_path}")
        return
    
    print(f"  Downloading {name}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        force_download=False
    )
    # Rename to name.wav
    os.rename(downloaded_path, local_path)
    print(f"  ✓ {name} saved to {local_path}")

def main():
    parser = argparse.ArgumentParser(description="Download built-in voices for PocketTTS")
    parser.add_argument("--output-dir", default="./voices", help="Output directory for voice files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading voices to {output_dir}")
    print("-" * 40)
    
    for name, url in _ORIGINS_OF_PREDEFINED_VOICES.items():
        try:
            download_voice(name, url, output_dir)
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")
    
    print("Done.")

if __name__ == "__main__":
    main()