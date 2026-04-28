#!/usr/bin/env python3
"""
Download MOSS-TTS-Nano models from HuggingFace.

This script downloads the required ONNX models and tokenizer from:
- https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX
- https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX
"""

import os
import sys
import argparse
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress indication."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        req = Request(url, headers={'User-Agent': 'MOSS-TTS-Nano-Downloader/1.0'})
        with urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r{destination.name}: [{bar}] {percent:.1f}%", end='', flush=True)
            
            print()  # newline after progress bar
            return True
    except (URLError, HTTPError, OSError) as e:
        print(f"\n✗ Failed to download {url}: {e}")
        return False

def get_huggingface_file_url(repo_id: str, filename: str, use_cdn: bool = True) -> str:
    """Get direct download URL for a file from HuggingFace."""
    if use_cdn:
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
    else:
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

def download_models(models_dir: Path = Path("models"), verbose: bool = False) -> bool:
    """Download all required models."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model mapping: expected name -> HuggingFace repo + filename
    # Supports multiple repository structures
    models_to_download = {
        # Standard MOSS-TTS-Nano naming
        "text_encoder.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "text_encoder.onnx"),
        "audio_encoder.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "audio_encoder.onnx"),
        "ar_model.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "ar_model.onnx"),
        "ar_model_int8.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "ar_model_int8.onnx"),
        "flow_model.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "flow_model.onnx"),
        "flow_model_int8.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "flow_model_int8.onnx"),
        "decoder.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "decoder.onnx"),
        "decoder_int8.onnx": ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "decoder_int8.onnx"),
        
        # Alternative MOSS naming scheme
        "text_conditioner.onnx": ("OpenMOSS-Team/MOSS-TTS-models", "text_conditioner.onnx"),
        "flow_lm_main.onnx": ("OpenMOSS-Team/MOSS-TTS-models", "flow_lm_main.onnx"),
        "flow_lm_main_int8.onnx": ("OpenMOSS-Team/MOSS-TTS-models", "flow_lm_main_int8.onnx"),
        "flow_lm_flow.onnx": ("OpenMOSS-Team/MOSS-TTS-models", "flow_lm_flow.onnx"),
        "flow_lm_flow_int8.onnx": ("OpenMOSS-Team/MOSS-TTS-models", "flow_lm_flow_int8.onnx"),
        "mimi_encoder.onnx": ("OpenMOSS-Team/MOSS-Audio-Tokenizer-models", "mimi_encoder.onnx"),
        "mimi_decoder.onnx": ("OpenMOSS-Team/MOSS-Audio-Tokenizer-models", "mimi_decoder.onnx"),
        "mimi_decoder_int8.onnx": ("OpenMOSS-Team/MOSS-Audio-Tokenizer-models", "mimi_decoder_int8.onnx"),
        
        # Tokenizer
        "tokenizer.model": ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", "tokenizer.model"),
    }
    
    all_files = list(models_to_download.items())
    
    print(f"📦 Downloading {len(all_files)} files to {models_dir.absolute()}")
    print()
    
    failed = []
    for filename, (repo_id, repo_filename) in all_files:
        filepath = models_dir / filename
        
        # Check if already exists
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)
            if verbose:
                print(f"✓ {filename} ({file_size:.1f} MB) already exists")
            continue
        
        url = get_huggingface_file_url(repo_id, repo_filename)
        if verbose:
            print(f"Downloading from: {url}")
        
        if not download_file(url, filepath):
            failed.append(filename)
            if filepath.exists():
                filepath.unlink()  # Remove partial download
    
    print()
    if failed:
        print(f"✗ Failed to download {len(failed)} file(s):")
        for f in failed:
            print(f"  - {f}")
        return False
    
    print(f"✓ All {len(all_files)} files downloaded successfully!")
    return True

def verify_models(models_dir: Path = Path("models")) -> bool:
    """Verify that required models exist (with flexible naming)."""
    # Models can be in different naming schemes
    required_patterns = {
        "text_encoding": ["text_encoder.onnx", "text_conditioner.onnx"],
        "audio_encoding": ["audio_encoder.onnx", "mimi_encoder.onnx"],
        "autoregressive": ["ar_model.onnx", "ar_model_int8.onnx", "flow_lm_main.onnx", "flow_lm_main_int8.onnx"],
        "flow_matching": ["flow_model.onnx", "flow_model_int8.onnx", "flow_lm_flow.onnx", "flow_lm_flow_int8.onnx"],
        "decoding": ["decoder.onnx", "decoder_int8.onnx", "mimi_decoder.onnx", "mimi_decoder_int8.onnx"],
        "tokenizer": ["tokenizer.model"],
    }
    
    print("\n🔍 Verifying model files...")
    
    all_files = sorted(models_dir.glob("*.onnx")) + sorted(models_dir.glob("*.model"))
    
    if not all_files:
        print("  ✗ No model files found in", models_dir)
        return False
    
    total_size = 0
    found_groups = {group: [] for group in required_patterns}
    
    for filepath in all_files:
        filename = filepath.name
        size = filepath.stat().st_size / (1024 * 1024)
        total_size += filepath.stat().st_size
        
        # Find which group this belongs to
        for group, patterns in required_patterns.items():
            if any(p == filename for p in patterns):
                found_groups[group].append(filename)
                break
        
        print(f"  ✓ {filename:30s} ({size:8.2f} MB)")
    
    print(f"\n📊 Total: {len(all_files)} files, {total_size / (1024 * 1024):.1f} MB")
    
    # Check if we have all required groups
    missing_groups = [g for g, files in found_groups.items() if not files]
    
    if missing_groups:
        print(f"\n⚠ Missing model groups: {', '.join(missing_groups)}")
        print("Note: Some functionality may not work with incomplete model sets")
    
    print("\n✓ Models verified!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download MOSS-TTS-Nano models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 download_models.py
  python3 download_models.py --models-dir /path/to/models
  python3 download_models.py --verify
        """
    )
    
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store models (default: models)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing models, don't download"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify:
            success = verify_models(args.models_dir)
        else:
            success = download_models(args.models_dir, args.verbose)
            if success:
                success = verify_models(args.models_dir)
        
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
