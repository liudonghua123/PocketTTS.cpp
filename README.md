# MOSS-TTS-Nano.cpp

Complete C++ implementation of MOSS-TTS-Nano-100M using ONNX Runtime.

- **Official Models**: [OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- **Audio Tokenizer**: [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

## Features

- ✅ Single-file C++17 implementation
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Command-line interface
- ✅ Streaming audio output
- ✅ INT8 quantization support
- ✅ Multi-threaded inference
- ✅ Real-time audio processing

## Quick Start

### Automated Setup (Recommended)

```bash
# Run the setup script
./setup-moss-tts.sh

# Or with custom directories
./setup-moss-tts.sh --models-dir /path/to/models --voices-dir /path/to/voices
```

This will:
1. Check dependencies
2. Build the binary if needed
3. Download models from HuggingFace
4. Create test voice samples
5. Run validation tests

### Manual Setup

#### Building

```bash
# Create build directory
cmake -B .build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .build -j$(nproc)

# Binary is at ./moss-tts-nano
```

### Running

```bash
# Basic usage
./moss-tts-nano "Hello world" reference_voice.wav output.wav

# With options
./moss-tts-nano --verbose --threads 4 "Text to convert" voice.wav out.wav

# Show help
./moss-tts-nano --help
```

## Model Setup

### Automatic (Recommended)

Use the setup script:
```bash
./setup-moss-tts.sh
```

### Manual Download

```bash
# Download using the download script
python3 download_models.py --models-dir models

# Or verify existing models
python3 download_models.py --verify
```

## Architecture

```
Text Input
    ↓
[Text Encoder] → Text Embeddings [B, T, hidden_dim]
                     ↓
Audio Reference ──[Audio Encoder] → Audio Conditioning [B, F, hidden_dim]
                     ↓
                [AR Model (Stateful)] ← KV Cache States
                     ↓
            [Flow Matching Model]
                     ↓
               [Audio Decoder]
                     ↓
            Output Audio (22050 Hz)
```

## Performance

Expected performance on modern CPUs:

| Metric | Value |
|--------|-------|
| Model Size | 100 MB (FP32), 50 MB (INT8) |
| Memory Usage | 100-150 MB |
| RTFx (4-core) | 2-5x |
| First Chunk Latency | <100 ms |
| Sample Rate | 22050 Hz |

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 384 |
| Latent Dimension | 32 |
| Number of Layers | 4 |
| Attention Heads | 8 |
| Max Sequence Length | 1000 |

### Supported Model Naming

The code supports flexible model naming for compatibility with different ONNX model sources:

| Function | Standard Name | Alternative Name |
|----------|---------------|------------------|
| Text Encoding | `text_encoder.onnx` | `text_conditioner.onnx` |
| Audio Encoding | `audio_encoder.onnx` | `mimi_encoder.onnx` |
| Autoregressive | `ar_model[_int8].onnx` | `flow_lm_main[_int8].onnx` |
| Flow Matching | `flow_model[_int8].onnx` | `flow_lm_flow[_int8].onnx` |
| Audio Decoding | `decoder[_int8].onnx` | `mimi_decoder[_int8].onnx` |
| Tokenization | `tokenizer.model` | (SentencePiece format) |

## Command Line Options

```
Usage: moss-tts-nano [OPTIONS] TEXT VOICE [OUTPUT]

Options:
  -h, --help              Show help message
  -v, --verbose           Enable verbose output
  --models-dir PATH       Models directory (default: models)
  --voices-dir PATH       Voices samples directory (default: voices)
  --threads N             Number of threads (default: auto)
  --precision int8|fp32   Model precision (default: int8)

Examples:
  moss-tts-nano "Hello world" voice.wav output.wav
  moss-tts-nano --verbose --threads 4 "Text" voice.wav
```

## Implementation Details

### Core Components

- **MossTTSNano**: Main TTS engine class
  - Model loading and initialization
  - Audio generation pipeline
  - Streaming support

- **OrtSession**: ONNX Runtime wrapper
  - Model inference
  - Input/output handling
  - Model metadata management

- **StateBuffer**: State management
  - KV cache handling
  - Double buffering
  - State persistence

- **Tokenizer**: SentencePiece integration
  - Text tokenization
  - Vocabulary management

### Audio Processing

- Input: WAV, MP3, FLAC formats (via dr_libs)
- Output: 32-bit IEEE float PCM @ 22050 Hz
- Frame-based processing
- Streaming chunk support

## Dependencies

- **ONNX Runtime** 1.23.2 (auto-downloaded)
- **SentencePiece** 0.2.1 (auto-built)
- **dr_libs** (headers only)
- **C++17 compatible compiler**

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x64 | ✅ Tested | Primary platform |
| macOS (Universal) | ✅ Supported | M1/M2/Intel |
| Windows x64 | ✅ Supported | MSVC 2019+ |
| Linux ARM64 | ✅ Supported | Via aarch64 builds |

## Limitations & Future Work

### Current Limitations

- Fixed voice samples (no voice cloning)
- Single speaker per reference
- Simplified state management
- No streaming partial output caching

### Future Enhancements

- [ ] HTTP server API
- [ ] Python bindings
- [ ] GPU acceleration (CUDA/CoreML)
- [ ] HTTP streaming API
- [ ] Multiple voice support
- [ ] Advanced voice conditioning

## Building from Source

### Requirements

- CMake 3.16+
- C++17 compiler
- ~2 GB disk space (for model downloads)
- Internet connection (first build)

### Build Steps

```bash
# Configure
cmake -B .build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++

# Build
cmake --build .build -j$(nproc)

# Test
./moss-tts-nano --help

# Install (optional)
cmake --install .build --prefix /usr/local
```

### Build Options

```bash
# Debug build
cmake -B .build -DCMAKE_BUILD_TYPE=Debug

# With custom thread count
cmake --build .build -j4

# Cross-compilation
cmake --toolchain arm-linux-gnueabihf.cmake -B .build
```

## Testing

### Unit Tests

```bash
# Generate with verbose output
./moss-tts-nano --verbose "Test audio generation" models/test.wav test_out.wav

# Check output
file test_out.wav
ffprobe test_out.wav
```

### Benchmarking

```bash
# Time generation
time ./moss-tts-nano "Short text" voice.wav /tmp/test.wav

# Profile with perf (Linux)
perf record -g ./moss-tts-nano "Text" voice.wav /tmp/test.wav
perf report
```

## Troubleshooting

### Common Issues

**"Failed to load models"**
- Verify model files exist in `models/` directory
- Check file permissions
- Ensure correct ONNX format

**"Tokenizer load failed"**
- Download `tokenizer.model` from HuggingFace
- Place in `models/` directory
- Check file integrity

**"ONNX Runtime not found"**
- First build may take time for dependency download
- Check CMake output for errors
- Verify internet connection

**"Audio output is silent"**
- Check voice sample quality
- Verify text input is not empty
- Check output file format

### Debug Mode

```bash
# Enable verbose output
./moss-tts-nano --verbose --threads 1 "Debug text" voice.wav debug.wav

# Check binary dependencies (Linux)
ldd ./moss-tts-nano

# Inspect generated audio (macOS/Linux)
sox debug.wav -n stat
```

## Performance Optimization

### Memory Usage

- Use INT8 quantization (`--precision int8`)
- Reduce thread count on memory-constrained systems
- Stream output to avoid full buffer storage

### Speed

- Increase thread count (`--threads 8`)
- Use FP32 on systems without quantization support
- Process multiple batches concurrently

### Quality

- Use FP32 precision for highest quality
- Use reference voices with good audio quality
- Longer audio segments for better context

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional language support
- [ ] Performance optimizations
- [ ] Platform-specific builds
- [ ] Advanced features (voice mixing, etc.)

## License

Apache 2.0

## References

- [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- [ONNX Runtime](https://onnxruntime.ai/)
- [SentencePiece](https://github.com/google/sentencepiece)
- [dr_libs](https://github.com/mackron/dr_libs)

## Citation

If you use MOSS-TTS-Nano.cpp, please cite:

```bibtex
@software{moss_tts_nano_cpp_2026,
  title={MOSS-TTS-Nano.cpp: C++ Runtime for MOSS-TTS-Nano},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/moss-tts-nano-cpp}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Status**: ✅ Stable  
**Version**: 1.0.0  
**Last Updated**: 2026-04-27
