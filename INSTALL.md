# MOSS-TTS-Nano.cpp Installation Guide

Complete step-by-step guide for building and running MOSS-TTS-Nano.cpp.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Linux Installation](#linux-installation)
3. [macOS Installation](#macos-installation)
4. [Windows Installation](#windows-installation)
5. [Model Setup](#model-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Disk Space**: ~3 GB (for models and dependencies)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Processor**: Any modern CPU with 2+ cores
- **Internet**: Required for first build (dependencies download)

### Software Requirements

- **CMake**: 3.16 or later
- **C++ Compiler**: C++17 support
  - GCC 7+
  - Clang 5+
  - MSVC 2019+
- **Git**: For cloning the repository

## Linux Installation

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Install optional (for audio testing)
sudo apt-get install -y sox ffmpeg

# Clone repository
git clone https://github.com/yourusername/moss-tts-nano-cpp.git
cd moss-tts-nano-cpp

# Configure
cmake -B .build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .build -j$(nproc)

# Binary is ready at ./moss-tts-nano
```

### Fedora/RHEL

```bash
# Install dependencies
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git

# Clone and build
git clone <repo-url>
cd moss-tts-nano-cpp
cmake -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build -j$(nproc)
```

### Alpine Linux

```bash
# Install dependencies
apk add --no-cache build-base cmake git g++ linux-headers

# Clone and build (may require additional flags)
cmake -B .build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build .build -j$(nproc)
```

## macOS Installation

### Using Homebrew

```bash
# Install Xcode Command Line Tools if not present
xcode-select --install

# Install dependencies
brew install cmake git

# Clone repository
git clone <repo-url>
cd moss-tts-nano-cpp

# Configure (uses system clang)
cmake -B .build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .build -j$(sysctl -n hw.ncpu)

# Verify
./moss-tts-nano --help
```

### Using MacPorts

```bash
# Install
sudo port install cmake git

# Standard build process
git clone <repo-url>
cd moss-tts-nano-cpp
cmake -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build -j$(sysctl -n hw.ncpu)
```

## Windows Installation

### Prerequisites

1. **Visual Studio Build Tools**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"

2. **CMake**
   - Download from: https://cmake.org/download/
   - Or: `choco install cmake` (with Chocolatey)

3. **Git**
   - Download from: https://git-scm.com/
   - Or: `choco install git`

### Build Steps

```bash
# Clone repository
git clone <repo-url>
cd moss-tts-nano-cpp

# Configure (MSVC auto-detected)
cmake -B .build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .build --config Release -j %NUMBER_OF_PROCESSORS%

# Binary at: .\moss-tts-nano.exe
```

### Using MinGW

```bash
# Configure for MinGW
cmake -B .build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .build -j %NUMBER_OF_PROCESSORS%
```

## Model Setup

### 1. Create Model Directory

```bash
mkdir -p models
```

### 2. Download Models from HuggingFace

**Option A: Using huggingface-cli**

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download MOSS-TTS-Nano models
huggingface-cli download \
  OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX \
  --local-dir models/

# Download MOSS-Audio-Tokenizer models
huggingface-cli download \
  OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX \
  --local-dir models/
```

**Option B: Manual Download**

1. Visit [MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
2. Download each `.onnx` file to `models/`
3. Download `tokenizer.model` to `models/`

### 3. Verify Model Structure

```bash
ls -la models/
# Expected files:
# - text_encoder.onnx (or similar name)
# - audio_encoder.onnx
# - ar_model.onnx / ar_model_int8.onnx
# - flow_model.onnx / flow_model_int8.onnx
# - decoder.onnx / decoder_int8.onnx
# - tokenizer.model
```

### 4. Prepare Voice Samples

```bash
# Create voices directory
mkdir -p voices

# Add WAV files (22050 Hz recommended)
# Examples:
# - voices/speaker1.wav
# - voices/speaker2.wav
```

Create or download sample voices:

```bash
# Generate test voice (using sox if available)
sox -r 22050 -n voices/test.wav synth 2 sine 440 vol 0.1

# Or download from voice libraries
# (Google Colab notebooks often have examples)
```

## Verification

### Test Installation

```bash
# Show help
./moss-tts-nano --help

# Test with sample text (create dummy audio first)
sox -r 22050 -n models/test.wav synth 1 sine 440 vol 0.05
./moss-tts-nano "Hello world" models/test.wav test_output.wav

# Check output
file test_output.wav
```

### Verify Audio Output

```bash
# Using sox
sox test_output.wav -n stat
sox test_output.wav -n remix - | head

# Using ffprobe (if installed)
ffprobe test_output.wav

# Using Python
python3 -c "import soundfile as sf; x, sr = sf.read('test_output.wav'); print(f'Duration: {len(x)/sr:.2f}s, SR: {sr}')"
```

## Building from Source with Options

### Debug Build

```bash
cmake -B .build -DCMAKE_BUILD_TYPE=Debug
cmake --build .build -j$(nproc)

# Run with debug symbols
gdb ./moss-tts-nano
```

### Custom Compiler

```bash
# Using GCC
cmake -B .build -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release

# Using Clang
cmake -B .build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
```

### Static Build

```bash
# Static linking (Linux/macOS only)
cmake -B .build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

cmake --build .build -j$(nproc)
```

### Custom Installation Path

```bash
cmake -B .build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/moss-tts-nano

cmake --build .build -j$(nproc)
cmake --install .build

# Binary at: /opt/moss-tts-nano/bin/moss-tts-nano
```

## Advanced Configuration

### Cross-Compilation for ARM

```bash
# For ARMv7
cmake -B .build \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/arm-linux-gnueabihf.cmake \
  -DCMAKE_BUILD_TYPE=Release

# For ARM64 (native on Apple Silicon)
# No special configuration needed, just build normally
```

### Performance Tuning

```bash
# Build with optimizations
cmake -B .build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"

cmake --build .build
```

## Troubleshooting

### Build Errors

**CMake not found**
```bash
# Install CMake
sudo apt-get install cmake          # Ubuntu
brew install cmake                   # macOS
choco install cmake                  # Windows
```

**Missing compiler**
```bash
# Install build tools
sudo apt-get install build-essential # Ubuntu
xcode-select --install               # macOS
# or Visual Studio (Windows)
```

**ONNX Runtime download fails**
- Check internet connection
- Try manual download and place in `.build/`
- Check disk space (needs ~1 GB)

### Runtime Errors

**"Models not found"**
```bash
# Verify model directory
ls -la models/
# Should contain all required ONNX files
```

**"Tokenizer load failed"**
```bash
# Check tokenizer file
file models/tokenizer.model
# Should be a SentencePiece model file
```

**"Audio output is silent"**
- Check that voice sample exists and is readable
- Verify audio format (WAV format with mono/stereo)
- Try: `sox voice.wav -n stat`

### Performance Issues

**Slow generation**
- Increase threads: `--threads 8`
- Check CPU usage: `top` or `htop`
- Try INT8 quantization: `--precision int8`

**High memory usage**
- Reduce thread count
- Use INT8 models
- Process in smaller batches

## Verification Checklist

- [ ] CMake builds successfully
- [ ] Binary `moss-tts-nano` executable created
- [ ] Models directory contains all required files
- [ ] Voice samples exist in `voices/` directory
- [ ] Test generation produces audio file
- [ ] Output audio can be played
- [ ] `--verbose` shows detailed output
- [ ] Help: `./moss-tts-nano --help` works

## Next Steps

1. Read [README.md](README.md) for usage guide
2. Test with various text inputs
3. Experiment with different voice samples
4. Set up automated scripts for batch processing

## Support

For build issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Verify all prerequisites installed
3. Try rebuild: `rm -rf .build && cmake -B .build ... && cmake --build .build`
4. Check CMake output for specific error messages

---

**Last Updated**: 2026-04-27  
**Version**: 1.0.0
