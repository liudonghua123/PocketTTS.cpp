#!/bin/bash
#
# setup-moss-tts.sh — Setup script for MOSS-TTS-Nano
#
# This script:
# 1. Downloads ONNX models from HuggingFace
# 2. Prepares voice sample directory
# 3. Verifies installation
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default directories
MODELS_DIR="${MODELS_DIR:-models}"
VOICES_DIR="${VOICES_DIR:-voices}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Functions
print_header() {
    echo -e "${BLUE}
╔════════════════════════════════════════════════════════════╗
║    MOSS-TTS-Nano Setup                                     ║
╚════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${GREEN}→${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_requirements() {
    print_step "Checking requirements..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3.6+"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    if [ ! -f "$SCRIPT_DIR/moss-tts-nano" ]; then
        print_warning "Binary moss-tts-nano not found"
        print_step "Building from source..."
        
        if ! command -v cmake &> /dev/null; then
            print_error "CMake not found. Please install CMake 3.16+"
            exit 1
        fi
        
        cd "$SCRIPT_DIR"
        cmake -B .build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
        cmake --build .build -j$(nproc)
        print_success "Binary built successfully"
    else
        print_success "Binary found: $(file "$SCRIPT_DIR/moss-tts-nano" | cut -d: -f2-)"
    fi
}

setup_directories() {
    print_step "Setting up directories..."
    
    mkdir -p "$MODELS_DIR"
    mkdir -p "$VOICES_DIR"
    
    print_success "Models directory: $(cd "$MODELS_DIR" && pwd)"
    print_success "Voices directory: $(cd "$VOICES_DIR" && pwd)"
}

download_models() {
    print_step "Downloading models from HuggingFace..."
    echo ""
    
    if python3 "$SCRIPT_DIR/download_models.py" --models-dir "$MODELS_DIR" -v; then
        print_success "Models downloaded and verified"
        return 0
    else
        print_error "Failed to download models"
        return 1
    fi
}

create_test_voice() {
    print_step "Creating test voice sample..."
    
    # Check if we already have a test voice
    if [ -f "$VOICES_DIR/test.wav" ]; then
        print_success "Test voice already exists"
        return 0
    fi
    
    # Try to create with sox if available
    if command -v sox &> /dev/null; then
        sox -r 22050 -n "$VOICES_DIR/test.wav" synth 2 sine 440 vol 0.05
        print_success "Test voice created: $VOICES_DIR/test.wav"
        return 0
    fi
    
    # Try with ffmpeg
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f lavfi -i "sine=frequency=440:duration=2" \
               -ar 22050 -q:a 9 \
               "$VOICES_DIR/test.wav" 2>/dev/null || true
        if [ -f "$VOICES_DIR/test.wav" ]; then
            print_success "Test voice created with ffmpeg: $VOICES_DIR/test.wav"
            return 0
        fi
    fi
    
    print_warning "Could not create test voice (sox or ffmpeg not found)"
    print_warning "You can add a voice sample manually to $VOICES_DIR/"
    return 0
}

test_installation() {
    print_step "Testing installation..."
    echo ""
    
    if [ ! -f "$VOICES_DIR/test.wav" ]; then
        print_warning "No voice sample found, skipping TTS test"
        print_warning "Add a voice sample to $VOICES_DIR/ to test"
        return 0
    fi
    
    TEST_OUTPUT="/tmp/moss_tts_test_$$.wav"
    
    if "$SCRIPT_DIR/moss-tts-nano" \
        --models-dir "$MODELS_DIR" \
        --verbose \
        "Hello from MOSS-TTS-Nano" \
        "$VOICES_DIR/test.wav" \
        "$TEST_OUTPUT"; then
        
        if [ -f "$TEST_OUTPUT" ]; then
            SIZE=$(stat -f%z "$TEST_OUTPUT" 2>/dev/null || stat -c%s "$TEST_OUTPUT" 2>/dev/null || echo "unknown")
            print_success "TTS test successful!"
            print_success "Output saved to: $TEST_OUTPUT ($(numfmt --to=iec $SIZE 2>/dev/null || echo "$SIZE bytes"))"
            
            # Try to play audio if possible
            if command -v play &> /dev/null; then
                print_step "Playing test audio..."
                play "$TEST_OUTPUT" 2>/dev/null || true
            fi
            
            rm -f "$TEST_OUTPUT"
            return 0
        else
            print_error "Output file not created"
            return 1
        fi
    else
        print_error "TTS test failed (check verbose output above)"
        rm -f "$TEST_OUTPUT"
        return 1
    fi
}

show_next_steps() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ Setup Complete! Next Steps:                                ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1. Add voice samples to the voices/ directory:"
    echo "   - Download from HuggingFace or online sources"
    echo "   - Recommended: 22050 Hz sample rate"
    echo "   - Format: WAV, MP3, or FLAC"
    echo ""
    echo "2. Run the TTS engine:"
    echo "   ./moss-tts-nano \"Your text here\" voices/speaker.wav output.wav"
    echo ""
    echo "3. Use command options:"
    echo "   ./moss-tts-nano --help"
    echo ""
    echo "For more information, see:"
    echo "   - README.md"
    echo "   - INSTALL.md"
    echo "   - DEVELOPER.md"
    echo ""
}

main() {
    print_header
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --models-dir)
                MODELS_DIR="$2"
                shift 2
                ;;
            --voices-dir)
                VOICES_DIR="$2"
                shift 2
                ;;
            --skip-test)
                SKIP_TEST=1
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --models-dir DIR     Models directory (default: models)"
                echo "  --voices-dir DIR     Voices directory (default: voices)"
                echo "  --skip-test          Skip installation test"
                echo "  -h, --help           Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_requirements
    setup_directories
    
    if ! download_models; then
        print_error "Setup failed at model download"
        exit 1
    fi
    
    create_test_voice
    
    if [ -z "$SKIP_TEST" ]; then
        if ! test_installation; then
            print_warning "Installation test had issues, but setup may still be usable"
        fi
    fi
    
    show_next_steps
    print_success "Setup completed!"
}

# Run main
main "$@"
