# MOSS-TTS-Nano Development Guide

Technical documentation for MOSS-TTS-Nano.cpp implementation and architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Classes](#core-classes)
3. [API Reference](#api-reference)
4. [Model Integration](#model-integration)
5. [Extending the Implementation](#extending-the-implementation)
6. [Performance Tuning](#performance-tuning)
7. [Debugging](#debugging)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   MOSS-TTS-Nano.cpp                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Config    │  │  AudioData   │  │  Tokenizer   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ▲                  ▲                  ▲             │
│         └──────────────────┼──────────────────┘             │
│                            │                               │
│                  ┌─────────────────────┐                   │
│                  │   MossTTSNano       │                   │
│                  │  (Main Engine)      │                   │
│                  └─────────────────────┘                   │
│                            │                               │
│          ┌─────────────────┼─────────────────┐             │
│          ▼                 ▼                 ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ OrtSession   │  │ StateBuffer  │  │ AudioCodec   │    │
│  │ (ONNX)       │  │ (State Mgmt) │  │ (I/O)        │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                 │                 │             │
│  ┌──────┴─────────────────┴─────────────────┴──────┐     │
│  │         ONNX Runtime Layer                       │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Classes

### MossTTSNano

Main TTS engine class. Orchestrates text-to-speech generation.

```cpp
class MossTTSNano {
public:
    explicit MossTTSNano(const Config& cfg = {});
    
    // Main generation API
    AudioData generate(const std::string& text, 
                      const std::string& voice_path);
    
    // Streaming API
    bool stream(const std::string& text, 
               const std::string& voice_path,
               std::function<bool(const float*, size_t)> callback);
    
    // Audio I/O
    static AudioData load_audio(const std::string& path);
    static bool save_audio(const AudioData& audio, 
                          const std::string& path);
    
    // Constants
    static constexpr int SR = ModelSpec::SAMPLE_RATE;
};
```

#### Key Methods

**`generate(text, voice_path)`**
- Generates complete audio for text
- Blocks until generation complete
- Returns `AudioData` with PCM samples

**`stream(text, voice_path, callback)`**
- Generates audio with streaming output
- Calls callback for each chunk
- Callback can signal abort (return false)

### OrtSession

ONNX Runtime wrapper for model inference.

```cpp
class OrtSession {
public:
    OrtSession(Ort::Env& env, const std::string& path,
              const Ort::SessionOptions& opts,
              const std::string& name = "");
    
    Ort::Session& session();
    
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& in);
    
    void print_info() const;
    
    // Metadata access
    const std::vector<std::string>& input_names() const;
    const std::vector<std::string>& output_names() const;
};
```

#### Usage Example

```cpp
// Load model
OrtSession model(env, "models/text_encoder.onnx", opts);

// Create input
std::vector<int64_t> input_shape = {1, 10};  // batch=1, seq_len=10
Ort::Value input = Ort::Value::CreateTensor<int>(/*...*/);

// Run inference
std::vector<Ort::Value> outputs = model.run({input});

// Access output
const float* out_data = outputs[0].GetTensorData<float>();
```

### StateBuffer

Manages KV cache and model states.

```cpp
struct StateBuffer {
    std::vector<std::vector<float>> f32[2];    // Float32 states (double-buffered)
    std::vector<std::vector<int64_t>> i64[2];  // Int64 states
    std::vector<std::pair<std::string, size_t>> state_info;
    
    void init(const std::vector<std::string>& state_names);
    int in_buf() const;   // Get input buffer index
    int out_buf() const;  // Get output buffer index
    void swap();          // Swap buffers for next step
    void reset();         // Reset all states
};
```

#### Usage Pattern

```cpp
StateBuffer state;
state.init({"kv_layer0", "kv_layer1", /*...*/});

for (int step = 0; step < max_steps; ++step) {
    // Create input from in_buf()
    Ort::Value state_in = create_state_tensor(state.in_buf());
    
    // Run model
    auto outputs = model.run({/*...*/, state_in});
    
    // Update from out_buf()
    update_state(state.out_buf(), outputs.back());
    
    // Prepare for next step
    state.swap();
}
```

### Config

Configuration structure for TTS engine.

```cpp
struct Config {
    std::string models_dir;        // Model files directory
    std::string tokenizer_path;    // Tokenizer model path
    std::string voices_dir;        // Reference voices directory
    std::string precision;         // "int8" or "fp32"
    float temperature;             // Sampling temperature (0-1)
    float eos_threshold;           // EOS detection threshold
    int num_threads;               // Thread count (0 = auto)
    int max_chunk_frames;          // Max frames per decoder chunk
    bool verbose;                  // Verbose output
    bool voice_cache;              // Cache voice encodings
};
```

## API Reference

### AudioData Structure

```cpp
struct AudioData {
    std::vector<float> samples;    // PCM samples (32-bit float)
    int sample_rate;               // Sample rate in Hz
    
    float duration_sec() const;    // Get duration in seconds
};
```

### Tensor Structure

```cpp
struct Tensor {
    std::vector<int64_t> shape;    // Tensor shape
    std::vector<float> data;       // Tensor data (float32)
    
    Tensor(std::vector<int64_t> s);  // Create with shape
    size_t numel() const;           // Total number of elements
    float* ptr();                   // Data pointer
    const float* ptr() const;
};
```

### RNG Utilities

```cpp
namespace rng {
    void seed(uint64_t v);                    // Seed RNG
    uint64_t next();                          // Next random uint64
    float uniform(float a = 0, float b = 1);  // Uniform [a, b)
    float normal(float mu = 0, float sigma = 1);  // Normal(mu, sigma)
}
```

## Model Integration

### Expected ONNX Model Signatures

All models use batch size = 1 unless otherwise noted.

**text_encoder.onnx**
- Inputs: `token_ids` [1, seq_len] int64
- Outputs: `embeddings` [1, seq_len, hidden_dim] float32

**audio_encoder.onnx**
- Inputs: `audio` [1, channels, samples] float32
- Outputs: `conditioning` [1, frames, hidden_dim] float32

**ar_model.onnx** (Autoregressive)
- Inputs: 
  - `x` [1, latent_dim] float32 (previous latent or BOS)
  - `text_emb` [1, seq_len, hidden_dim] float32
  - `audio_cond` [1, frames, hidden_dim] float32
  - state inputs (see State Tensors)
- Outputs:
  - `conditioning` [1, hidden_dim] float32 (next hidden state)
  - `eos_logit` [1] float32 (end-of-speech probability)
  - state outputs

**flow_model.onnx** (Flow Matching)
- Inputs:
  - `c` [hidden_dim] float32 (conditioning)
  - `s` [1, 1] float32 (time step start)
  - `t` [1, 1] float32 (time step end)
  - `x` [1, latent_dim] float32 (noise)
- Outputs:
  - `flow_direction` [latent_dim] float32

**decoder.onnx** (Audio Decoder)
- Inputs:
  - `codes` [1, 1, latent_dim] float32
  - state inputs
- Outputs:
  - `audio` [1, samples] float32
  - state outputs

### Adding Custom Models

To support new models:

1. **Update ModelSpec constants** (if needed)
   ```cpp
   struct ModelSpec {
       static constexpr int HIDDEN_DIM = 384;  // Your model's hidden dim
       static constexpr int LATENT_DIM = 32;
       static constexpr int NUM_LAYERS = 4;
   };
   ```

2. **Implement model-specific encoding**
   ```cpp
   Tensor encode_text(const std::vector<int>& token_ids) {
       // Your custom implementation
   }
   ```

3. **Update generation pipeline**
   ```cpp
   AudioData generate_audio(const Tensor& text_emb, 
                           const Tensor& audio_cond) {
       // Your custom pipeline
   }
   ```

## Extending the Implementation

### Adding Language Support

```cpp
// In MossTTSNano class
private:
    std::unordered_map<std::string, std::unique_ptr<Tokenizer>> tokenizers_;
    
public:
    void load_tokenizer(const std::string& lang, 
                       const std::string& model_path) {
        tokenizers_[lang] = std::make_unique<Tokenizer>(model_path);
    }
    
    std::vector<int> tokenize(const std::string& text, 
                             const std::string& lang) {
        return tokenizers_[lang]->encode(text);
    }
```

### Adding Voice Cloning

```cpp
// Store voice embeddings
struct VoiceEmbedding {
    std::string name;
    Tensor embedding;  // Speaker embedding vector
};

private:
    std::unordered_map<std::string, VoiceEmbedding> voice_cache_;
    
public:
    void register_voice(const std::string& name, 
                       const std::string& audio_path) {
        AudioData audio = load_audio(audio_path);
        Tensor emb = encode_audio(audio);
        voice_cache_[name] = {name, emb};
    }
```

### Adding Batch Processing

```cpp
// Generate for multiple texts
std::vector<AudioData> batch_generate(
    const std::vector<std::string>& texts,
    const std::string& voice_path) {
    
    std::vector<AudioData> results;
    for (const auto& text : texts) {
        results.push_back(generate(text, voice_path));
    }
    return results;
}
```

## Performance Tuning

### Thread Strategy

```cpp
// Determine optimal thread count
int get_optimal_threads() {
    int physical = std::thread::hardware_concurrency();
    // Use half the cores for TTS (leave room for OS)
    return physical > 2 ? physical / 2 : 1;
}

Config cfg;
cfg.num_threads = get_optimal_threads();
```

### Memory Optimization

```cpp
// Use INT8 quantization
Config cfg;
cfg.precision = "int8";  // ~2x smaller models

// Stream output instead of buffering
tts.stream(text, voice, [](const float* data, size_t n) {
    // Process chunk immediately
    write_to_file(data, n);
    return true;  // Continue streaming
});
```

### Inference Optimization

```cpp
// Batch similar-length texts
std::sort(texts.begin(), texts.end(), 
         [](const auto& a, const auto& b) { return a.size() < b.size(); });

// Reuse models across multiple generations
for (const auto& text : texts) {
    auto audio = tts.generate(text, voice);
}
```

## Debugging

### Enable Verbose Output

```cpp
Config cfg;
cfg.verbose = true;

MossTTSNano tts(cfg);  // Detailed initialization info
audio = tts.generate("text", "voice.wav");  // Generation logs
```

### Trace Model Execution

```cpp
// Add logging to OrtSession::run()
std::vector<Ort::Value> run(const std::vector<Ort::Value>& in) {
    std::cout << "Running " << name_ << " with " << in.size() 
              << " inputs\n";
    
    auto result = sess_.Run(/*...*/);
    
    std::cout << "Got " << result.size() << " outputs\n";
    return result;
}
```

### Inspect Tensor Contents

```cpp
// Helper function
void print_tensor(const Ort::Value& tensor, int max_elems = 10) {
    auto info = tensor.GetTensorTypeAndShapeInfo();
    const float* data = tensor.GetTensorData<float>();
    
    std::cout << "Shape: [";
    for (auto d : info.GetShape()) std::cout << d << " ";
    std::cout << "]\nData (first " << max_elems << "): [";
    for (int i = 0; i < std::min(max_elems, (int)info.GetElementCount()); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "]\n";
}
```

### Audio Quality Inspection

```bash
# Check generated audio properties
sox output.wav -n stat

# Visualize waveform
sox output.wav -n spectrogram -o output.png

# Listen to sample
play output.wav

# Compare with reference
sox ref.wav output.wav -n stat
```

### Performance Profiling

```bash
# Linux: Use perf
perf record -F 99 -g ./moss-tts-nano "text" voice.wav out.wav
perf report

# macOS: Use Instruments
xcrun xctrace recordsample ./moss-tts-nano "text" voice.wav

# All platforms: Time measurement
time ./moss-tts-nano "text" voice.wav out.wav
```

## Common Issues and Solutions

### Issue: Models not found
**Solution**: Verify `models/` directory structure
```bash
ls -la models/
# Should show: text_encoder.onnx, audio_encoder.onnx, etc.
```

### Issue: Slow inference
**Solution**: Enable parallel processing
```cpp
cfg.num_threads = 4;  // Use multiple threads
cfg.precision = "int8";  // Use quantized models
```

### Issue: Memory errors
**Solution**: Check state buffer initialization
```cpp
// Ensure state_buf is properly initialized
state_buf_.init({"state0", "state1", /*...*/});

// Verify state shapes match model expectations
```

## References

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [SentencePiece Documentation](https://github.com/google/sentencepiece)
- [dr_libs Audio Libraries](https://github.com/mackron/dr_libs)

---

**Last Updated**: 2026-04-27  
**Version**: 1.0.0
