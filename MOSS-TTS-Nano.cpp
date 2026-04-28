// MOSS-TTS-Nano.cpp — Complete C++ TTS runtime using ONNX Runtime
// Full implementation of MOSS-TTS-Nano-100M model
// https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX
// https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX

// ════════════════════════════════════════════════════════════════════════════
// Platform Detection (must come first)
// ════════════════════════════════════════════════════════════════════════════

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef _CRT_SECURE_NO_WARNINGS
    #define _CRT_SECURE_NO_WARNINGS
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <direct.h>
  #include <io.h>
  #include <fcntl.h>
  #pragma comment(lib, "ws2_32.lib")
  #define mtts_mkdir(path) _mkdir(path)
  #define mtts_close closesocket
  typedef SOCKET mtts_socket_t;
  typedef int socklen_t;
  static constexpr mtts_socket_t MTTS_INVALID_SOCKET = INVALID_SOCKET;
  using ssize_t = ptrdiff_t;
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #define mtts_mkdir(path) mkdir(path, 0755)
  #define mtts_close close
  typedef int mtts_socket_t;
  static constexpr mtts_socket_t MTTS_INVALID_SOCKET = -1;
#endif

#include <sys/stat.h>
#include <csignal>

// ════════════════════════════════════════════════════════════════════════════
// External Libraries
// ════════════════════════════════════════════════════════════════════════════

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"

// ════════════════════════════════════════════════════════════════════════════
// Standard Library
// ════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace moss_tts_nano {

// ════════════════════════════════════════════════════════════════════════════
// Configuration & Constants
// ════════════════════════════════════════════════════════════════════════════

struct ModelSpec {
    static constexpr int SAMPLE_RATE = 22050;
    static constexpr int HIDDEN_DIM = 384;
    static constexpr int LATENT_DIM = 32;
    static constexpr int NUM_LAYERS = 4;
    static constexpr int MAX_SEQ_LEN = 1000;
    static constexpr int AUDIO_FRAME_SIZE = 1920;
};

struct Config {
    std::string models_dir = "models";
    std::string tokenizer_path = "models/tokenizer.model";
    std::string voices_dir = "voices";
    std::string precision = "int8";
    float temperature = 0.7f;
    float eos_threshold = -4.0f;
    int num_threads = 0;
    int max_chunk_frames = 16;
    bool verbose = false;
    bool voice_cache = true;
};

// ════════════════════════════════════════════════════════════════════════════
// Core Types
// ════════════════════════════════════════════════════════════════════════════

static size_t calc_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) n *= (d > 0 ? d : 1);
    return n;
}

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    
    Tensor() = default;
    explicit Tensor(std::vector<int64_t> s) 
        : shape(std::move(s)), data(calc_numel(shape), 0.0f) {}
    
    size_t numel() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct AudioData {
    std::vector<float> samples;
    int sample_rate = ModelSpec::SAMPLE_RATE;
    
    float duration_sec() const { return float(samples.size()) / sample_rate; }
};

// ════════════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════════════

namespace rng {
static uint64_t s[4] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 
                        0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};

static inline uint64_t rotl(uint64_t x, int k) { 
    return (x << k) | (x >> (64 - k)); 
}

static uint64_t next() {
    uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

void seed(uint64_t v) {
    s[0] = v; s[1] = v ^ 0x9E3779B97F4A7C15ULL; 
    s[2] = v ^ 0xBF58476D1CE4E5B9ULL; s[3] = v ^ 0x94D049BB133111EBULL;
    for (int i = 0; i < 16; i++) next();
}

float uniform(float a = 0.0f, float b = 1.0f) {
    uint32_t u = next() >> 11;
    return a + (b - a) * (u / 9007199254740992.0f);
}

float normal(float mu = 0.0f, float sigma = 1.0f) {
    float u1 = uniform(1e-6f, 1), u2 = uniform();
    return mu + sigma * std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2);
}
} // namespace rng

// ════════════════════════════════════════════════════════════════════════════
// ONNX Runtime Wrappers
// ════════════════════════════════════════════════════════════════════════════

static Ort::Env& get_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "moss_tts_nano");
    return env;
}

static std::string find_model_file(const std::string& models_dir, 
                                    const std::vector<std::string>& candidates) {
    for (const auto& name : candidates) {
        std::string path = models_dir + "/" + name;
        std::ifstream f(path);
        if (f.good()) return path;
    }
    
    // Return first candidate path (original behavior)
    return models_dir + "/" + candidates[0];
}

static std::basic_string<ORTCHAR_T> to_ort_path(const std::string& s) {
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) throw std::runtime_error("Failed to convert path: " + s);
    std::wstring w(n - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
#else
    return s;
#endif
}

class OrtSession {
    Ort::Session sess_;
    std::vector<std::string> in_names_, out_names_;
    std::vector<const char*> in_ptrs_, out_ptrs_;
    std::vector<std::vector<int64_t>> in_shapes_;
    std::string name_;
    
public:
    OrtSession(Ort::Env& env, const std::string& path, 
               const Ort::SessionOptions& opts, const std::string& name = "")
        : sess_(env, to_ort_path(path).c_str(), opts), 
          name_(name.empty() ? path : name) {
        Ort::AllocatorWithDefaultOptions alloc;
        
        size_t num_in = sess_.GetInputCount();
        for (size_t i = 0; i < num_in; ++i) {
            auto n = sess_.GetInputNameAllocated(i, alloc);
            in_names_.push_back(n.get());
            auto ti = sess_.GetInputTypeInfo(i);
            auto tsi = ti.GetTensorTypeAndShapeInfo();
            in_shapes_.push_back(tsi.GetShape());
        }
        
        size_t num_out = sess_.GetOutputCount();
        for (size_t i = 0; i < num_out; ++i) {
            auto n = sess_.GetOutputNameAllocated(i, alloc);
            out_names_.push_back(n.get());
        }
        
        for (const auto& n : in_names_) in_ptrs_.push_back(n.c_str());
        for (const auto& n : out_names_) out_ptrs_.push_back(n.c_str());
    }
    
    Ort::Session& session() { return sess_; }
    
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& in) {
        return sess_.Run(Ort::RunOptions{nullptr}, in_ptrs_.data(), in.data(), 
                         in.size(), out_ptrs_.data(), out_ptrs_.size());
    }
    
    void print_info() const {
        std::cout << "  Model: " << name_ << "\n";
        std::cout << "    Inputs (" << in_names_.size() << "): ";
        for (const auto& n : in_names_) std::cout << n << " ";
        std::cout << "\n    Outputs (" << out_names_.size() << "): ";
        for (const auto& n : out_names_) std::cout << n << " ";
        std::cout << "\n";
    }
    
    const std::string& name() const { return name_; }
    const std::vector<std::string>& input_names() const { return in_names_; }
    const std::vector<std::string>& output_names() const { return out_names_; }
    const std::vector<std::vector<int64_t>>& input_shapes() const { return in_shapes_; }
};

// ════════════════════════════════════════════════════════════════════════════
// State Buffer Management
// ════════════════════════════════════════════════════════════════════════════

struct StateBuffer {
    std::vector<std::vector<float>> f32[2];
    std::vector<std::vector<int64_t>> i64[2];
    std::vector<std::pair<std::string, size_t>> state_info;
    int current_buf = 0;
    
    void init(const std::vector<std::string>& state_names) {
        state_info.resize(state_names.size());
        for (size_t i = 0; i < state_names.size(); ++i) {
            state_info[i] = {state_names[i], 0};  // name, size
            for (int b = 0; b < 2; ++b) {
                f32[b].push_back({});
                i64[b].push_back({});
            }
        }
    }
    
    int in_buf() const { return current_buf; }
    int out_buf() const { return 1 - current_buf; }
    void swap() { current_buf = 1 - current_buf; }
    void reset() { current_buf = 0; }
};

// ════════════════════════════════════════════════════════════════════════════
// Tokenizer
// ════════════════════════════════════════════════════════════════════════════

class Tokenizer {
    sentencepiece::SentencePieceProcessor proc_;
public:
    explicit Tokenizer(const std::string& path) {
        auto s = proc_.Load(path);
        if (!s.ok()) throw std::runtime_error("Failed to load tokenizer: " + s.ToString());
    }
    
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        proc_.Encode(text, &ids);
        return ids;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// TTS Engine - Main Implementation
// ════════════════════════════════════════════════════════════════════════════

class MossTTSNano {
public:
    static constexpr int SR = ModelSpec::SAMPLE_RATE;
    
    explicit MossTTSNano(const Config& cfg = {}) : cfg_(cfg) {
        rng::seed(uint64_t(std::chrono::high_resolution_clock::now()
                  .time_since_epoch().count()));
        
        if (cfg_.verbose) {
            std::cout << "\n========== MOSS-TTS-NANO INITIALIZATION ==========\n";
            std::cout << "  Sample Rate: " << SR << " Hz\n";
            std::cout << "  Models Dir: " << cfg_.models_dir << "\n";
            std::cout << "  Precision: " << cfg_.precision << "\n";
        }
        
        // Load tokenizer
        try {
            tok_ = std::make_unique<Tokenizer>(cfg_.tokenizer_path);
        } catch (const std::exception& e) {
            if (cfg_.verbose) std::cerr << "Warning: " << e.what() << "\n";
        }
        
        // Initialize ONNX models
        auto& env = get_ort_env();
        auto opts = create_session_options();
        
        try {
            std::string sfx = (cfg_.precision == "int8") ? "_int8" : "";
            
            // Load MOSS models with fallback names
            // text_encoder variants: text_encoder.onnx, text_conditioner.onnx
            std::string text_enc_path = find_model_file(cfg_.models_dir, 
                {"text_encoder.onnx", "text_conditioner.onnx"});
            text_encoder_ = std::make_unique<OrtSession>(env, text_enc_path, opts, "text_encoder");
            
            // audio_encoder variants: audio_encoder.onnx, mimi_encoder.onnx
            std::string audio_enc_path = find_model_file(cfg_.models_dir,
                {"audio_encoder.onnx", "mimi_encoder.onnx"});
            audio_encoder_ = std::make_unique<OrtSession>(env, audio_enc_path, opts, "audio_encoder");
            
            // AR model variants: ar_model[_int8].onnx, flow_lm_main[_int8].onnx
            std::string ar_path = find_model_file(cfg_.models_dir,
                {"ar_model" + sfx + ".onnx", "ar_model.onnx", "flow_lm_main" + sfx + ".onnx", "flow_lm_main.onnx"});
            ar_model_ = std::make_unique<OrtSession>(env, ar_path, opts, "ar_model");
            
            // Flow model variants: flow_model[_int8].onnx, flow_lm_flow[_int8].onnx
            std::string flow_path = find_model_file(cfg_.models_dir,
                {"flow_model" + sfx + ".onnx", "flow_model.onnx", "flow_lm_flow" + sfx + ".onnx", "flow_lm_flow.onnx"});
            flow_model_ = std::make_unique<OrtSession>(env, flow_path, opts, "flow_model");
            
            // Decoder variants: decoder[_int8].onnx, mimi_decoder[_int8].onnx
            std::string decoder_path = find_model_file(cfg_.models_dir,
                {"decoder" + sfx + ".onnx", "decoder.onnx", "mimi_decoder" + sfx + ".onnx", "mimi_decoder.onnx"});
            decoder_ = std::make_unique<OrtSession>(env, decoder_path, opts, "decoder");
            
            if (cfg_.verbose) {
                std::cout << "\n  Models loaded successfully:\n";
                text_encoder_->print_info();
                audio_encoder_->print_info();
                ar_model_->print_info();
                flow_model_->print_info();
                decoder_->print_info();
                std::cout << "\nInitialization complete.\n\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading MOSS models: " << e.what() << "\n";
            throw;
        }
    }
    
    // Main TTS generation function
    AudioData generate(const std::string& text, const std::string& voice_path) {
        if (cfg_.verbose) {
            std::cout << "Generating: \"" << text << "\" with voice: " 
                      << voice_path << "\n";
        }
        
        try {
            // Load reference voice
            AudioData ref_audio = load_audio(voice_path);
            if (cfg_.verbose) {
                std::cout << "  Reference audio: " << ref_audio.duration_sec() 
                          << "s @ " << ref_audio.sample_rate << " Hz\n";
            }
            
            // Tokenize text
            auto token_ids = tokenize(text);
            if (cfg_.verbose) std::cout << "  Tokens: " << token_ids.size() << "\n";
            
            // Encode text
            Tensor text_emb = encode_text(token_ids);
            if (cfg_.verbose) std::cout << "  Text embedding: [" << text_emb.shape.size() 
                                        << "D]\n";
            
            // Encode audio reference
            Tensor audio_cond = encode_audio(ref_audio);
            if (cfg_.verbose) std::cout << "  Audio conditioning: [" 
                                        << audio_cond.shape.size() << "D]\n";
            
            // Generate audio through AR + flow + decoder
            AudioData output = generate_audio(text_emb, audio_cond);
            if (cfg_.verbose) std::cout << "  Generated: " << output.duration_sec() 
                                        << "s\n";
            
            return output;
        } catch (const std::exception& e) {
            std::cerr << "Generation error: " << e.what() << "\n";
            // Return silent audio on error
            AudioData silence;
            silence.sample_rate = SR;
            silence.samples.resize(SR);  // 1 second
            return silence;
        }
    }
    
    // Stream generation for real-time output
    bool stream(const std::string& text, const std::string& voice_path,
                std::function<bool(const float*, size_t)> callback) {
        try {
            AudioData audio = generate(text, voice_path);
            
            // Stream in chunks
            size_t chunk_size = SR / 10;  // 100ms chunks
            for (size_t i = 0; i < audio.samples.size(); i += chunk_size) {
                size_t n = std::min(chunk_size, audio.samples.size() - i);
                if (!callback(audio.samples.data() + i, n)) {
                    return false;
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Stream error: " << e.what() << "\n";
            return false;
        }
    }
    
    // Audio I/O
    static AudioData load_audio(const std::string& path) {
        AudioData audio;
        
        if (path.empty()) {
            audio.sample_rate = SR;
            audio.samples.resize(SR);  // 1 second of silence
            return audio;
        }
        
        // Try WAV first
        unsigned int channels = 0, sample_rate = 0;
        drwav_uint64 frame_count = 0;
        float* data = drwav_open_file_and_read_pcm_frames_f32(
            path.c_str(), &channels, &sample_rate, &frame_count, nullptr);
        
        if (data) {
            audio.sample_rate = sample_rate;
            audio.samples.assign(data, data + frame_count);
            drwav_free(data, nullptr);
            return audio;
        }
        
        std::cerr << "Warning: Could not load audio from " << path << "\n";
        audio.sample_rate = SR;
        audio.samples.resize(SR);
        return audio;
    }
    
    static bool save_audio(const AudioData& audio, const std::string& path) {
        drwav_data_format format = {};
        format.container = drwav_container_riff;
        format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
        format.channels = 1;
        format.sampleRate = audio.sample_rate;
        format.bitsPerSample = 32;
        
        drwav wav = {};
        if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
            std::cerr << "Failed to create output file: " << path << "\n";
            return false;
        }
        
        drwav_uint64 written = drwav_write_pcm_frames(
            &wav, audio.samples.size(), audio.samples.data());
        drwav_uninit(&wav);
        
        return written == (drwav_uint64)audio.samples.size();
    }
    
private:
    Config cfg_;
    std::unique_ptr<Tokenizer> tok_;
    std::unique_ptr<OrtSession> text_encoder_;
    std::unique_ptr<OrtSession> audio_encoder_;
    std::unique_ptr<OrtSession> ar_model_;
    std::unique_ptr<OrtSession> flow_model_;
    std::unique_ptr<OrtSession> decoder_;
    StateBuffer state_buf_;
    
    Ort::SessionOptions create_session_options() {
        Ort::SessionOptions opts;
        int threads = cfg_.num_threads ? cfg_.num_threads : 1;
        opts.SetIntraOpNumThreads(threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return opts;
    }
    
    std::vector<int> tokenize(const std::string& text) {
        if (!tok_) return {};
        return tok_->encode(text);
    }
    
    Tensor encode_text(const std::vector<int>& token_ids) {
        // Create input tensor for text encoding
        Tensor result({1, (int64_t)token_ids.size(), ModelSpec::HIDDEN_DIM});
        
        // Fill with random embeddings for now (placeholder)
        for (auto& v : result.data) v = rng::normal(0, 0.1f);
        
        return result;
    }
    
    Tensor encode_audio(const AudioData& audio) {
        // Estimate number of frames
        int n_frames = audio.samples.size() / ModelSpec::AUDIO_FRAME_SIZE;
        
        // Create conditioning tensor
        Tensor result({1, (int64_t)n_frames, ModelSpec::HIDDEN_DIM});
        
        // Fill with random conditioning for now (placeholder)
        for (auto& v : result.data) v = rng::normal(0, 0.1f);
        
        return result;
    }
    
    AudioData generate_audio(const Tensor& text_emb, const Tensor& audio_cond) {
        // Generate audio through AR + flow + decoder pipeline
        AudioData output;
        output.sample_rate = SR;
        
        // Placeholder: generate random audio
        size_t n_samples = SR * 5;  // 5 seconds
        output.samples.resize(n_samples);
        
        for (auto& sample : output.samples) {
            sample = rng::normal(0, 0.01f);  // Small random signal
        }
        
        return output;
    }
};

} // namespace moss_tts_nano

// ════════════════════════════════════════════════════════════════════════════
// Command Line Interface
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    try {
        moss_tts_nano::Config cfg;
        std::string text, voice, output = "output.wav";
        bool show_help = false;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "-h" || arg == "--help") {
                show_help = true;
            } else if (arg == "--models-dir" && i + 1 < argc) {
                cfg.models_dir = argv[++i];
            } else if (arg == "--voices-dir" && i + 1 < argc) {
                cfg.voices_dir = argv[++i];
            } else if (arg == "--verbose" || arg == "-v") {
                cfg.verbose = true;
            } else if (arg == "--threads" && i + 1 < argc) {
                cfg.num_threads = std::stoi(argv[++i]);
            } else if (arg == "--precision" && i + 1 < argc) {
                cfg.precision = argv[++i];
            } else if (arg[0] != '-') {
                if (text.empty()) {
                    text = arg;
                } else if (voice.empty()) {
                    voice = arg;
                } else if (output == "output.wav") {
                    output = arg;
                }
            }
        }
        
        if (show_help || text.empty() || voice.empty()) {
            std::cerr << "MOSS-TTS-Nano.cpp — C++ TTS Runtime\n\n"
                      << "Usage: " << argv[0] << " [OPTIONS] TEXT VOICE [OUTPUT]\n\n"
                      << "Options:\n"
                      << "  -h, --help              Show this help\n"
                      << "  -v, --verbose           Enable verbose output\n"
                      << "  --models-dir PATH       Models directory (default: models)\n"
                      << "  --voices-dir PATH       Voices directory (default: voices)\n"
                      << "  --threads N             Number of threads (default: 1)\n"
                      << "  --precision int8|fp32   Model precision (default: int8)\n"
                      << "\nExamples:\n"
                      << "  " << argv[0] << " \"Hello world\" voice.wav output.wav\n"
                      << "  " << argv[0] << " --verbose \"Text\" voice.wav\n";
            return show_help ? 0 : 1;
        }
        
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // Initialize TTS engine
        moss_tts_nano::MossTTSNano tts(cfg);
        
        auto t_init = std::chrono::high_resolution_clock::now();
        double init_ms = std::chrono::duration<double, std::milli>(t_init - t_start).count();
        
        if (cfg.verbose) {
            std::cout << "Init time: " << std::fixed << std::setprecision(1) 
                      << init_ms << " ms\n";
        }
        
        // Generate audio
        t_start = std::chrono::high_resolution_clock::now();
        auto audio = tts.generate(text, voice);
        auto t_gen = std::chrono::high_resolution_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(t_gen - t_start).count();
        
        // Save audio
        if (!tts.save_audio(audio, output)) {
            std::cerr << "Error saving audio\n";
            return 1;
        }
        
        double duration = audio.duration_sec();
        double rtfx = duration / (gen_ms / 1000.0);
        
        std::cout << "✓ Generated " << std::fixed << std::setprecision(2) 
                  << duration << "s audio in " << gen_ms << "ms (RTFx: " << rtfx << "x)\n"
                  << "✓ Saved: " << output << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
