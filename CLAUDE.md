# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fun-ASR is a multilingual automatic speech recognition (ASR) service built on FunAudioLLM's Fun-ASR-MLT-Nano model. The project provides FastAPI-based HTTP servers with batch processing optimization for high-throughput transcription of 31 languages.

**Key Architecture Characteristics:**
- MLT model uses autoregressive decoding (serial token prediction) vs. Paraformer's non-autoregressive parallel decoding
- Batch processing provides 3.5x speedup at optimal batch_size=6
- GPU utilization ~7% (architecture-limited, not a bug)
- Performance: ~770 hours/day throughput (single worker), RTF ~0.03

## Common Commands

### Running the Server

```bash
# Local deployment (recommended for development)
./run.sh local

# Docker deployment (production)
./run.sh docker

# Docker Compose deployment
./run.sh compose

# Direct Python execution
python server_batch.py
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Single file transcription
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"

# Batch transcription (6 files optimal)
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "files=@audio4.wav" \
  -F "files=@audio5.wav" \
  -F "files=@audio6.wav"

# Performance stats
curl http://localhost:8000/stats
```

### Multilingual Testing

```bash
# Download test data (10 languages, 5 samples each)
python tests/download_multilingual_test_data.py

# Run comprehensive multilingual batch tests
python tests/test_multilingual_batch.py

# Batch size performance comparison
python tests/test_batch_sizes.py
```

### Development Setup

```bash
# Install dependencies (Python 3.8+, 3.11 recommended)
pip install -r requirements.txt                    # Standard server
pip install -r requirements-batch-server.txt       # Batch server

# The startup scripts handle venv creation automatically
# Prefers venv311 > venv > env-3.8.8 in that order
```

## Code Architecture

### Server Implementation

**server_batch.py** - Batch-optimized server (recommended)
- Uses `model_batch.py` (custom FunASRNano wrapper)
- True batch processing support with optimal batch_size=6
- Delivers 3.5x speedup vs sequential processing
- Automatic CPU fallback with detailed CUDA error logging
- RESTful endpoints: `/transcribe`, `/transcribe_batch`, `/health`, `/stats`

### Model Layer

**model_batch.py** - Custom model wrapper for batch inference
- Registers as "FunASRNanoBatch" in FunASR tables
- Inherits from nn.Module, wraps AutoModelForCausalLM (LLM) + audio encoder
- Key methods:
  - `from_pretrained()` - Loads model with caching
  - `inference()` - Batch inference with proper padding/batching
  - Auto-detects model cache path (~/.cache/modelscope)
  - Supports LORA fine-tuning adapters

**model.py** - Base model implementation (used by demo scripts)

### Critical Architecture Details

**Why batch performance is limited:**
- MLT uses autoregressive decoding: `token[i] = decoder(audio, token[0:i])`
- Each token depends on previous tokens → serialization required
- Contrast with Paraformer's non-autoregressive: `all_tokens = decoder(audio)` in parallel
- Multi-worker scaling has diminishing returns due to this architectural constraint
- This is by design for multilingual support, not a bug to fix

**CUDA Fallback Logic** (server_batch.py:47-80):
- Automatically detects CUDA availability
- On CUDA error (esp. error 803), logs detailed version info and falls back to CPU
- Logs PyTorch version, CUDA availability, CUDA version, NVCC version
- Critical for deployment across different CUDA environments

### Configuration

**Environment Variables:**
```bash
MODEL_PATH=FunAudioLLM/Fun-ASR-MLT-Nano-2512  # Model identifier
DEVICE=cuda:0                                  # Device (cuda:0 or cpu)
USE_GPU=true                                   # Enable GPU mode
BATCH_SIZE=6                                   # Optimal batch size
MAX_BATCH_SIZE=10                              # Hard limit per request
NUM_WORKERS=1                                  # Uvicorn workers (Docker unstable >1)
```

**Model Auto-download:**
- Models cached to `~/.cache/modelscope/hub/`
- Auto-downloads if not found
- Cache path auto-detected via ModelScope API

### Deployment Modes

**Local (Recommended for Development):**
- Stable multi-worker support
- Easier debugging
- Direct file system access for models

**Docker (Production):**
- Dockerfile: Batch-optimized server using server_batch.py
- docker-compose.yml: Orchestration
- **Known Issue:** NUM_WORKERS>1 unstable in Docker, use NUM_WORKERS=1

**GPU Requirements:**
- Tesla T4 (15GB): 4 workers → ~340h/day
- Tesla L20 (44GB): 12 workers → ~510h/day (standard) / ~770h/day (batch)
- A10 (24GB): 6 workers → ~425h/day

### Testing Structure

**tests/** directory contains:
- `download_multilingual_test_data.py` - Downloads test samples from Hugging Face datasets
- `test_multilingual_batch.py` - Comprehensive 31-language batch performance test
- `test_batch_sizes.py` - Batch size optimization testing (1, 2, 6, 7, 10...)
- `performance_test.py` - RTF and throughput measurement
- `test_remote_*.py` - Remote server testing utilities

### Important Implementation Notes

**When modifying transcription logic:**
- Single file transcription internally calls batch endpoint with batch_size=1
- Batch results format: `results[0]` is list of dicts with "text" key
- Always clean up temp files in finally blocks
- Language parameter: "auto" for detection, or specific codes (zh, en, ja, etc.)

**When debugging performance:**
- Expected RTF at batch_size=6: ~0.03 (33x realtime)
- Per-file time at batch_size=6: ~0.31s for 10s audio
- GPU utilization ~7% is normal (architecture-limited)
- Check `/stats` endpoint for current configuration

**When handling errors:**
- CUDA errors trigger automatic CPU fallback (server_batch.py)
- Model loading failures are logged with detailed environment info
- Use try/finally for temp file cleanup
- Batch size validation: reject if > MAX_BATCH_SIZE

## Supported Languages (31 total)

Chinese, English, Japanese, Korean, Cantonese, Vietnamese, Indonesian, Thai, Malay, Filipino, Arabic, Hindi, Bulgarian, Croatian, Czech, Danish, Dutch, Estonian, Finnish, Greek, Hungarian, Irish, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Swedish

## Key Files Reference

- `server_batch.py` - Main production server (batch optimized)
- `model_batch.py` - Custom FunASRNano batch inference wrapper
- `model.py` - Base model implementation (demo reference)
- `run.sh` - Production startup script (handles venv, dependencies)
- `run_legacy.sh` - Legacy startup script (reference only)
- `requirements.txt` - Standard server dependencies
- `requirements-batch-server.txt` - Batch server dependencies
- `Dockerfile` - Production Docker image (uses server_batch.py)
- `docker-compose.yml` - Docker Compose configuration
- `BATCH_SERVER_README.md` - Batch server feature documentation
- `DEPLOY.md` - Deployment guide with performance comparisons
- `QUICK_START_BATCH_SERVER.md` - Quick installation guide

## Dependency Management

**Critical versions:**
- Python: 3.8+ (3.11 recommended for best compatibility)
- FunASR: >= 1.3.0 (batch server), == 1.2.9 (legacy server)
- PyTorch: >= 2.0.0
- torchaudio: >= 2.0.0 (must match PyTorch CUDA version)
- FastAPI: >= 0.95.0, < 0.100.0
- pydantic: >= 1.10.0, < 2.0.0 (v2 incompatible)
- transformers: >= 4.30.0

**Known conflicts:**
- pydantic v2.x breaks compatibility, must use v1.x
- CUDA version mismatch between PyTorch and system NVCC causes error 803
- ffmpeg required for audio format conversion (conda/apt/brew install)
