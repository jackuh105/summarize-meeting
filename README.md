# Summarize Meeting

A Python tool for automatic meeting transcription with speaker diarization. This project uses PyAnnote.audio for speaker detection and segmentation, combined with an OpenAI-compatible STT (Speech-to-Text) API to generate timestamped transcripts with speaker labels in SRT format.

## Features

- **Speaker Diarization**: Automatically detects and separates different speakers using PyAnnote.audio
- **Audio Format Support**: Handles various audio formats (MP3, WAV, etc.) with automatic conversion
- **Intelligent Segmentation**: Merges nearby segments from the same speaker for better readability
- **Parallel Processing**: Transcribes multiple segments concurrently for faster processing
- **SRT Output**: Generates standard SRT subtitle files with speaker labels
- **GPU Acceleration**: Supports MPS (Apple Silicon) and CUDA for faster processing

## Prerequisites

- Python 3.11 or higher
- FFmpeg 7.0.0
- Hugging Face account with access to PyAnnote models
- OpenAI-compatible STT API server

## Setup

### 1. Install FFmpeg 7

The project requires `torchcodec` to be installed with `ffmpeg` 7.0.0. If you have `ffmpeg` 8.0.0 installed as a dependency of other projects, you can install `ffmpeg 7` separately:

```bash
brew install ffmpeg@7
```

Set the environment variable before running the script:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_FALLBACK_LIBRARY_PATH"
```

Or use the provided `run.sh` script which sets this automatically.

### 2. Install Python Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
HF_TOKEN=your-huggingface-token
STT_BASE_API=http://localhost:8000/v1
```

- `HF_TOKEN`: Your Hugging Face access token (required for PyAnnote models)
- `STT_BASE_API`: Base URL of your OpenAI-compatible STT API server, the trailing `/v1` is required

## Usage

### Basic Usage

Process an audio file:

```bash
uv run main.py -i path/to/audio/file.mp3
# or
python main.py -i path/to/audio/file.mp3
```

Or using the run script (which handles FFmpeg environment):

```bash
./run.sh python main.py -i path/to/audio/file.mp3
```

## How It Works

1. **Audio Conversion**: Converts input audio to WAV format (16kHz, mono) if needed
2. **Speaker Diarization**: Uses PyAnnote.audio to detect and segment speakers
3. **Segment Merging**: Combines nearby segments from the same speaker (max gap: 2 seconds)
4. **Transcription**: Sends audio segments to STT API for transcription with configurable parallelism
5. **SRT Generation**: Creates an SRT file with timestamps and speaker labels

## Configuration

You can adjust the following constants in `main.py`:

- `PADDING`: Audio padding when splitting segments (default: 0.3 seconds)
- `MAX_WORKERS`: Maximum parallel STT API calls (default: 3)
- `STT_TIMEOUT`: API call timeout in seconds (default: None)

## Output

The script generates an SRT file with the following format:

```
1
00:00:00,200 --> 00:00:01,500
[SPEAKER_00] Hello, how are you today?

2
00:00:01,800 --> 00:00:03,900
[SPEAKER_01] I'm doing great, thanks for asking!
```

The output file is saved in the same directory as the input audio with a `.srt` extension.

## Requirements

See `pyproject.toml` for the complete list of dependencies. Main packages include:

- `pyannote-audio`: Speaker diarization
- `torchcodec`: Audio processing with FFmpeg integration
- `httpx`: Async HTTP client for STT API calls
- `tqdm`: Progress bars
- `python-dotenv`: Environment variable management

## Troubleshooting

### PyTorch Safe Globals Warning

If you encounter warnings about unsafe globals with PyTorch 2.6+, the code already includes the necessary safe globals configuration for PyAnnote models.

### FFmpeg Version Conflicts

Make sure you're using FFmpeg 7.0.0 by setting the `DYLD_FALLBACK_LIBRARY_PATH` environment variable or using the `run.sh` script.

### STT API Connection

Ensure your STT API server is running and accessible at the URL specified in `STT_BASE_API`.
