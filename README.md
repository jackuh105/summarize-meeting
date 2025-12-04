# Summarize Meeting

## Setup

### ffmpeg

The project requires `torchcodec` to be installed with `ffmpeg` 7.0.0. However, if you `ffmpeg` 8.0.0 installed as a dependency of other project, you can install `ffmpeg 7` separately:

```bash
brew install ffmpeg@7
```

And run the below command before you run the script:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_FALLBACK_LIBRARY_PATH"
```