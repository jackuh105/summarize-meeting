#!/bin/bash

# 專案啟動腳本 - 自動設置 FFmpeg 7 環境

export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_FALLBACK_LIBRARY_PATH"

echo "🎯 使用 FFmpeg 7 環境運行專案..."
echo "環境變數: DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH"
echo ""

uv run "$@"
