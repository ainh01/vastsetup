#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/workspace/models"
LLAMA_DIR="/workspace/llama.cpp"

MODEL="$MODEL_DIR/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
DRAFT="$MODEL_DIR/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
BIN="$LLAMA_DIR/build/bin/llama-server"
LOG="/workspace/fastmtp-server.log"
PATCH_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch"
MODEL_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
DRAFT_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"

# Stop and disable Vast's normal llama service: this template intentionally
# has no LLAMA_MODEL because we launch the patched standalone server below.
supervisorctl stop llama 2>/dev/null || true

LLAMA_CONF="$(grep -ril '^\[program:llama\]' /etc/supervisor/conf.d 2>/dev/null | head -n 1 || true)"
if [ -n "${LLAMA_CONF}" ]; then
  sed -i \
    -e 's/^autostart=.*/autostart=false/' \
    -e 's/^autorestart=.*/autorestart=false/' \
    "${LLAMA_CONF}"
  supervisorctl reread || true
  supervisorctl update || true
  supervisorctl stop llama || true
fi

mkdir -p "${MODEL_DIR}"

# Resume downloads if the instance was stopped during a download.
if [ ! -f "${MODEL}" ]; then
  curl -fL -C - --retry 5 --retry-delay 3 \
    -o "${MODEL}" \
    "${MODEL_URL}"
fi

if [ ! -f "${DRAFT}" ]; then
  curl -fL -C - --retry 5 --retry-delay 3 \
    -o "${DRAFT}" \
    "${DRAFT_URL}"
fi

# Build HauhauCS's required patched llama.cpp once per persistent workspace.
if [ ! -x "${BIN}" ]; then
  apt-get update
  apt-get install -y \
    git curl cmake build-essential libcublas-dev-12-9

  rm -rf "${LLAMA_DIR}"

  git clone https://github.com/ggerganov/llama.cpp "${LLAMA_DIR}"
  cd "${LLAMA_DIR}"

  git checkout 4df29be4f4c3673f428170fda944a5b19f743bb8

  curl -fL --retry 5 --retry-delay 3 \
    -o HauhauCS-FastMTP-llama.cpp.patch \
    "${PATCH_URL}"

  git apply --check HauhauCS-FastMTP-llama.cpp.patch
  git apply HauhauCS-FastMTP-llama.cpp.patch

  cmake -S . -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build build --config Release -j"$(nproc)"
fi

# Stop any old standalone server, then start the FastMTP server.
pkill -f "${BIN}" 2>/dev/null || true
sleep 2

nohup "${BIN}" \
  --model "${MODEL}" \
  --spec-draft-model "${DRAFT}" \
  --spec-draft-ngl all \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --host 127.0.0.1 \
  --port 18000 \
  --ctx-size 190000 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --n-gpu-layers all \
  --split-mode none \
  --flash-attn on \
  --no-mmap \
  --jinja \
  --reasoning on \
  --reasoning-effort xhigh \
  --reasoning-preserve \
  --reasoning-format deepseek \
  --temp 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0 \
  --presence-penalty 0 \
  --repeat-penalty 1.0 \
  --api-key ainhkey \
  > "${LOG}" 2>&1 &

echo "FastMTP server launched at 127.0.0.1:18000"
