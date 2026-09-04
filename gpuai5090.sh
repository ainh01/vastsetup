#!/usr/bin/env bash
set -euo pipefail

# Supported environment:
#   - Ubuntu 22.04 or 24.04, or Debian 12
#   - root privileges
#   - apt-get package management
#   - an NVIDIA GPU/driver exposed inside this VPS or container
#
# The CUDA repository must already be configured if the OS repositories do not
# provide nvcc and the cuBLAS development files. Prefer a CUDA "devel" image on
# container platforms such as Vast.ai.

readonly MODEL_DIR="/workspace/models"
readonly LLAMA_DIR="/workspace/llama.cpp"
readonly WORKSPACE="/workspace"

# MODEL UPDATE: IQ4_XS quantization (faster than Q6_K)
readonly MODEL="${MODEL_DIR}/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
# REMOVE: vision projector (MMPROJ)
readonly DRAFT="${MODEL_DIR}/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
readonly BIN="${LLAMA_DIR}/build/bin/llama-server"
readonly LOG="/workspace/fastmtp-server.log"
readonly PID_FILE="/workspace/fastmtp-server.pid"
readonly PATCH_FILE="/workspace/HauhauCS-FastMTP-llama.cpp.patch"

readonly PATCH_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch"
readonly MODEL_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
# DRAFT URL (kept for speculative decoding)
readonly DRAFT_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"

readonly LLAMA_REPO="https://github.com/ggerganov/llama.cpp"
readonly LLAMA_COMMIT="4df29be4f4c3673f428170fda944a5b19f743bb8"

readonly SERVER_HOST="127.0.0.1"
readonly SERVER_PORT="3000"
readonly STARTUP_TIMEOUT="${FASTMTP_STARTUP_TIMEOUT:-600}"
readonly API_KEY="${FASTMTP_API_KEY:-ainhkey}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

info() {
    printf '==> %s\n' "$*"
}

if (( EUID != 0 )); then
    die "This setup must run as root, for example: sudo bash $0"
fi

if ! command -v apt-get >/dev/null 2>&1; then
    die "Unsupported system: apt-get is required. Use Ubuntu 22.04/24.04 or Debian 12."
fi

if [[ ! -r /etc/os-release ]]; then
    die "Cannot identify the operating system: /etc/os-release is missing or unreadable."
fi

# shellcheck disable=SC1091
. /etc/os-release

case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04|ubuntu:24.04|debian:12)
        ;;
    *)
        die "Unsupported operating system '${PRETTY_NAME:-unknown}'. Supported: Ubuntu 22.04/24.04 and Debian 12."
        ;;
esac

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

info "Updating APT metadata"
apt-get update

info "Installing base download, build, inspection, and process-management tools"
apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    coreutils \
    curl \
    git \
    cmake \
    build-essential \
    pkg-config \
    libc-bin \
    procps \
    sed \
    grep \
    findutils \
    gawk \
    iproute2 \
    util-linux \
    libssl-dev

update-ca-certificates

required_commands=(
    apt-cache
    awk
    cmake
    curl
    date
    dd
    git
    grep
    head
    ldconfig
    ldd
    mkdir
    mv
    nohup
    nproc
    od
    readlink
    rm
    sed
    sha256sum
    sleep
    sort
    ss
    stat
    tail
    tr
)

for required_command in "${required_commands[@]}"; do
    command -v "${required_command}" >/dev/null 2>&1 ||
        die "Required command '${required_command}' is unavailable after package installation."
done

[[ "${STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] ||
    die "FASTMTP_STARTUP_TIMEOUT must be a positive integer; got '${STARTUP_TIMEOUT}'."

[[ -n "${API_KEY}" ]] ||
    die "FASTMTP_API_KEY must not be empty."

mkdir -p "${WORKSPACE}" "${MODEL_DIR}"

exec 9>"/workspace/fastmtp-setup.lock"
if ! flock -n 9; then
    die "Another FastMTP setup process holds /workspace/fastmtp-setup.lock."
fi

package_available() {
    apt-cache show "$1" >/dev/null 2>&1
}

add_cuda_to_path() {
    local candidate
    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi
    for candidate in \
        /usr/local/cuda/bin \
        /usr/local/cuda-*/bin \
        /opt/cuda/bin
    do
        if [[ -x "${candidate}/nvcc" ]]; then
            PATH="${candidate}:${PATH}"
            export PATH
            return 0
        fi
    done
    return 1
}

install_first_available_package() {
    local description="$1"
    shift
    local package
    for package in "$@"; do
        if package_available "${package}"; then
            info "Installing ${description} from package '${package}'"
            if apt-get install -y --no-install-recommends "${package}"; then
                return 0
            fi
            warn "APT found '${package}', but its installation failed; trying another candidate if available."
        fi
    done
    return 1
}

version_at_least() {
    local actual="$1"
    local minimum="$2"
    local first
    first="$(
        printf '%s\n%s\n' "${minimum}" "${actual}" |
            sort -V |
            head -n 1
    )"
    [[ "${first}" == "${minimum}" ]]
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi is unavailable. The NVIDIA host driver/GPU is not exposed in this environment."
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    die "nvidia-smi exists but cannot access an NVIDIA GPU. Check GPU passthrough."
fi

info "Detected NVIDIA GPU and accessible host driver"
nvidia-smi -L

if ! add_cuda_to_path; then
    info "CUDA compiler not found; checking configured APT repositories"
    install_first_available_package \
        "CUDA compiler/toolkit" \
        cuda-toolkit-12-9 \
        cuda-toolkit \
        nvidia-cuda-toolkit ||
        die "No installable CUDA toolkit containing nvcc was found."
    add_cuda_to_path ||
        die "nvcc is still unavailable after installation."
fi

readonly NVCC="$(command -v nvcc)"
readonly NVCC_REAL="$(readlink -f "${NVCC}")"
readonly CUDA_BIN_DIR="${NVCC_REAL%/*}"
readonly CUDA_ROOT="${CUDA_BIN_DIR%/*}"

CUDA_RELEASE="$(
    "${NVCC}" --version |
        sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' |
        head -n 1
)"

[[ -n "${CUDA_RELEASE}" ]] ||
    die "nvcc is present at '${NVCC}', but its CUDA release could not be determined."

CUDA_MAJOR="${CUDA_RELEASE%%.*}"
CUDA_MINOR="${CUDA_RELEASE#*.}"
CUDA_PACKAGE_SUFFIX="${CUDA_MAJOR}-${CUDA_MINOR}"

cublas_header_present() {
    [[ -f "${CUDA_ROOT}/include/cublas_v2.h" ]] ||
    [[ -f "${CUDA_ROOT}/targets/x86_64-linux/include/cublas_v2.h" ]] ||
    [[ -f /usr/include/cublas_v2.h ]] ||
    [[ -f /usr/include/x86_64-linux-gnu/cublas_v2.h ]]
}

cublas_library_present() {
    ldconfig -p 2>/dev/null | grep -q 'libcublas\.so'
}

if ! cublas_header_present || ! cublas_library_present; then
    info "cuBLAS development/runtime files are incomplete; checking configured APT repositories"
    install_first_available_package \
        "cuBLAS development files" \
        "libcublas-dev-${CUDA_PACKAGE_SUFFIX}" \
        libcublas-dev \
        "cuda-toolkit-${CUDA_PACKAGE_SUFFIX}" \
        cuda-toolkit \
        nvidia-cuda-toolkit ||
        die "cuBLAS headers/libraries are missing."
    ldconfig
fi

cublas_header_present ||
    die "CUDA ${CUDA_RELEASE} is installed, but cublas_v2.h is missing."

cublas_library_present ||
    die "CUDA ${CUDA_RELEASE} is installed, but libcublas.so is absent from cache."

DRIVER_VERSION="$(
    nvidia-smi --query-gpu=driver_version --format=csv,noheader |
        head -n 1 |
        tr -d '[:space:]'
)"

[[ -n "${DRIVER_VERSION}" ]] ||
    die "The NVIDIA driver version could not be read through nvidia-smi."

case "${CUDA_MAJOR}" in
    13)
        MIN_DRIVER="580.00"
        ;;
    12)
        MIN_DRIVER="525.60.13"
        ;;
    11)
        MIN_DRIVER="450.80.02"
        ;;
    *)
        die "CUDA ${CUDA_RELEASE} is outside the validated CUDA 11/12/13 range."
        ;;
esac

version_at_least "${DRIVER_VERSION}" "${MIN_DRIVER}" ||
    die "NVIDIA driver ${DRIVER_VERSION} is too old."

info "CUDA validation passed: nvcc ${CUDA_RELEASE}, driver ${DRIVER_VERSION}, cuBLAS headers and libraries present"

valid_gguf() {
    local path="$1"
    local minimum_size=$((1024 * 1024))
    local size
    local magic
    [[ -f "${path}" ]] || return 1
    size="$(stat -c '%s' "${path}" 2>/dev/null)" || return 1
    (( size >= minimum_size )) || return 1
    magic="$(
        od -An -tx1 -N4 "${path}" 2>/dev/null |
            tr -d '[:space:]'
    )"
    [[ "${magic}" == "47475546" ]]
}

download_gguf() {
    local url="$1"
    local destination="$2"
    local label="$3"
    local partial="${destination}.partial"
    if valid_gguf "${destination}"; then
        info "${label} is already complete and valid"
        return 0
    fi
    if [[ -e "${destination}" ]]; then
        quarantine="${destination}.invalid.$(date +%Y%m%d%H%M%S)"
        warn "${label} destination exists but is not a valid GGUF; moving it to ${quarantine}"
        mv "${destination}" "${quarantine}"
    fi
    info "Downloading ${label}; interrupted transfers resume from ${partial}"
    if ! curl \
        --fail \
        --location \
        --continue-at - \
        --retry 5 \
        --retry-delay 3 \
        --retry-all-errors \
        --connect-timeout 30 \
        --output "${partial}" \
        "${url}"
    then
        if valid_gguf "${partial}"; then
            warn "Transfer returned an error, but the complete partial file passed GGUF validation."
        else
            die "${label} download failed. The resumable partial file remains at '${partial}'"
        fi
    fi
    valid_gguf "${partial}" ||
        die "${label} transfer completed, but '${partial}' is invalid."
    mv "${partial}" "${destination}"
    valid_gguf "${destination}" ||
        die "${label} failed validation after atomic rename."
    info "${label} downloaded and validated"
}

download_patch() {
    local partial="${PATCH_FILE}.partial"
    if [[ -s "${PATCH_FILE}" ]] && grep -q '^diff --git ' "${PATCH_FILE}"; then
        info "Patch download is already present and structurally valid"
        return 0
    fi
    if [[ -e "${PATCH_FILE}" ]]; then
        mv "${PATCH_FILE}" "${PATCH_FILE}.invalid.$(date +%Y%m%d%H%M%S)"
    fi
    info "Downloading the HauhauCS FastMTP patch"
    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 5 \
        --retry-delay 3 \
        --retry-all-errors \
        --connect-timeout 30 \
        --output "${partial}" \
        "${PATCH_URL}" ||
        die "Patch download failed."
    [[ -s "${partial}" ]] ||
        die "The downloaded patch is empty."
    grep -q '^diff --git ' "${partial}" ||
        die "The downloaded patch is invalid."
    mv "${partial}" "${PATCH_FILE}"
    info "Patch downloaded and structurally validated"
}

download_gguf "${MODEL_URL}" "${MODEL}" "Primary model"
download_gguf "${DRAFT_URL}" "${DRAFT}" "FastMTP draft model"
download_patch

readonly PATCH_SHA256="$(sha256sum "${PATCH_FILE}" | awk '{print $1}')"
readonly SOURCE_STATE_FILE="${LLAMA_DIR}/.fastmtp-source-state"
readonly BUILD_STATE_FILE="${LLAMA_DIR}/build/.fastmtp-build-state"

source_is_valid() {
    local head
    local diff_sha
    local expected_state
    local recorded_state
    [[ -d "${LLAMA_DIR}/.git" ]] || return 1
    [[ -f "${SOURCE_STATE_FILE}" ]] || return 1
    head="$(git -C "${LLAMA_DIR}" rev-parse HEAD 2>/dev/null)" || return 1
    [[ "${head}" == "${LLAMA_COMMIT}" ]] || return 1
    git -C "${LLAMA_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1 || return 1
    diff_sha="$(
        git -C "${LLAMA_DIR}" diff --binary "${LLAMA_COMMIT}" |
            sha256sum |
            awk '{print $1}'
    )" || return 1
    expected_state="$(
        printf 'commit=%s\npatch_sha256=%s\ndiff_sha256=%s' \
            "${LLAMA_COMMIT}" \
            "${PATCH_SHA256}" \
            "${diff_sha}"
    )"
    recorded_state="$(<"${SOURCE_STATE_FILE}")"
    [[ "${recorded_state}" == "${expected_state}" ]]
}

prepare_source() {
    local clone_dir="${LLAMA_DIR}.clone.$$"
    if source_is_valid; then
        info "Pinned and patched llama.cpp source tree is valid"
        return 0
    fi
    if [[ -e "${LLAMA_DIR}" ]]; then
        backup="${LLAMA_DIR}.invalid.$(date +%Y%m%d%H%M%S)"
        warn "Existing llama.cpp tree is incomplete, dirty, or mismatched; preserving it at ${backup}"
        mv "${LLAMA_DIR}" "${backup}"
    fi
    rm -rf "${clone_dir}"
    info "Cloning llama.cpp and checking out pinned commit ${LLAMA_COMMIT}"
    git clone --no-checkout "${LLAMA_REPO}" "${clone_dir}"
    git -C "${clone_dir}" checkout --detach "${LLAMA_COMMIT}"
    [[ "$(git -C "${clone_dir}" rev-parse HEAD)" == "${LLAMA_COMMIT}" ]] ||
        die "Git checkout failed."
    git -C "${clone_dir}" apply --check "${PATCH_FILE}" ||
        die "Patch does not apply cleanly."
    git -C "${clone_dir}" apply "${PATCH_FILE}"
    diff_sha="$(
        git -C "${clone_dir}" diff --binary "${LLAMA_COMMIT}" |
            sha256sum |
            awk '{print $1}'
    )"
    {
        printf 'commit=%s\n' "${LLAMA_COMMIT}"
        printf 'patch_sha256=%s\n' "${PATCH_SHA256}"
        printf 'diff_sha256=%s\n' "${diff_sha}"
    } >"${clone_dir}/.fastmtp-source-state"
    mv "${clone_dir}" "${LLAMA_DIR}"
    source_is_valid ||
        die "Source validation failed."
    info "HauhauCS patch applied exactly once to the pinned source revision"
}

prepare_source

readonly NVCC_STATE_SHA256="$(
    "${NVCC}" --version |
        sha256sum |
        awk '{print $1}'
)"

expected_build_state() {
    printf 'commit=%s\n' "${LLAMA_COMMIT}"
    printf 'patch_sha256=%s\n' "${PATCH_SHA256}"
    printf 'nvcc_sha256=%s\n' "${NVCC_STATE_SHA256}"
    printf 'cmake_options=-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release\n'
}

binary_runtime_is_valid() {
    local ldd_output
    [[ -x "${BIN}" ]] || return 1
    ldd_output="$(ldd "${BIN}" 2>&1)" || return 1
    if grep -q 'not found' <<<"${ldd_output}"; then
        return 1
    fi
    return 0
}

build_is_valid() {
    local expected
    local recorded
    source_is_valid || return 1
    binary_runtime_is_valid || return 1
    [[ -f "${BUILD_STATE_FILE}" ]] || return 1
    expected="$(expected_build_state)"
    recorded="$(<"${BUILD_STATE_FILE}")"
    [[ "${recorded}" == "${expected}" ]]
}

if build_is_valid; then
    info "Existing CUDA llama-server build is executable, linked, and matches the pinned patched source"
else
    info "Building the patched CUDA llama-server"
    rm -rf "${LLAMA_DIR}/build"
    CUDACXX="${NVCC}" cmake \
        -S "${LLAMA_DIR}" \
        -B "${LLAMA_DIR}/build" \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake \
        --build "${LLAMA_DIR}/build" \
        --config Release \
        -j"$(nproc)"
    [[ -x "${BIN}" ]] ||
        die "Build completed without producing executable."
    if ! binary_runtime_is_valid; then
        ldd "${BIN}" >&2 || true
        die "Build has missing shared libraries."
    fi
    expected_build_state >"${BUILD_STATE_FILE}"
    build_is_valid ||
        die "Build validation failed."
    info "CUDA llama-server build completed and validated"
fi

if command -v supervisorctl >/dev/null 2>&1; then
    info "Supervisor detected; disabling the template's ordinary llama service when present"
    supervisorctl stop llama >/dev/null 2>&1 || true
    LLAMA_CONF="$(
        grep -ril -m1 '^\[program:llama\]' /etc/supervisor/conf.d 2>/dev/null |
            head -n 1 ||
            true
    )"
    if [[ -n "${LLAMA_CONF}" ]]; then
        sed -i \
            -e 's/^autostart=.*/autostart=false/' \
            -e 's/^autorestart=.*/autorestart=false/' \
            "${LLAMA_CONF}"
        supervisorctl reread >/dev/null 2>&1 || true
        supervisorctl update >/dev/null 2>&1 || true
        supervisorctl stop llama >/dev/null 2>&1 || true
    fi
fi

pid_is_expected_server() {
    local pid="$1"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    [[ -e "/proc/${pid}/exe" ]] || return 1
    process_exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null)" || return 1
    expected_exe="$(readlink -f "${BIN}")" || return 1
    [[ "${process_exe}" == "${expected_exe}" ]]
}

stop_managed_server() {
    local pid
    [[ -f "${PID_FILE}" ]] || return 0
    pid="$(<"${PID_FILE}")"
    if ! pid_is_expected_server "${pid}"; then
        warn "Removing stale PID file."
        rm -f "${PID_FILE}"
        return 0
    fi
    kill "${pid}"
    for (( attempt = 0; attempt < 30; attempt++ )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${PID_FILE}"
            return 0
        fi
        sleep 1
    done
    kill -KILL "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
}

stop_managed_server

if [[ -n "$(ss -ltnH "sport = :${SERVER_PORT}" 2>/dev/null)" ]]; then
    die "TCP port ${SERVER_PORT} is already occupied."
fi

valid_gguf "${MODEL}" ||
    die "Primary model failed validation before startup."

valid_gguf "${DRAFT}" ||
    die "Draft model failed validation before startup."

source_is_valid ||
    die "llama.cpp source state became invalid before startup."

build_is_valid ||
    die "llama-server build state became invalid before startup."

server_args=(
    --model "${MODEL}"
    --spec-draft-model "${DRAFT}"
    --spec-draft-ngl all
    --spec-type draft-mtp
    --spec-draft-n-max 3
    --spec-draft-p-min 0
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --ctx-size 190000
    --parallel 1
    --batch-size 2048
    --ubatch-size 512
    --n-gpu-layers all
    --split-mode none
    --flash-attn on
    --no-mmap
    --jinja
    --reasoning on
    --reasoning-effort low
    --reasoning-preserve
    --reasoning-format deepseek
    --temp 1.0
    --top-k 20
    --top-p 0.95
    --min-p 0
    --presence-penalty 0
    --repeat-penalty 1.0
    --api-key "${API_KEY}"
)

: >"${LOG}"

info "Starting FastMTP llama-server; output is being written to ${LOG}"
nohup "${BIN}" "${server_args[@]}" >"${LOG}" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" >"${PID_FILE}"

sleep 2

if ! pid_is_expected_server "${server_pid}"; then
    rm -f "${PID_FILE}"
    tail -n 100 "${LOG}" >&2 || true
    die "llama-server exited or changed identity immediately after startup."
fi

deadline=$((SECONDS + STARTUP_TIMEOUT))
http_status="000"

while (( SECONDS < deadline )); do
    if ! pid_is_expected_server "${server_pid}"; then
        rm -f "${PID_FILE}"
        tail -n 100 "${LOG}" >&2 || true
        die "llama-server exited before becoming reachable."
    fi

    http_status="$(
        curl \
            --silent \
            --show-error \
            --output /dev/null \
            --write-out '%{http_code}' \
            --connect-timeout 2 \
            --max-time 5 \
            "http://${SERVER_HOST}:${SERVER_PORT}/health" 2>/dev/null ||
            true
    )"

    if [[ "${http_status}" =~ ^[1-5][0-9][0-9]$ ]]; then
        info "FastMTP server is reachable at http://${SERVER_HOST}:${SERVER_PORT} (health HTTP ${http_status})"
        
        # === NGROK INTEGRATION ===
        info "Setting up ngrok..."
        if ! command -v ngrok >/dev/null 2>&1; then
            apt-get update && apt-get install -y curl unzip >/dev/null 2>&1
            curl -fL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip -o /tmp/ngrok.zip >/dev/null 2>&1
            unzip -o /tmp/ngrok.zip -d /usr/local/bin >/dev/null 2>&1
            chmod +x /usr/local/bin/ngrok
        fi

        ngrok config add-authtoken '3Ip0GNgjLFPpOK2dMCdaUfhtbsd_3mGsMGUxFZ4ws1NGf9Tku' >/dev/null 2>&1
        ngrok http 127.0.0.1:3000 &
        ngrok_pid=$!
        sleep 3

        url=""
        for i in {1..15}; do
            url=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"\K[^"]+' | head -1)
            [[ -n "$url" ]] && break
            sleep 1
        done

        if [[ -z "$url" ]]; then
            info "Could not fetch ngrok URL from API."
        else
            api_url="${url}/chat/completions"
            echo ""
            echo "============================================="
            echo "BYOK API Endpoint Ready:"
            echo "============================================="
            echo "${api_url}"
            echo "============================================="
            echo "API Key: ${API_KEY}"
            echo "============================================="
            echo ""
        fi

        wait $ngrok_pid

        info "Log: ${LOG}"
        info "PID file: ${PID_FILE}"
        exit 0
    fi

    sleep 2
done

if pid_is_expected_server "${server_pid}"; then
    kill "${server_pid}" 2>/dev/null || true
fi
rm -f "${PID_FILE}"

printf '\nLast server log lines:\n' >&2
tail -n 100 "${LOG}" >&2 || true
die "Timed out after ${STARTUP_TIMEOUT}s waiting for llama-server."
