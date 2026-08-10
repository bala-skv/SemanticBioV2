#!/usr/bin/env bash
# =============================================================================
# prepare_celeba.sh
# =============================================================================
# Downloads and verifies all CelebA files required by Stage 1.
#
# Usage
# -----
#   bash scripts/prepare_celeba.sh              # auto (gdown → Kaggle fallback)
#   bash scripts/prepare_celeba.sh --kaggle     # force Kaggle API path (see below)
#   bash scripts/prepare_celeba.sh --skip-verify # skip final file-count check
#
# Quota-free mirror override (no Google Drive, no Kaggle account needed):
#   export CELEBA_ZIP_URL='https://…/img_align_celeba.zip'   # any direct link
#   export CELEBA_ATTR_URL='https://…/list_attr_celeba.txt'  # (optional)
#   bash scripts/prepare_celeba.sh
#   → fetched with wget/curl before falling back to gdown.
#
# What this script does
# ---------------------
#  1. Creates data/celeba/ directory structure
#  2. Installs gdown if not present
#  3. Downloads all 4 CelebA files from Google Drive via gdown
#     (img_align_celeba.zip, list_attr_celeba.txt, identity_CelebA.txt,
#      list_eval_partition.txt)
#  4. Extracts img_align_celeba.zip
#  5. Verifies the expected file counts
#  6. Runs Stage 1 smoke-test (py -3 main.py --stage 1)
#
# Google Drive IDs (MMLAB / CUHK, official release)
# --------------------------------------------------
#   img_align_celeba.zip   0B7EVK8r0v71pZjFTYXZWM3FlRnM   (~1.34 GB)
#   list_attr_celeba.txt   0B7EVK8r0v71pblRyaVFSWGxPY0U
#   identity_CelebA.txt    1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS
#   list_eval_partition.txt 0B7EVK8r0v71pY0NSMzRuSXJEVkk
#
# Kaggle alternative (if Google Drive quota is exceeded)
# -------------------------------------------------------
#   1. Install kaggle CLI:  pip install kaggle
#   2. Place kaggle.json at ~/.kaggle/kaggle.json  (from kaggle.com/settings)
#   3. Run:  bash scripts/prepare_celeba.sh --kaggle
#
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[prepare_celeba]${RESET} $*"; }
success() { echo -e "${GREEN}[prepare_celeba] ✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}[prepare_celeba] ⚠${RESET}  $*"; }
die()     { echo -e "${RED}[prepare_celeba] ✗${RESET} $*" >&2; exit 1; }

# ── Argument parsing ─────────────────────────────────────────────────────────
USE_KAGGLE=false
SKIP_VERIFY=false
RUN_SMOKE=true
for arg in "$@"; do
    case "$arg" in
        --kaggle)       USE_KAGGLE=true ;;
        --skip-verify)  SKIP_VERIFY=true ;;
        --no-smoke)     RUN_SMOKE=false ;;
        -h|--help)
            sed -n '2,50p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

# ── Locate project root (script lives in scripts/, so go up one level) ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
info "Project root: ${PROJECT_ROOT}"

# ── Ensure Python / pip are available (also used to read the config) ──────────
PYTHON="${PYTHON:-$(command -v python3 || command -v python || echo "")}"
[[ -z "${PYTHON}" ]] && die "Python 3 not found. Install Python 3.9+ and try again."
PY_VER="$("${PYTHON}" --version 2>&1)"
info "Using Python: ${PYTHON} (${PY_VER})"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Stage into the SAME location the pipeline reads from (paths.data_root in
# configs/config.yaml); otherwise the download lands in ./data/celeba while
# training looks under (e.g.) /ssd_scratch/.../data/celeba and fails. An
# explicit DATA_ROOT env var still wins.
CONFIG_DATA_ROOT="$(
    "${PYTHON}" -c "import yaml; print(yaml.safe_load(open('configs/config.yaml'))['paths']['data_root'])" \
    2>/dev/null || echo ""
)"
DATA_ROOT="${DATA_ROOT:-${CONFIG_DATA_ROOT:-data}}"
info "Data root: ${DATA_ROOT}  (matches configs/config.yaml paths.data_root; override with DATA_ROOT=…)"
CELEBA_DIR="${DATA_ROOT}/celeba"
IMG_DIR="${CELEBA_DIR}/img_align_celeba"
ZIP_FILE="${CELEBA_DIR}/img_align_celeba.zip"
ATTR_FILE="${CELEBA_DIR}/list_attr_celeba.txt"
ID_FILE="${CELEBA_DIR}/identity_CelebA.txt"
PART_FILE="${CELEBA_DIR}/list_eval_partition.txt"

# Google Drive file IDs
GDRIVE_ZIP="0B7EVK8r0v71pZjFTYXZWM3FlRnM"
GDRIVE_ATTR="0B7EVK8r0v71pblRyaVFSWGxPY0U"
GDRIVE_ID="1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS"
GDRIVE_PART="0B7EVK8r0v71pY0NSMzRuSXJEVkk"

# ── Step 0: Create directory structure ────────────────────────────────────────
info "Creating directory structure …"
mkdir -p "${CELEBA_DIR}"
mkdir -p "${DATA_ROOT}/processed"
mkdir -p "checkpoints"
mkdir -p "results"
success "Directories ready."

# ── Step 2: Install gdown if needed ───────────────────────────────────────────
if ! "${PYTHON}" -c "import gdown" 2>/dev/null; then
    info "gdown not found — installing …"
    "${PYTHON}" -m pip install gdown \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --quiet
    success "gdown installed."
else
    success "gdown already available."
fi

# ── Helper: download one file via gdown (with retries + fuzzy) ─────────────────
gdown_file() {
    local file_id="$1"
    local dest="$2"
    local label="$3"
    local attempts=3

    if [[ -f "${dest}" ]]; then
        success "${label} already present — skipping."
        return 0
    fi

    info "Downloading ${label} …"
    local i
    for (( i = 1; i <= attempts; i++ )); do
        if "${PYTHON}" -m gdown \
                "https://drive.google.com/uc?id=${file_id}" \
                -O "${dest}" \
                --fuzzy --no-cookies; then
            success "${label} downloaded."
            return 0
        fi
        warn "gdown attempt ${i}/${attempts} failed for ${label}."
        rm -f "${dest}"   # remove any partial/HTML error file
        if (( i < attempts )); then sleep $(( i * 3 )); fi
    done

    warn "gdown failed for ${label} after ${attempts} attempts."
    warn "Google Drive quota may be exceeded. Will try mirror / Kaggle fallbacks."
    warn "Direct link: https://drive.google.com/file/d/${file_id}/view?usp=sharing"
    return 1
}

# ── Helper: download from a user-supplied mirror URL (env override) ────────────
# Set e.g. CELEBA_ZIP_URL to any direct-download link (HF resolve, S3, your own
# server) and the script will fetch it with wget or curl — no quota, no guessing.
mirror_file() {
    local url="$1"
    local dest="$2"
    local label="$3"

    [[ -z "${url}" ]] && return 1
    [[ -f "${dest}" ]] && { success "${label} already present — skipping."; return 0; }

    info "Downloading ${label} from mirror: ${url}"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${dest}" "${url}" && { success "${label} (mirror)."; return 0; }
    elif command -v curl &>/dev/null; then
        curl -fL --progress-bar -o "${dest}" "${url}" && { success "${label} (mirror)."; return 0; }
    else
        warn "Neither wget nor curl is available for mirror download."
    fi
    warn "Mirror download failed for ${label}."
    rm -f "${dest}"
    return 1
}

# ── Helper: download the full dataset via the Kaggle API ──────────────────────
download_via_kaggle() {
    info "Using Kaggle API to download CelebA …"

    if ! command -v kaggle &>/dev/null; then
        info "Kaggle CLI not found — installing …"
        "${PYTHON}" -m pip install kaggle --quiet \
            --trusted-host pypi.org --trusted-host files.pythonhosted.org
    fi

    if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
        warn "~/.kaggle/kaggle.json not found — cannot use Kaggle fallback."
        warn "Get your API token from https://www.kaggle.com/settings and place it there."
        return 1
    fi
    chmod 600 "${HOME}/.kaggle/kaggle.json"

    info "Downloading jessicali9530/celeba-dataset from Kaggle …"
    if ! kaggle datasets download \
            -d jessicali9530/celeba-dataset \
            -p "${CELEBA_DIR}" --unzip; then
        warn "Kaggle download failed."
        return 1
    fi

    # Kaggle nests the images at img_align_celeba/img_align_celeba/*.jpg.
    # Flatten by moving the *directory* (a glob mv of 202k files would exceed
    # ARG_MAX → "Argument list too long").
    local nested="${IMG_DIR}/img_align_celeba"
    if [[ -d "${nested}" ]]; then
        info "Flattening nested img_align_celeba/ …"
        mv "${nested}" "${CELEBA_DIR}/img_align_celeba_flat"
        rm -rf "${IMG_DIR}"
        mv "${CELEBA_DIR}/img_align_celeba_flat" "${IMG_DIR}"
    fi
    if [[ -d "${CELEBA_DIR}/celeba" ]]; then
        info "Flattening Kaggle sub-directory …"
        mv "${CELEBA_DIR}/celeba/"* "${CELEBA_DIR}/" 2>/dev/null || true
        rmdir "${CELEBA_DIR}/celeba" 2>/dev/null || true
    fi
    # Kaggle uses lowercase 'identity_celeba.txt'; normalise to expected name.
    if [[ ! -f "${ID_FILE}" && -f "${CELEBA_DIR}/identity_celeba.txt" ]]; then
        cp "${CELEBA_DIR}/identity_celeba.txt" "${ID_FILE}"
    fi

    local n_img
    n_img="$(find "${IMG_DIR}" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
    success "Kaggle download complete (${n_img} images in ${IMG_DIR})."
    return 0
}

# ── Step 3: Download or Kaggle ────────────────────────────────────────────────
if [[ "${USE_KAGGLE}" == true ]]; then
    # ── Kaggle path (explicit) ────────────────────────────────────────────────
    download_via_kaggle || die "Kaggle download failed. See messages above."

else
    # ── gdown path ────────────────────────────────────────────────────────────
    DOWNLOAD_FAILED=false

    # Optional mirror overrides (any direct-download URL). Tried before gdown.
    mirror_file "${CELEBA_ZIP_URL:-}"  "${ZIP_FILE}"  "img_align_celeba.zip (~1.34 GB)" || true
    mirror_file "${CELEBA_ATTR_URL:-}" "${ATTR_FILE}" "list_attr_celeba.txt"            || true
    mirror_file "${CELEBA_ID_URL:-}"   "${ID_FILE}"   "identity_CelebA.txt"             || true
    mirror_file "${CELEBA_PART_URL:-}" "${PART_FILE}" "list_eval_partition.txt"         || true

    gdown_file "${GDRIVE_ZIP}"  "${ZIP_FILE}"  "img_align_celeba.zip (~1.34 GB)" \
        || DOWNLOAD_FAILED=true
    gdown_file "${GDRIVE_ATTR}" "${ATTR_FILE}" "list_attr_celeba.txt" \
        || DOWNLOAD_FAILED=true
    gdown_file "${GDRIVE_ID}"   "${ID_FILE}"   "identity_CelebA.txt" \
        || DOWNLOAD_FAILED=true
    gdown_file "${GDRIVE_PART}" "${PART_FILE}" "list_eval_partition.txt" \
        || DOWNLOAD_FAILED=true

    # Auto-fallback: if the big zip is still missing, try Kaggle automatically
    # (only works if ~/.kaggle/kaggle.json is configured).
    if [[ ! -f "${ZIP_FILE}" && ! -d "${IMG_DIR}" ]]; then
        warn "Image zip still missing after gdown — attempting automatic Kaggle fallback …"
        if download_via_kaggle; then
            DOWNLOAD_FAILED=false
        fi
    fi

    if [[ "${DOWNLOAD_FAILED}" == true && ! -f "${ZIP_FILE}" && ! -d "${IMG_DIR}" ]]; then
        echo ""
        warn "One or more downloads failed (Google Drive quota likely exceeded)."
        echo -e "${BOLD}Fastest fix — use a quota-free source:${RESET}"
        echo "  • Kaggle mirror (recommended on HPC):"
        echo "      1) pip install kaggle  &&  put kaggle.json at ~/.kaggle/kaggle.json"
        echo "      2) bash scripts/prepare_celeba.sh --kaggle"
        echo "  • Or point the script at any direct mirror link you have access to:"
        echo "      export CELEBA_ZIP_URL='https://…/img_align_celeba.zip'"
        echo "      bash scripts/prepare_celeba.sh"
        echo ""
        echo -e "${BOLD}Manual Google Drive download (if quota resets):${RESET}"
        echo "  ① Images (~1.34 GB):"
        echo "       https://drive.google.com/file/d/${GDRIVE_ZIP}/view?usp=sharing"
        echo "       → save as: ${ZIP_FILE}"
        echo "  ② Attributes:"
        echo "       https://drive.google.com/file/d/${GDRIVE_ATTR}/view?usp=sharing"
        echo "       → save as: ${ATTR_FILE}"
        echo "  ③ Identity labels:"
        echo "       https://drive.google.com/file/d/${GDRIVE_ID}/view?usp=sharing"
        echo "       → save as: ${ID_FILE}"
        echo "  ④ Eval partition (optional):"
        echo "       https://drive.google.com/file/d/${GDRIVE_PART}/view?usp=sharing"
        echo "       → save as: ${PART_FILE}"
        echo ""
        # Continue: some files may have downloaded — try to extract and proceed
    fi
fi

# ── Step 4: Extract images zip ────────────────────────────────────────────────
if [[ -f "${ZIP_FILE}" && ! -d "${IMG_DIR}" ]]; then
    info "Extracting img_align_celeba.zip (202,599 files, ~1.4 GB extracted) …"
    unzip -q "${ZIP_FILE}" -d "${CELEBA_DIR}"
    success "Extraction complete → ${IMG_DIR}"
elif [[ -d "${IMG_DIR}" ]]; then
    success "img_align_celeba/ already extracted — skipping."
else
    warn "img_align_celeba.zip not found — images not extracted."
fi

# ── Step 5: Verify expected layout ────────────────────────────────────────────
if [[ "${SKIP_VERIFY}" == false ]]; then
    info "Verifying dataset layout …"
    ALL_OK=true

    check_file() {
        local path="$1"; local label="$2"
        if [[ -f "${path}" ]]; then
            success "${label}"
        else
            warn "MISSING: ${label} (${path})"
            ALL_OK=false
        fi
    }

    check_file "${ATTR_FILE}" "list_attr_celeba.txt"
    check_file "${ID_FILE}"   "identity_CelebA.txt"
    check_file "${PART_FILE}" "list_eval_partition.txt (optional)"

    if [[ -d "${IMG_DIR}" ]]; then
        IMG_COUNT="$(find "${IMG_DIR}" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')"
        if (( IMG_COUNT >= 202599 )); then
            success "img_align_celeba/  — ${IMG_COUNT} images found (expected 202,599)"
        elif (( IMG_COUNT > 0 )); then
            warn "img_align_celeba/ — only ${IMG_COUNT}/202,599 images found (extraction may be incomplete)"
            ALL_OK=false
        else
            warn "img_align_celeba/ directory exists but contains no .jpg files"
            ALL_OK=false
        fi
    else
        warn "MISSING: img_align_celeba/ directory"
        ALL_OK=false
    fi

    if [[ "${ALL_OK}" == true ]]; then
        success "All required files verified."
    else
        warn "Some files are missing — Stage 1 will fall back to synthetic attributes or fail."
        warn "Re-run after placing missing files, or use: bash scripts/prepare_celeba.sh --kaggle"
    fi
fi

# ── Step 6: Stage 1 smoke-test ────────────────────────────────────────────────
if [[ "${RUN_SMOKE}" == false ]]; then
    info "Skipping Stage 1 smoke-test (--no-smoke)."
else
    info "Running Stage 1 smoke-test …"
    if "${PYTHON}" main.py --stage 1; then
        echo ""
        success "========================================"
        success " Stage 1 completed successfully."
        success " CelebA data is ready for training."
        success "========================================"
    else
        echo ""
        die "Stage 1 smoke-test failed. Check the output above for details."
    fi
fi
