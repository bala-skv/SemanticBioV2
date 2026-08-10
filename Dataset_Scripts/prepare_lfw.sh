#!/usr/bin/env bash
# =============================================================================
# prepare_lfw.sh
# =============================================================================
# Downloads and verifies the LFW (Labeled Faces in the Wild) dataset plus the
# Kumar et al. attribute scores required to run the pipeline with
# dataset.name = "LFW" (see configs/lfw.yaml). Mirrors prepare_celeba.sh: it
# tries several sources in order and verifies the final layout.
#
# Usage
# -----
#   bash scripts/prepare_lfw.sh                # auto (official → Kaggle)
#   bash scripts/prepare_lfw.sh --kaggle       # force Kaggle API path
#   bash scripts/prepare_lfw.sh --skip-verify  # skip final layout check
#   bash scripts/prepare_lfw.sh --no-smoke     # skip Stage-1 smoke test
#
# Mirror overrides (no external account needed — any direct link):
#   export LFW_IMAGES_URL='https://…/lfw-deepfunneled.tgz'
#   export LFW_ATTR_URL='https://…/lfw_attributes.txt'
#   bash scripts/prepare_lfw.sh
#
# Sources (in order of attempt)
# -----------------------------
#  1. Env mirror overrides (LFW_IMAGES_URL / LFW_ATTR_URL) via wget/curl
#  2. Official UMass:   https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
#     Attributes (Columbia CAVE):
#       https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
#  3. Kaggle mirror:    jessicali9530/lfw-dataset   (needs ~/.kaggle/kaggle.json)
#
# Expected layout after setup
# ---------------------------
#   data/lfw/
#   ├── lfw-deepfunneled/
#   │   ├── Aaron_Eckhart/Aaron_Eckhart_0001.jpg
#   │   └── ...
#   └── lfw_attributes.txt
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[prepare_lfw]${RESET} $*"; }
success() { echo -e "${GREEN}[prepare_lfw] ✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}[prepare_lfw] ⚠${RESET}  $*"; }
die()     { echo -e "${RED}[prepare_lfw] ✗${RESET} $*" >&2; exit 1; }

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
            sed -n '2,40p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

# ── Locate project root (script lives in scripts/, so go up one level) ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
info "Project root: ${PROJECT_ROOT}"

# ── Ensure Python is available (for config parsing / smoke test / Kaggle) ─────
PYTHON="${PYTHON:-$(command -v python3 || command -v python || echo "")}"
[[ -z "${PYTHON}" ]] && die "Python 3 not found. Install Python 3.9+ and try again."
info "Using Python: ${PYTHON} ($("${PYTHON}" --version 2>&1))"

# ── Paths ─────────────────────────────────────────────────────────────────────
# IMPORTANT: stage the data into the SAME location the pipeline reads from,
# i.e. paths.data_root in configs/config.yaml. Otherwise the download lands in
# ./data/lfw while training looks under (e.g.) /ssd_scratch/.../data/lfw and
# fails with "No LFW identities found". We therefore default DATA_ROOT to the
# config's paths.data_root; an explicit DATA_ROOT env var still wins.
CONFIG_DATA_ROOT="$(
    "${PYTHON}" -c "import yaml,sys; print(yaml.safe_load(open('configs/config.yaml'))['paths']['data_root'])" \
    2>/dev/null || echo ""
)"
DATA_ROOT="${DATA_ROOT:-${CONFIG_DATA_ROOT:-data}}"
info "Data root: ${DATA_ROOT}  (matches configs/config.yaml paths.data_root; override with DATA_ROOT=…)"
LFW_DIR="${DATA_ROOT}/lfw"
IMG_DIR="${LFW_DIR}/lfw-deepfunneled"
TGZ_FILE="${LFW_DIR}/lfw-deepfunneled.tgz"
ATTR_FILE="${LFW_DIR}/lfw_attributes.txt"

# Canonical source URLs
OFFICIAL_IMAGES_URL="https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
OFFICIAL_ATTR_URL="https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"

# ── Step 0: Create directory structure ────────────────────────────────────────
info "Creating directory structure …"
mkdir -p "${LFW_DIR}"
mkdir -p "${DATA_ROOT}/processed"
mkdir -p "checkpoints"
mkdir -p "results"
success "Directories ready."

# ── Helper: download from a direct URL via wget or curl ───────────────────────
fetch_url() {
    local url="$1"; local dest="$2"; local label="$3"
    [[ -z "${url}" ]] && return 1
    [[ -f "${dest}" ]] && { success "${label} already present — skipping."; return 0; }

    info "Downloading ${label} …"
    info "  ${url}"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${dest}" "${url}" && { success "${label}."; return 0; }
    elif command -v curl &>/dev/null; then
        curl -fL --progress-bar -o "${dest}" "${url}" && { success "${label}."; return 0; }
    else
        warn "Neither wget nor curl is available."
    fi
    warn "Download failed for ${label}."
    rm -f "${dest}"
    return 1
}

# ── Helper: Kaggle fallback (jessicali9530/lfw-dataset) ───────────────────────
download_via_kaggle() {
    info "Using Kaggle API to download LFW …"

    if ! command -v kaggle &>/dev/null; then
        info "Kaggle CLI not found — installing …"
        "${PYTHON}" -m pip install kaggle --quiet \
            --trusted-host pypi.org --trusted-host files.pythonhosted.org || true
    fi
    if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
        warn "~/.kaggle/kaggle.json not found — cannot use Kaggle fallback."
        warn "Get your API token from https://www.kaggle.com/settings and place it there."
        return 1
    fi
    chmod 600 "${HOME}/.kaggle/kaggle.json"

    info "Downloading jessicali9530/lfw-dataset from Kaggle …"
    if ! kaggle datasets download -d jessicali9530/lfw-dataset -p "${LFW_DIR}" --unzip; then
        warn "Kaggle download failed."
        return 1
    fi

    # The Kaggle archive nests images under lfw-deepfunneled/lfw-deepfunneled/.
    local nested="${IMG_DIR}/lfw-deepfunneled"
    if [[ -d "${nested}" ]]; then
        info "Flattening nested lfw-deepfunneled/ …"
        mv "${nested}" "${LFW_DIR}/_lfw_flat"
        rm -rf "${IMG_DIR}"
        mv "${LFW_DIR}/_lfw_flat" "${IMG_DIR}"
    fi
    # Kaggle also ships the attribute file under a couple of possible names.
    if [[ ! -f "${ATTR_FILE}" ]]; then
        for cand in "${LFW_DIR}"/lfw_attributes*.txt "${LFW_DIR}"/*attributes*.txt; do
            [[ -f "${cand}" ]] && { cp "${cand}" "${ATTR_FILE}"; break; }
        done
    fi
    return 0
}

# ── Step 2: Fetch images + attributes ─────────────────────────────────────────
if [[ "${USE_KAGGLE}" == true ]]; then
    download_via_kaggle || die "Kaggle download failed. See messages above."
else
    # (a) Env mirror overrides first (fully account-free).
    fetch_url "${LFW_IMAGES_URL:-}" "${TGZ_FILE}"  "lfw-deepfunneled.tgz (mirror)" || true
    fetch_url "${LFW_ATTR_URL:-}"   "${ATTR_FILE}" "lfw_attributes.txt (mirror)"   || true

    # (b) Official UMass / Columbia sources.
    if [[ ! -d "${IMG_DIR}" && ! -f "${TGZ_FILE}" ]]; then
        fetch_url "${OFFICIAL_IMAGES_URL}" "${TGZ_FILE}" "lfw-deepfunneled.tgz (~111 MB)" || true
    fi
    fetch_url "${OFFICIAL_ATTR_URL}" "${ATTR_FILE}" "lfw_attributes.txt (Kumar et al.)" || true

    # (c) Kaggle auto-fallback if images still missing.
    if [[ ! -d "${IMG_DIR}" && ! -f "${TGZ_FILE}" ]]; then
        warn "Images still missing — attempting automatic Kaggle fallback …"
        download_via_kaggle || true
    fi
fi

# ── Step 3: Extract images tarball ────────────────────────────────────────────
if [[ -f "${TGZ_FILE}" && ! -d "${IMG_DIR}" ]]; then
    info "Extracting lfw-deepfunneled.tgz …"
    tar -xzf "${TGZ_FILE}" -C "${LFW_DIR}"
    success "Extraction complete → ${IMG_DIR}"
elif [[ -d "${IMG_DIR}" ]]; then
    success "lfw-deepfunneled/ already extracted — skipping."
else
    warn "lfw-deepfunneled.tgz not found — images not extracted."
fi

# ── Step 4: Manual-instructions fallback ──────────────────────────────────────
if [[ ! -d "${IMG_DIR}" ]]; then
    echo ""
    warn "Automatic image download failed from all sources."
    echo -e "${BOLD}Fastest fixes:${RESET}"
    echo "  • Kaggle:  put kaggle.json at ~/.kaggle/kaggle.json, then"
    echo "             bash scripts/prepare_lfw.sh --kaggle"
    echo "  • Mirror:  export LFW_IMAGES_URL='https://…/lfw-deepfunneled.tgz'"
    echo "             bash scripts/prepare_lfw.sh"
    echo ""
    echo -e "${BOLD}Manual download:${RESET}"
    echo "  ① Images (~111 MB):  ${OFFICIAL_IMAGES_URL}"
    echo "       → extract to: ${IMG_DIR}"
    echo "  ② Attributes:        ${OFFICIAL_ATTR_URL}"
    echo "       → save as:     ${ATTR_FILE}"
    echo ""
fi

# ── Step 5: Verify expected layout ────────────────────────────────────────────
if [[ "${SKIP_VERIFY}" == false ]]; then
    info "Verifying dataset layout …"
    ALL_OK=true

    if [[ -f "${ATTR_FILE}" ]]; then
        success "lfw_attributes.txt"
    else
        warn "MISSING (optional): lfw_attributes.txt — captions will fall back to synthetic."
    fi

    if [[ -d "${IMG_DIR}" ]]; then
        N_PEOPLE="$(find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
        N_IMG="$(find "${IMG_DIR}" -type f -name '*.jpg' | wc -l | tr -d ' ')"
        if (( N_IMG >= 13000 )); then
            success "lfw-deepfunneled/ — ${N_IMG} images across ${N_PEOPLE} people (expected ~13 233 / 5 749)."
        elif (( N_IMG > 0 )); then
            warn "lfw-deepfunneled/ — only ${N_IMG} images / ${N_PEOPLE} people (extraction may be incomplete)."
            ALL_OK=false
        else
            warn "lfw-deepfunneled/ exists but contains no .jpg files."
            ALL_OK=false
        fi
    else
        warn "MISSING: lfw-deepfunneled/ directory."
        ALL_OK=false
    fi

    if [[ "${ALL_OK}" == true ]]; then
        success "All required files verified."
    else
        warn "Some files are missing — re-run after fixing, or use --kaggle."
    fi
fi

# ── Step 6: Stage 1 smoke-test (uses the LFW override config) ─────────────────
if [[ "${RUN_SMOKE}" == false ]]; then
    info "Skipping Stage 1 smoke-test (--no-smoke)."
else
    if [[ ! -d "${IMG_DIR}" ]]; then
        warn "Skipping smoke-test: images not present."
    else
        info "Building merged LFW config and running Stage 1 smoke-test …"
        GEN_CFG="configs/_generated_lfw.yaml"
        "${PYTHON}" scripts/merge_config.py \
            --base configs/config.yaml \
            --override configs/lfw.yaml \
            --out "${GEN_CFG}"
        if "${PYTHON}" main.py --stage 1 --config "${GEN_CFG}"; then
            echo ""
            success "========================================"
            success " Stage 1 completed successfully on LFW."
            success " Merged config: ${GEN_CFG}"
            success "========================================"
        else
            echo ""
            die "Stage 1 smoke-test failed. Check the output above for details."
        fi
    fi
fi
