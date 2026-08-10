#!/usr/bin/env bash
# =============================================================================
# prepare_vggface2.sh
# =============================================================================
# Stages the VGGFace2 (images) + MAAD-Face (attributes) dataset required to run
# the pipeline with dataset.name = "VGGFace2" (see configs/vggface2.yaml).
# Mirrors prepare_lfw.sh / prepare_celeba.sh: it tries several sources in order
# and verifies the final layout.
#
# VGGFace2 is LARGE (~36 GB train, 3.3M images / 9,131 identities). The original
# host (Oxford VGG) is offline, so this uses academic mirrors. MAAD-Face ships
# only the annotation CSV (per-image, 47 attributes) via pterhoer/MAAD-Face.
#
# Usage
# -----
#   bash scripts/prepare_vggface2.sh                # auto (mirror → Kaggle)
#   bash scripts/prepare_vggface2.sh --kaggle       # force Kaggle API path
#   bash scripts/prepare_vggface2.sh --skip-verify  # skip final layout check
#   bash scripts/prepare_vggface2.sh --no-smoke     # skip Stage-1 smoke test
#
# Mirror overrides (no external account needed — any direct link):
#   export VGGFACE2_IMAGES_URL='https://…/vggface2_train.tar.gz'
#   export MAADFACE_CSV_URL='https://…/MAAD_Face.csv'
#   bash scripts/prepare_vggface2.sh
#
# Kaggle slug override (an image mirror that unzips to <id>/*.jpg folders):
#   export VGGFACE2_KAGGLE_SLUG='hearfool/vggface2'
#
# Sources (in order of attempt)
# -----------------------------
#  1. Env mirror overrides (VGGFACE2_IMAGES_URL / MAADFACE_CSV_URL) via wget/curl
#  2. Kaggle mirror (VGGFACE2_KAGGLE_SLUG, needs ~/.kaggle/kaggle.json)
#  3. Manual-instructions fallback
#
# Expected layout after setup
# ---------------------------
#   data/vggface2/
#   ├── train/
#   │   ├── n000002/0001_01.jpg
#   │   └── ...
#   └── MAAD_Face.csv
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[prepare_vggface2]${RESET} $*"; }
success() { echo -e "${GREEN}[prepare_vggface2] ✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}[prepare_vggface2] ⚠${RESET}  $*"; }
die()     { echo -e "${RED}[prepare_vggface2] ✗${RESET} $*" >&2; exit 1; }

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
            sed -n '2,44p' "$0" | grep '^#' | sed 's/^# \?//'
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
# Stage into the SAME location the pipeline reads from, i.e. paths.data_root in
# configs/config.yaml, so training does not fail with "No VGGFace2 identities
# found". An explicit DATA_ROOT env var still wins.
CONFIG_DATA_ROOT="$(
    "${PYTHON}" -c "import yaml,sys; print(yaml.safe_load(open('configs/config.yaml'))['paths']['data_root'])" \
    2>/dev/null || echo ""
)"
DATA_ROOT="${DATA_ROOT:-${CONFIG_DATA_ROOT:-data}}"
info "Data root: ${DATA_ROOT}  (matches configs/config.yaml paths.data_root; override with DATA_ROOT=…)"
VGG_DIR="${DATA_ROOT}/vggface2"
IMG_DIR="${VGG_DIR}/train"
TAR_FILE="${VGG_DIR}/vggface2_train.tar.gz"
ATTR_FILE="${VGG_DIR}/MAAD_Face.csv"

# Kaggle mirror slug (image re-host that unzips to <id>/*.jpg identity folders).
VGGFACE2_KAGGLE_SLUG="${VGGFACE2_KAGGLE_SLUG:-hearfool/vggface2}"

# ── Step 0: Create directory structure ────────────────────────────────────────
info "Creating directory structure …"
mkdir -p "${VGG_DIR}"
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

# ── Helper: normalise the image layout to train/<id>/*.jpg ────────────────────
# Different mirrors nest images differently (e.g. vggface2/train/train/<id>,
# or VGG-Face2/data/train/<id>). Flatten so identities live directly under
# ${IMG_DIR}.
normalise_layout() {
    # If train/ already holds nXXXXXX/ folders, we are done.
    if [[ -d "${IMG_DIR}" ]] && \
       find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'n*' -print -quit \
       | grep -q .; then
        return 0
    fi
    # Search for a directory that directly contains nXXXXXX/ identity folders.
    local found
    found="$(
        find "${VGG_DIR}" -type d -name 'n0*' -print -quit 2>/dev/null \
        | xargs -r dirname 2>/dev/null || true
    )"
    if [[ -n "${found}" && "${found}" != "${IMG_DIR}" ]]; then
        info "Flattening image layout: ${found} → ${IMG_DIR}"
        rm -rf "${IMG_DIR}"
        mv "${found}" "${IMG_DIR}"
    fi
}

# ── Helper: Kaggle fallback for images ────────────────────────────────────────
download_via_kaggle() {
    info "Using Kaggle API to download VGGFace2 (${VGGFACE2_KAGGLE_SLUG}) …"

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

    info "Downloading ${VGGFACE2_KAGGLE_SLUG} from Kaggle (this is large) …"
    if ! kaggle datasets download -d "${VGGFACE2_KAGGLE_SLUG}" -p "${VGG_DIR}" --unzip; then
        warn "Kaggle download failed for ${VGGFACE2_KAGGLE_SLUG}."
        return 1
    fi
    normalise_layout
    return 0
}

# ── Step 2: Fetch images ──────────────────────────────────────────────────────
if [[ "${USE_KAGGLE}" == true ]]; then
    download_via_kaggle || die "Kaggle download failed. See messages above."
    normalise_layout
else
    # (a) Env mirror override first (fully account-free).
    if [[ ! -d "${IMG_DIR}" ]]; then
        fetch_url "${VGGFACE2_IMAGES_URL:-}" "${TAR_FILE}" "vggface2_train.tar.gz (mirror)" || true
    fi
    # (b) Extract if a tarball was fetched.
    if [[ -f "${TAR_FILE}" ]] && \
       ! find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'n*' -print -quit 2>/dev/null | grep -q .; then
        info "Extracting vggface2_train.tar.gz (this takes a while) …"
        tar -xzf "${TAR_FILE}" -C "${VGG_DIR}"
        normalise_layout
        success "Extraction complete."
    fi
    # (c) Kaggle auto-fallback if images still missing.
    if ! find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'n*' -print -quit 2>/dev/null | grep -q .; then
        warn "Images still missing — attempting automatic Kaggle fallback …"
        download_via_kaggle || true
    fi
fi

# ── Step 3: Fetch MAAD-Face attribute CSV ─────────────────────────────────────
# The CSV is hosted from the pterhoer/MAAD-Face GitHub repo (large, ~1 GB).
# Provide MAADFACE_CSV_URL for a direct mirror; otherwise fall back to gdown
# if the repo's Google-Drive id is supplied via MAADFACE_GDRIVE_ID.
if [[ ! -f "${ATTR_FILE}" ]]; then
    fetch_url "${MAADFACE_CSV_URL:-}" "${ATTR_FILE}" "MAAD_Face.csv (mirror)" || true
fi
if [[ ! -f "${ATTR_FILE}" && -n "${MAADFACE_GDRIVE_ID:-}" ]]; then
    info "Fetching MAAD_Face.csv via gdown (id=${MAADFACE_GDRIVE_ID}) …"
    "${PYTHON}" -m pip install gdown --quiet \
        --trusted-host pypi.org --trusted-host files.pythonhosted.org || true
    "${PYTHON}" -m gdown "${MAADFACE_GDRIVE_ID}" -O "${ATTR_FILE}" --fuzzy || \
        warn "gdown fetch failed for MAAD_Face.csv."
fi

# ── Step 4: Manual-instructions fallback ──────────────────────────────────────
if ! find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'n*' -print -quit 2>/dev/null | grep -q .; then
    echo ""
    warn "Automatic image download failed from all sources."
    echo -e "${BOLD}Fastest fixes:${RESET}"
    echo "  • Kaggle:  put kaggle.json at ~/.kaggle/kaggle.json, then"
    echo "             VGGFACE2_KAGGLE_SLUG='<owner/slug>' bash scripts/prepare_vggface2.sh --kaggle"
    echo "  • Mirror:  export VGGFACE2_IMAGES_URL='https://…/vggface2_train.tar.gz'"
    echo "             bash scripts/prepare_vggface2.sh"
    echo ""
    echo -e "${BOLD}Manual download:${RESET}"
    echo "  ① Images:      an academic VGGFace2 mirror (train split, ~36 GB)"
    echo "       → extract to: ${IMG_DIR}/<nXXXXXX>/*.jpg"
    echo "  ② Attributes:  https://github.com/pterhoer/MAAD-Face  (MAAD_Face.csv)"
    echo "       → save as:     ${ATTR_FILE}"
    echo ""
fi

# ── Step 5: Verify expected layout ────────────────────────────────────────────
if [[ "${SKIP_VERIFY}" == false ]]; then
    info "Verifying dataset layout …"
    ALL_OK=true

    if [[ -f "${ATTR_FILE}" ]]; then
        success "MAAD_Face.csv"
    else
        warn "MISSING (optional): MAAD_Face.csv — captions will fall back to synthetic."
    fi

    if [[ -d "${IMG_DIR}" ]]; then
        N_PEOPLE="$(find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
        N_IMG="$(find "${IMG_DIR}" -type f -name '*.jpg' | wc -l | tr -d ' ')"
        if (( N_PEOPLE >= 8000 )); then
            success "train/ — ${N_IMG} images across ${N_PEOPLE} identities (expected ~3.3M / 9 131)."
        elif (( N_IMG > 0 )); then
            warn "train/ — only ${N_IMG} images / ${N_PEOPLE} identities (mirror may be a subset)."
        else
            warn "train/ exists but contains no .jpg files."
            ALL_OK=false
        fi
    else
        warn "MISSING: train/ directory."
        ALL_OK=false
    fi

    if [[ "${ALL_OK}" == true ]]; then
        success "All required files verified."
    else
        warn "Some files are missing — re-run after fixing, or use --kaggle."
    fi
fi

# ── Step 6: Stage 1 smoke-test (uses the VGGFace2 override config) ────────────
if [[ "${RUN_SMOKE}" == false ]]; then
    info "Skipping Stage 1 smoke-test (--no-smoke)."
else
    if [[ ! -d "${IMG_DIR}" ]]; then
        warn "Skipping smoke-test: images not present."
    else
        info "Building merged VGGFace2 config and running Stage 1 smoke-test …"
        GEN_CFG="configs/_generated_vggface2.yaml"
        "${PYTHON}" scripts/merge_config.py \
            --base configs/config.yaml \
            --override configs/vggface2.yaml \
            --out "${GEN_CFG}"
        if "${PYTHON}" main.py --stage 1 --config "${GEN_CFG}"; then
            echo ""
            success "========================================"
            success " Stage 1 completed successfully on VGGFace2."
            success " Merged config: ${GEN_CFG}"
            success "========================================"
        else
            echo ""
            die "Stage 1 smoke-test failed. Check the output above for details."
        fi
    fi
fi
