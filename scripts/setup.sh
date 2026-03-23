#!/usr/bin/env bash
# scripts/setup.sh
# ----------------
# Instalacija svih ovisnosti za DS2 projekt.
# Pokretanje: bash scripts/setup.sh

set -e  # Zaustavi na grešci

echo "=============================================="
echo "  DeepSpeech2 — HR Govor | Setup skripte"
echo "=============================================="

# ── Provjera Python verzije ────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3
REQUIRED_MINOR=8

echo ""
echo "[1/5] Provjera Python verzije ..."
python3 -c "
import sys
v = sys.version_info
if v.major < 3 or (v.major == 3 and v.minor < 8):
    print(f'GREŠKA: Potreban Python >= 3.8, pronađen {v.major}.{v.minor}')
    sys.exit(1)
print(f'OK: Python {v.major}.{v.minor}.{v.micro}')
"

# ── Virtualno okruženje ────────────────────────────────────────────
echo ""
echo "[2/5] Kreiranje virtualnog okruženja (.venv) ..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Virtualno okruženje kreirano: .venv/"
else
    echo "  .venv već postoji — preskačem kreiranje"
fi

# Aktiviraj
source .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate 2>/dev/null || {
    echo "UPOZORENJE: Ne mogu aktivirati .venv automatski."
    echo "Ručno: source .venv/bin/activate  (Linux/Mac)"
    echo "       .venv\\Scripts\\activate     (Windows)"
}

# ── Nadogradi pip ──────────────────────────────────────────────────
echo ""
echo "[3/5] Nadogradnja pip-a ..."
pip install --upgrade pip --quiet

# ── Instaliraj ovisnosti ───────────────────────────────────────────
echo ""
echo "[4/5] Instalacija ovisnosti iz requirements.txt ..."

# Detektuj GPU
if python3 -c "import subprocess; result = subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if result.returncode == 0 else 1)" 2>/dev/null; then
    echo "  GPU (NVIDIA) detektiran — instaliram PyTorch s CUDA podrškom"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "  Nema NVIDIA GPU-a — instaliram CPU verziju PyTorcha"
    pip install torch torchaudio --quiet
fi

# Ostale ovisnosti
pip install \
    speechbrain \
    jiwer \
    librosa \
    soundfile \
    pandas \
    numpy \
    tqdm \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    --quiet

echo "  Instalacija završena!"

# ── Provjera instalacije ───────────────────────────────────────────
echo ""
echo "[5/5] Provjera instalacije ..."

python3 -c "
import torch, torchaudio, librosa, pandas, numpy, jiwer
print(f'  torch:      {torch.__version__}')
print(f'  torchaudio: {torchaudio.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
print(f'  librosa:    {librosa.__version__}')
print(f'  pandas:     {pandas.__version__}')
print(f'  numpy:      {numpy.__version__}')
print(f'  jiwer:      {jiwer.__version__}')
"

echo ""
echo "=============================================="
echo "  Setup uspješno završen!"
echo "=============================================="
echo ""
echo "Sljedeći korak:"
echo "  bash scripts/prepare_data.sh"
echo ""
