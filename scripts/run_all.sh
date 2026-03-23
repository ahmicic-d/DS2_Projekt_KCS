#!/usr/bin/env bash
# scripts/run_all.sh
# ------------------
# Pokretanje cijelog pipelinea od podataka do evaluacije.
# Koristi za potpuno automatsko izvršavanje svega.
#
# Pokretanje: bash scripts/run_all.sh
# Opcije:
#   --epochs N      Broj epoha treniranja (default: 100)
#   --device DEV    cpu / cuda / mps (auto-detect)
#   --debug         Brzi test s minimalnim podacima

set -e

EPOCHS=100
DEVICE=""
DEBUG=""
CONFIG="configs/training_config.json"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift ;;
        --device) DEVICE="--device $2"; shift ;;
        --debug)  DEBUG="--debug" ;;
        --config) CONFIG="$2"; shift ;;
        *) echo "Nepoznat argument: $1"; exit 1 ;;
    esac
    shift
done

echo "=============================================="
echo "  DeepSpeech2 — Kompletan pipeline"
echo "=============================================="
echo "  Epohe:  $EPOCHS"
echo "  Config: $CONFIG"
echo "  Debug:  ${DEBUG:-ne}"
echo "=============================================="

# Aktiviraj venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    . .venv/Scripts/activate
fi

# Ažuriraj broj epoha u konfig datoteci
python3 -c "
import json
with open('$CONFIG') as f: cfg = json.load(f)
cfg['training']['epochs'] = $EPOCHS
with open('$CONFIG', 'w') as f: json.dump(cfg, f, indent=2)
print(f'Konfig ažuriran: epochs={$EPOCHS}')
"

echo ""
echo "=== KORAK 1/3: Priprema podataka ==="
bash scripts/prepare_data.sh

echo ""
echo "=== KORAK 2/3: Treniranje ==="
python src/train.py \
    --config "$CONFIG" \
    $DEVICE \
    $DEBUG

echo ""
echo "=== KORAK 3/3: Evaluacija na test skupu ==="
python src/evaluate.py \
    --model   "models/best_model.pth" \
    --manifest "data/processed/test/manifest.csv" \
    --config  "$CONFIG" \
    --output  "results/test_results.csv" \
    $DEVICE

echo ""
echo "=============================================="
echo "  Pipeline završen!"
echo "  Rezultati: results/test_results.csv"
echo "  Modeli:    models/best_model.pth"
echo "=============================================="
