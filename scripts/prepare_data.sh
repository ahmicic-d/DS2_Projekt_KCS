#!/usr/bin/env bash
# scripts/prepare_data.sh
# -----------------------
# Priprema podataka za treniranje:
#   1. Normalizacija transkripata (ISO-8859-2 → UTF-8, dijakritici)
#   2. Provjera audio datoteka
#   3. Kreiranje CSV manifesta (train/dev/test)
#
# Pokretanje: bash scripts/prepare_data.sh
# Opcije:
#   --audio_dir   putanja do WAV datoteka (default: data/raw/audio)
#   --text_dir    putanja do TXT datoteka (default: data/raw/text)
#   --train_ratio udio train skupa       (default: 0.8)
#   --dev_ratio   udio dev skupa         (default: 0.1)

set -e

# ── Defaults ────────────────────────────────────────────────────────
AUDIO_DIR="data/raw/audio"
TEXT_DIR="data/raw/text"
NORM_DIR="data/processed/normalized"
PROC_DIR="data/processed"
TRAIN_RATIO="0.8"
DEV_RATIO="0.1"
MIN_DUR="0.5"
MAX_DUR="20.0"
SEED="42"

# ── Parse argumenata ────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --audio_dir)   AUDIO_DIR="$2";   shift ;;
        --text_dir)    TEXT_DIR="$2";    shift ;;
        --train_ratio) TRAIN_RATIO="$2"; shift ;;
        --dev_ratio)   DEV_RATIO="$2";   shift ;;
        --min_dur)     MIN_DUR="$2";     shift ;;
        --max_dur)     MAX_DUR="$2";     shift ;;
        --seed)        SEED="$2";        shift ;;
        *) echo "Nepoznat argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Aktiviraj venv ako postoji ──────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    . .venv/Scripts/activate
fi

echo "=============================================="
echo "  DeepSpeech2 — Priprema podataka"
echo "=============================================="

# ── Provjera audio mape ─────────────────────────────────────────────
echo ""
echo "[1/4] Provjera podataka ..."

N_AUDIO=$(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l)
N_TEXT=$(ls "$TEXT_DIR"/*.txt 2>/dev/null | wc -l)

echo "  WAV datoteka:  $N_AUDIO"
echo "  TXT datoteka:  $N_TEXT"

if [ "$N_AUDIO" -eq 0 ]; then
    echo ""
    echo "GREŠKA: Nema WAV datoteka u $AUDIO_DIR"
    echo ""
    echo "Postavi podatke:"
    echo "  cp /putanja/do/audio/*.wav $AUDIO_DIR/"
    echo "  cp /putanja/do/text/*.txt  $TEXT_DIR/"
    exit 1
fi

if [ "$N_TEXT" -eq 0 ]; then
    echo "GREŠKA: Nema TXT datoteka u $TEXT_DIR"
    exit 1
fi

# ── Analiza audio datoteka ──────────────────────────────────────────
echo ""
echo "[2/4] Analiza audio datoteka ..."

python3 - <<PYEOF
import wave, os
from pathlib import Path

audio_dir = "$AUDIO_DIR"
files = sorted(Path(audio_dir).glob("*.wav"))

durations = []
problems = []

for fpath in files:
    try:
        with wave.open(str(fpath)) as w:
            sr  = w.getframerate()
            n   = w.getnframes()
            ch  = w.getnchannels()
            sw  = w.getsampwidth()
            dur = n / sr
            durations.append(dur)
            
            issues = []
            if sr != 16000: issues.append(f"sr={sr} (očekivano 16000)")
            if ch != 1:     issues.append(f"kanali={ch} (očekivano 1=mono)")
            if sw != 2:     issues.append(f"bit_depth={sw*8} (očekivano 16)")
            
            if issues:
                problems.append((fpath.name, ", ".join(issues)))
    except Exception as e:
        problems.append((fpath.name, f"GREŠKA: {e}"))

print(f"  Analizirano:      {len(durations)} snimaka")
print(f"  Ukupno trajanje:  {sum(durations)/60:.1f} min")
print(f"  Min / Avg / Max:  {min(durations):.2f}s / {sum(durations)/len(durations):.2f}s / {max(durations):.2f}s")

if problems:
    print(f"\n  UPOZORENJA ({len(problems)} datoteka):")
    for name, issue in problems[:10]:
        print(f"    {name}: {issue}")
    if len(problems) > 10:
        print(f"    ... i još {len(problems)-10} upozorenja")
    print("\n  SAVJET: Konverzija u ispravni format:")
    print("    ffmpeg -i ulaz.wav -ar 16000 -ac 1 -sample_fmt s16 izlaz.wav")
else:
    print("  Sve WAV datoteke su u ispravnom formatu!")
PYEOF

# ── Normalizacija transkripata ──────────────────────────────────────
echo ""
echo "[3/4] Normalizacija transkripata ..."

mkdir -p "$NORM_DIR"

python3 src/normalize.py \
    --input    "$TEXT_DIR" \
    --output   "$NORM_DIR" \
    --encoding "iso-8859-2" \
    --alphabet "configs/alphabet_hr.txt" \
    --verify

echo "  Normalizirani transkripti: $NORM_DIR"

# ── Kreiranje manifesta ─────────────────────────────────────────────
echo ""
echo "[4/4] Kreiranje CSV manifesta ..."

python3 src/make_manifest.py \
    --audio_dir   "$AUDIO_DIR" \
    --text_dir    "$NORM_DIR" \
    --output_dir  "$PROC_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --dev_ratio   "$DEV_RATIO" \
    --min_dur     "$MIN_DUR" \
    --max_dur     "$MAX_DUR" \
    --seed        "$SEED"

# ── Sažetak ─────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Priprema podataka završena!"
echo "=============================================="
echo ""
echo "Kreirani manifesti:"
echo "  data/processed/train/manifest.csv"
echo "  data/processed/dev/manifest.csv"
echo "  data/processed/test/manifest.csv"
echo ""
echo "Sljedeći korak:"
echo "  python src/train.py --config configs/training_config.json"
echo ""
