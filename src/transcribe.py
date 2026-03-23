"""
transcribe.py
-------------
Transkripcija jedne ili više WAV datoteka naučenim modelom.

Pokretanje:
    # Jedna datoteka:
    python src/transcribe.py --model models/best_model.pth --audio snimka.wav

    # Više datoteka:
    python src/transcribe.py --model models/best_model.pth --audio *.wav

    # Uz beam search:
    python src/transcribe.py --model models/best_model.pth --audio snimka.wav \
        --beam_search --beam_width 50
"""

import json
import logging
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transkripcija WAV datoteka DeepSpeech2 modelom"
    )
    parser.add_argument("--model",  required=True,
                        help="Putanja do .pth checkpoint datoteke")
    parser.add_argument("--audio",  nargs="+", required=True,
                        help="WAV datoteke za transkripciju")
    parser.add_argument("--config", default="configs/training_config.json",
                        help="Putanja do JSON konfig datoteke")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps")
    parser.add_argument("--beam_search", action="store_true",
                        help="Koristi beam search umjesto greedy dekodiranja")
    parser.add_argument("--beam_width",  type=int, default=10,
                        help="Širina beama za beam search (default: 10)")
    parser.add_argument("--output", default=None,
                        help="Spremi rezultate u TXT datoteku")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Konfig
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    # Uređaj
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log.info(f"Uređaj: {device}")
    log.info(f"Dekodiranje: {'beam search (width=' + str(args.beam_width) + ')' if args.beam_search else 'greedy'}")

    # Učitaj evaluator (koji sadrži model i preprocessor)
    evaluator = Evaluator(args.model, config, device)

    results = []
    print(f"\n{'='*65}")
    print(f"{'Datoteka':<35} {'Transkript'}")
    print(f"{'-'*65}")

    for wav_path in args.audio:
        wav_path = str(wav_path)
        if not Path(wav_path).exists():
            log.warning(f"Datoteka ne postoji: {wav_path}")
            continue

        try:
            transcript = evaluator.transcribe_file(
                wav_path,
                use_beam_search=args.beam_search,
                beam_width=args.beam_width,
            )
        except Exception as e:
            log.error(f"Greška pri transkripciji {wav_path}: {e}")
            transcript = "[GREŠKA]"

        fname = Path(wav_path).name
        print(f"{fname:<35} {transcript}")
        results.append({"file": wav_path, "transcript": transcript})

    print(f"{'='*65}\n")

    # Spremi rezultate
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"{r['file']}\t{r['transcript']}\n")
        log.info(f"Rezultati spremljeni: {args.output}")
