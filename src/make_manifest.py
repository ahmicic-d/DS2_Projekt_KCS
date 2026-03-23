import os
import csv
import wave
import random
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

def get_wav_info(wav_path: str) -> tuple:

    size = os.path.getsize(wav_path)
    try:
        with wave.open(wav_path, "r") as wf:
            frames    = wf.getnframes()
            framerate = wf.getframerate()
            duration  = frames / float(framerate)
    except Exception as e:
        log.warning(f"Ne mogu čitati WAV {wav_path}: {e}")
        duration = 0.0
    return duration, size


def load_pairs(audio_dir: str, text_dir: str,
               min_dur: float = 0.5, max_dur: float = 20.0) -> list:

    audio_path = Path(audio_dir)
    text_path  = Path(text_dir)

    wav_stems = {p.stem: p for p in audio_path.glob("*.wav")}
    txt_stems = {p.stem: p for p in text_path.glob("*.txt")}

    common = sorted(set(wav_stems.keys()) & set(txt_stems.keys()))
    log.info(f"WAV datoteka:  {len(wav_stems)}")
    log.info(f"TXT datoteka:  {len(txt_stems)}")
    log.info(f"Parova:        {len(common)}")

    missing_audio = set(txt_stems) - set(wav_stems)
    missing_text  = set(wav_stems) - set(txt_stems)
    if missing_audio:
        log.warning(f"Nedostaje audio za {len(missing_audio)} transkripata")
    if missing_text:
        log.warning(f"Nedostaje transkript za {len(missing_text)} snimaka")

    pairs = []
    skipped_empty    = 0
    skipped_duration = 0

    for stem in common:
        wav_file = str(wav_stems[stem])
        txt_file = txt_stems[stem]

        transcript = txt_file.read_text(encoding="utf-8").strip()
        if not transcript:
            skipped_empty += 1
            continue

        duration, size = get_wav_info(wav_file)

        if duration < min_dur or duration > max_dur:
            log.debug(f"  Preskačem {stem}: trajanje={duration:.2f}s")
            skipped_duration += 1
            continue

        pairs.append({
            "wav_path":   wav_file,
            "size":       size,
            "duration":   round(duration, 3),
            "transcript": transcript,
            "stem":       stem,
        })

    log.info(
        f"Parovi učitani: {len(pairs)}, "
        f"preskočeno (prazno)={skipped_empty}, "
        f"preskočeno (trajanje)={skipped_duration}"
    )
    return pairs


def split_data(pairs: list, train_ratio: float = 0.8,
               dev_ratio: float = 0.1, seed: int = 42) -> tuple:

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)

    train = shuffled[:n_train]
    dev   = shuffled[n_train : n_train + n_dev]
    test  = shuffled[n_train + n_dev :]

    log.info(f"Split: train={len(train)}, dev={len(dev)}, test={len(test)}")
    return train, dev, test


def write_manifest(pairs: list, output_path: str) -> None:

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["wav_filepath", "wav_filesize", "duration", "transcript"]
        )
        writer.writeheader()
        for p in pairs:
            writer.writerow({
                "wav_filepath": p["wav_path"],
                "wav_filesize": p["size"],
                "duration":     p["duration"],
                "transcript":   p["transcript"],
            })
    log.info(f"  Manifest spreman: {output_path} ({len(pairs)} unosa)")


def print_stats(pairs: list, name: str) -> None:

    if not pairs:
        return
    durations = [p["duration"] for p in pairs]
    words = [len(p["transcript"].split()) for p in pairs]
    total_min = sum(durations) / 60

    print(f"\n  {name}:")
    print(f"    Snimaka:          {len(pairs)}")
    print(f"    Ukupno trajanje:  {total_min:.1f} min")
    print(f"    Avg trajanje:     {sum(durations)/len(durations):.2f}s")
    print(f"    Min / Max:        {min(durations):.2f}s / {max(durations):.2f}s")
    print(f"    Avg br. riječi:   {sum(words)/len(words):.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kreiranje CSV manifesta za DeepSpeech2"
    )
    parser.add_argument("--audio_dir",  default="data/raw/audio",
                        help="Mapa s WAV datotekama")
    parser.add_argument("--text_dir",   default="data/processed/normalized",
                        help="Mapa s normaliziranim TXT transkriptima")
    parser.add_argument("--output_dir", default="data/processed",
                        help="Izlazna mapa za manifeste")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio",   type=float, default=0.1)
    parser.add_argument("--min_dur",  type=float, default=0.5,
                        help="Minimalno trajanje snimke (sekunde)")
    parser.add_argument("--max_dur",  type=float, default=20.0,
                        help="Maksimalno trajanje snimke (sekunde)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed za reproducibilnost")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pairs = load_pairs(
        args.audio_dir, args.text_dir,
        min_dur=args.min_dur, max_dur=args.max_dur
    )

    if not pairs:
        print("GREŠKA: Nema valjanih parova! Provjeri putanje i datoteke.")
        exit(1)

    train, dev, test = split_data(
        pairs,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed
    )

    out = Path(args.output_dir)
    write_manifest(train, str(out / "train" / "manifest.csv"))
    write_manifest(dev,   str(out / "dev"   / "manifest.csv"))
    write_manifest(test,  str(out / "test"  / "manifest.csv"))

    print(f"\n{'='*55}")
    print("Statistika skupova podataka:")
    print_stats(train, "TRAIN")
    print_stats(dev,   "DEV")
    print_stats(test,  "TEST")
    print(f"{'='*55}")
    print("\nManifestu su spremljeni u:", args.output_dir)
