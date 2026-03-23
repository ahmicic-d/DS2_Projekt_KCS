"""
normalize.py
------------
Normalizacija transkripata iz ISO-8859-2 kodne stranice u UTF-8.

Rješava problem ASCII supstitucija za hrvatska slova:
    {  →  š
    ~  →  č
    `  →  ž
    ^  →  đ
    d` →  dž
    <sil>, <ah>, <eee> ... → uklonjeno ili zamijenjeno

Pokretanje:
    python src/normalize.py --input data/raw/text --output data/processed/normalized
"""

import os
import re
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Mapa: ASCII supstitucija → ispravni UTF-8 znak
# VAŽNO: 'd`' mora biti prije '`' u zamjeni!
# ──────────────────────────────────────────────────────────────────
ASCII_MAP = {
    "d`": "dž",   # dvostruki znak — mora biti prvi!
    "{":  "š",
    "~":  "č",
    "`":  "ž",
    "^":  "ć",    # ISPRAVNO: ^ = ć (fonem: cc), ne đ!
    "}":  "đ",    # ISPRAVNO: } = đ (fonem: dz)
}

# Posebni tokeni koji se uklanjaju
NOISE_TOKENS = re.compile(
    r"<sil>|<[a-z]+>|\[[a-z]+\]|{[a-z]+}",
    re.IGNORECASE
)


def normalize_transcript(raw: str) -> str:
    """
    Pretvori jedan sirovi transkript u normaliziran UTF-8 string.

    Koraci:
    1. Ukloni specijalne tokene (<sil>, <ah> itd.)
    2. Zamijeni ASCII supstitucije s ispravnim slovima:
         d` → dž  |  { → š  |  ~ → č  |  ` → ž  |  ^ → ć  |  } → đ
    3. Pretvori u mala slova
    4. Ukloni višestruke razmake
    5. Trim
    """
    text = raw.strip()

    # 1. Ukloni noise tokene
    text = NOISE_TOKENS.sub(" ", text)

    # 2. Zamijeni ASCII supstitucije (d` mora biti prije `)
    for ascii_sub, correct in ASCII_MAP.items():
        text = text.replace(ascii_sub, correct)

    # 3. Mala slova
    text = text.lower()

    # 4. Normalizacija razmaka
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Ukloni znakove koji nisu slova, apostrof, razmak
    #    (ali zadrži sva hrvatska slova)
    allowed = re.compile(
        r"[^a-zšđčćžabcčćdđefghijklmnoprsštuvzž' ]",
        re.UNICODE
    )
    text = allowed.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_directory(input_dir: str, output_dir: str,
                         encoding: str = "iso-8859-2") -> dict:
    """
    Normalizira sve .txt datoteke u mapi input_dir i
    sprema rezultate u output_dir (UTF-8).

    Vraća: {'ok': int, 'empty': int, 'error': int}
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "empty": 0, "error": 0}
    txt_files = sorted(input_path.glob("*.txt"))

    if not txt_files:
        log.warning(f"Nema .txt datoteka u: {input_dir}")
        return stats

    log.info(f"Obrađujem {len(txt_files)} datoteka iz '{input_dir}' ...")

    for fpath in txt_files:
        try:
            raw = fpath.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            # Pokušaj UTF-8 ako ISO-8859-2 ne radi
            try:
                raw = fpath.read_text(encoding="utf-8").strip()
            except Exception as e:
                log.error(f"  Greška pri čitanju {fpath.name}: {e}")
                stats["error"] += 1
                continue
        except Exception as e:
            log.error(f"  Greška {fpath.name}: {e}")
            stats["error"] += 1
            continue

        normalized = normalize_transcript(raw)

        if not normalized:
            log.debug(f"  Prazan transkript: {fpath.name} (raw: {repr(raw)})")
            stats["empty"] += 1
            # Spremi praznu datoteku da bismo mogli pratiti koji su prazni
            out_file = output_path / fpath.name
            out_file.write_text("", encoding="utf-8")
            continue

        out_file = output_path / fpath.name
        out_file.write_text(normalized, encoding="utf-8")
        stats["ok"] += 1

    log.info(
        f"Normalizacija završena: OK={stats['ok']}, "
        f"praznih={stats['empty']}, grešaka={stats['error']}"
    )
    return stats


def verify_charset(normalized_dir: str, alphabet_path: str) -> list:
    """
    Provjeri koriste li normalizirani transkripti samo znakove
    koji se nalaze u alphabet_hr.txt. Vrati listu nepoznatih znakova.
    """
    # Učitaj abecedu
    with open(alphabet_path, encoding="utf-8") as f:
        alphabet = set(f.read())
    alphabet.discard("\n")

    unknown = {}
    for fpath in sorted(Path(normalized_dir).glob("*.txt")):
        text = fpath.read_text(encoding="utf-8")
        for ch in text:
            if ch not in alphabet:
                unknown[ch] = unknown.get(ch, 0) + 1

    if unknown:
        log.warning("Nepoznati znakovi (nisu u abecedi):")
        for ch, cnt in sorted(unknown.items(), key=lambda x: -x[1]):
            log.warning(f"  {repr(ch):10s}  pojavljivanja: {cnt}")
    else:
        log.info("Svi znakovi su unutar definirane abecede.")

    return list(unknown.keys())


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalizacija transkripata iz ISO-8859-2 u UTF-8"
    )
    parser.add_argument(
        "--input", default="data/raw/text",
        help="Mapa s izvornim .txt transkriptima (ISO-8859-2)"
    )
    parser.add_argument(
        "--output", default="data/processed/normalized",
        help="Mapa za normalizirane transkripte (UTF-8)"
    )
    parser.add_argument(
        "--encoding", default="iso-8859-2",
        help="Kodna stranica izvornih datoteka (default: iso-8859-2)"
    )
    parser.add_argument(
        "--alphabet", default="configs/alphabet_hr.txt",
        help="Putanja do datoteke s abecedom (za provjeru)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Provjeri znakove prema abecedi nakon normalizacije"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = normalize_directory(args.input, args.output, args.encoding)

    print(f"\n{'='*50}")
    print(f"Rezultat normalizacije:")
    print(f"  Uspješno:  {stats['ok']}")
    print(f"  Prazno:    {stats['empty']}")
    print(f"  Greške:    {stats['error']}")
    print(f"{'='*50}")

    if args.verify and os.path.exists(args.alphabet):
        print("\nProvjera abecede ...")
        unknown = verify_charset(args.output, args.alphabet)
        if not unknown:
            print("OK — svi znakovi su u abecedi.")
        else:
            print(f"UPOZORENJE: {len(unknown)} nepoznatih znakova!")
