import csv
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model   import DeepSpeech2, build_model
from dataset import Alphabet, SpeechDataset, collate_fn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def levenshtein(seq1: list, seq2: list) -> int:

    n, m = len(seq1), len(seq2)
    if n == 0: return m
    if m == 0: return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],    
                                   curr[j - 1],
                                   prev[j - 1]) 
        prev, curr = curr, prev

    return prev[m]


def compute_wer(references: list, hypotheses: list) -> float:

    total_dist = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_dist  += levenshtein(ref_words, hyp_words)
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0
    return 100.0 * total_dist / total_words


def compute_cer(references: list, hypotheses: list) -> float:

    total_dist  = 0
    total_chars = 0
    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        total_dist  += levenshtein(ref_chars, hyp_chars)
        total_chars += len(ref_chars)

    if total_chars == 0:
        return 0.0
    return 100.0 * total_dist / total_chars


def greedy_decode(log_probs: torch.Tensor, alphabet: Alphabet) -> str:

    indices = torch.argmax(log_probs, dim=-1).tolist()  # (T,)
    return alphabet.decode(indices, remove_duplicates=True)


def beam_search_decode(log_probs: torch.Tensor, alphabet: Alphabet,
                        beam_width: int = 10) -> str:


    blank = Alphabet.BLANK_IDX
    T = log_probs.shape[0]


    beams = {(): 0.0}

    for t in range(T):
        probs = log_probs[t].exp().tolist()
        new_beams = {}

        for prefix, prefix_score in beams.items():
            for c, prob in enumerate(probs):
                if prob < 1e-15:
                    continue

                import math
                log_p = math.log(prob + 1e-30)

                if c == blank:

                    new_prefix = prefix
                elif prefix and prefix[-1] == c:

                    new_prefix = prefix
                else:
                    new_prefix = prefix + (c,)

                score = prefix_score + log_p
                if new_prefix in new_beams:

                    import math
                    m = max(new_beams[new_prefix], score)
                    new_beams[new_prefix] = m + math.log(
                        math.exp(new_beams[new_prefix] - m) + math.exp(score - m)
                    )
                else:
                    new_beams[new_prefix] = score

        beams = dict(sorted(new_beams.items(), key=lambda x: -x[1])[:beam_width])

    best_prefix = max(beams, key=beams.get)
    return alphabet.decode(list(best_prefix), remove_duplicates=False)


class Evaluator:
    def __init__(self, model_path: str, config: dict, device: torch.device):
        self.device   = device
        self.config   = config
        self.alphabet = Alphabet(config["data"]["alphabet"])

        ckpt = torch.load(model_path, map_location=device)
        config["model"]["n_class"] = len(self.alphabet)
        self.model = build_model(config).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        log.info(f"Model učitan: {model_path}")

        audio_cfg = config["audio"]
        sr = audio_cfg["sample_rate"]
        wl = int(sr * audio_cfg["win_length_ms"] / 1000)
        hl = int(sr * audio_cfg["hop_length_ms"] / 1000)
        self.preprocessor = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sr,
                n_fft=audio_cfg["n_fft"],
                win_length=wl,
                hop_length=hl,
                n_mels=audio_cfg["n_mels"],
            ),
            T.AmplitudeToDB(),
        ).to(device)
        self.sample_rate = sr

    @torch.no_grad()
    def evaluate_manifest(self, manifest_path: str,
                           output_csv: str = None) -> dict:

        dataset = SpeechDataset(
            manifest_path, self.alphabet,
            sample_rate=self.config["audio"]["sample_rate"],
            n_mels=self.config["audio"]["n_mels"],
            win_length_ms=self.config["audio"]["win_length_ms"],
            hop_length_ms=self.config["audio"]["hop_length_ms"],
            n_fft=self.config["audio"]["n_fft"],
            augment=False,
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

        refs, hyps, filenames = [], [], []

        print(f"\n{'='*70}")
        print(f"Evaluacija: {manifest_path}")
        print(f"{'='*70}")


        with open(manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fnames = [Path(row["wav_filepath"]).stem for row in reader]

        idx = 0
        for specs, labels, input_lens, label_lens in loader:
            specs  = specs.to(self.device)
            labels = labels.to(self.device)

            log_probs = self.model(specs)  # (T, B, n_class)
            T_out = log_probs.shape[0]
            scaled_input_lens = torch.clamp(
                (input_lens.float() * T_out / specs.shape[-1]).long(),
                max=T_out
            )

            for b in range(log_probs.shape[1]):
                T_b     = scaled_input_lens[b].item()
                L_b     = label_lens[b].item()
                probs_b = log_probs[:T_b, b, :]

                ref_ids = labels[b, :L_b].tolist()
                ref = self.alphabet.decode(ref_ids, remove_duplicates=False)
                hyp = greedy_decode(probs_b, self.alphabet)

                refs.append(ref)
                hyps.append(hyp)
                fname = fnames[idx] if idx < len(fnames) else str(idx)
                filenames.append(fname)
                idx += 1


        print(f"\n{'Datoteka':<25} {'Referenca':<35} {'Hipoteza':<35} {'WER%':>6}")
        print("-" * 103)

        per_sample = []
        for fname, ref, hyp in zip(filenames, refs, hyps):
            w = compute_wer([ref], [hyp])
            c = compute_cer([ref], [hyp])
            per_sample.append({
                "file": fname, "reference": ref,
                "hypothesis": hyp, "wer": round(w, 1), "cer": round(c, 1)
            })
            flag = "✓" if w == 0 else "✗"
            print(f"{fname:<25} {ref:<35} {hyp:<35} {w:>5.1f}% {flag}")

        wer = compute_wer(refs, hyps)
        cer = compute_cer(refs, hyps)
        n_correct = sum(1 for r, h in zip(refs, hyps) if r == h)

        print(f"\n{'='*70}")
        print(f"Ukupno uzoraka:     {len(refs)}")
        print(f"Točnih transkripta: {n_correct} ({100*n_correct/max(len(refs),1):.1f}%)")
        print(f"WER (agregirani):   {wer:.2f}%")
        print(f"CER (agregirani):   {cer:.2f}%")
        print(f"{'='*70}")


        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["file","reference","hypothesis","wer","cer"])
                w.writeheader()
                w.writerows(per_sample)
            log.info(f"Rezultati po uzorku spremljeni: {output_csv}")

        return {
            "wer": wer, "cer": cer,
            "n_total": len(refs), "n_correct": n_correct,
            "per_sample": per_sample,
        }

    @torch.no_grad()
    def transcribe_file(self, wav_path: str,
                         use_beam_search: bool = False,
                         beam_width: int = 10) -> str:
        """Transkribiraj jednu WAV datoteku i vrati string."""
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)

        waveform = waveform.to(self.device)
        spec = self.preprocessor(waveform)  # (1, n_mels, T)
        mean, std = spec.mean(), spec.std().clamp(min=1e-9)
        spec = (spec - mean) / std

        spec = spec.unsqueeze(0)  # (1, 1, n_mels, T)
        log_probs = self.model(spec)[:, 0, :]  # (T, n_class)

        if use_beam_search:
            return beam_search_decode(log_probs, self.alphabet, beam_width)
        else:
            return greedy_decode(log_probs, self.alphabet)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluacija DeepSpeech2 modela")
    parser.add_argument("--model",    required=True,
                        help="Putanja do .pth checkpoint datoteke")
    parser.add_argument("--manifest", required=True,
                        help="Putanja do CSV manifesta za evaluaciju")
    parser.add_argument("--config",   default="configs/training_config.json",
                        help="Putanja do JSON konfig datoteke")
    parser.add_argument("--output",   default=None,
                        help="Spremi rezultate u CSV (opcionalno)")
    parser.add_argument("--device",   default=None,
                        help="cpu / cuda / mps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log.info(f"Uređaj: {device}")

    evaluator = Evaluator(args.model, config, device)
    results = evaluator.evaluate_manifest(
        args.manifest,
        output_csv=args.output or "results/test_results.csv"
    )
