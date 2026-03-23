import csv
import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)


class Alphabet:

    BLANK_IDX = 0

    def __init__(self, alphabet_path: str):
        with open(alphabet_path, encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f if line.rstrip("\n")]

        seen = set()
        unique_chars = []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                unique_chars.append(ch)

        self._char_to_idx = {ch: i + 1 for i, ch in enumerate(unique_chars)}
        self._idx_to_char = {i + 1: ch for i, ch in enumerate(unique_chars)}
        self._idx_to_char[self.BLANK_IDX] = ""  # blank → prazan string

        log.info(
            f"Abeceda učitana: {len(unique_chars)} znakova, "
            f"ukupno indeksa (s blankom): {len(unique_chars) + 1}"
        )

    def encode(self, text: str) -> list:

        return [
            self._char_to_idx[ch]
            for ch in text
            if ch in self._char_to_idx
        ]

    def decode(self, indices: list, remove_duplicates: bool = True) -> str:

        if remove_duplicates:

            result = []
            prev = None
            for idx in indices:
                if idx != prev:
                    result.append(idx)
                    prev = idx
            indices = result

        return "".join(
            self._idx_to_char.get(i, "")
            for i in indices
            if i != self.BLANK_IDX
        )

    def __len__(self):
        return len(self._char_to_idx) + 1  # +1 za blank


class SpeechDataset(Dataset):

    def __init__(self, manifest_path: str, alphabet: Alphabet,
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 win_length_ms: float = 20,
                 hop_length_ms: float = 10,
                 n_fft: int = 512,
                 augment: bool = False,
                 aug_config: dict = None):
        super().__init__()
        self.alphabet     = alphabet
        self.sample_rate  = sample_rate
        self.augment      = augment

        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel_transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
            ),
            T.AmplitudeToDB(),
        )

        if augment and aug_config:
            self.spec_augment = torch.nn.Sequential(
                T.FrequencyMasking(aug_config.get("freq_mask_max", 15)),
                T.TimeMasking(aug_config.get("time_mask_max", 50)),
            )
        else:
            self.spec_augment = None

        self.samples = self._load_manifest(manifest_path)
        log.info(
            f"Dataset učitan: {manifest_path} "
            f"({len(self.samples)} uzoraka, augment={augment})"
        )

    def _load_manifest(self, path: str) -> list:
        samples = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_path   = row["wav_filepath"]
                transcript = row["transcript"].strip()
                if not transcript:
                    continue
                encoded = self.alphabet.encode(transcript)
                if not encoded:
                    log.debug(f"Prazan enkodiran transkript: {wav_path}")
                    continue
                samples.append({
                    "wav_path":   wav_path,
                    "transcript": transcript,
                    "encoded":    encoded,
                })
        return samples

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        """Učitaj WAV i vrati mono waveform na ispravnom sample ratu."""
        waveform, sr = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform  = resampler(waveform)

        return waveform  

    def _audio_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:

        spec = self.mel_transform(waveform)  


        if self.spec_augment is not None:
            spec = self.spec_augment(spec)


        mean = spec.mean()
        std  = spec.std().clamp(min=1e-9)
        spec = (spec - mean) / std

        return spec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        waveform = self._load_audio(sample["wav_path"])
        spec     = self._audio_to_spectrogram(waveform)

        label    = torch.tensor(sample["encoded"], dtype=torch.long)

        return spec, label, spec.shape[-1], len(label)


def collate_fn(batch):

    specs, labels, input_lens, label_lens = zip(*batch)


    max_T = max(s.shape[-1] for s in specs)
    padded_specs = []
    for s in specs:
        pad_len = max_T - s.shape[-1]
        padded  = torch.nn.functional.pad(s, (0, pad_len))
        padded_specs.append(padded)
    spectrograms = torch.stack(padded_specs)  # (B, 1, n_mels, T_max)

    labels_padded = pad_sequence(
        [l for l in labels],
        batch_first=True, padding_value=0
    )  # (B, L_max)

    input_lens = torch.tensor(input_lens, dtype=torch.long)
    label_lens = torch.tensor(label_lens, dtype=torch.long)

    return spectrograms, labels_padded, input_lens, label_lens


def build_dataloaders(config: dict, alphabet: Alphabet) -> tuple:

    audio_cfg = config["audio"]
    data_cfg  = config["data"]
    aug_cfg   = config.get("augmentation", {})

    kwargs = dict(
        alphabet      = alphabet,
        sample_rate   = audio_cfg["sample_rate"],
        n_mels        = audio_cfg["n_mels"],
        win_length_ms = audio_cfg["win_length_ms"],
        hop_length_ms = audio_cfg["hop_length_ms"],
        n_fft         = audio_cfg["n_fft"],
    )

    train_dataset = SpeechDataset(
        data_cfg["train_manifest"], augment=aug_cfg.get("enabled", True),
        aug_config=aug_cfg, **kwargs
    )
    dev_dataset = SpeechDataset(
        data_cfg["dev_manifest"], augment=False, **kwargs
    )
    test_dataset = SpeechDataset(
        data_cfg["test_manifest"], augment=False, **kwargs
    )

    bs    = config["training"]["batch_size"]
    nw    = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=nw, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=nw, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=nw, pin_memory=True
    )

    return train_loader, dev_loader, test_loader
