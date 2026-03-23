"""
train.py
--------
Treniranje DeepSpeech2 modela za raspoznavanje hrvatskog govora.

Pokretanje:
    python src/train.py --config configs/training_config.json

Opcije:
    --config        Putanja do JSON konfig datoteke
    --resume        Putanja do checkpointa (nastavak treniranja)
    --device        cpu / cuda / mps (auto-detect ako nije navedeno)
    --debug         Smanji dataset na 20 uzoraka za brzi test
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Lokalni moduli
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model   import DeepSpeech2, build_model
from dataset import Alphabet, build_dataloaders
from evaluate import compute_wer, compute_cer, greedy_decode

# ──────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Trainer klasa
# ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict, device: torch.device,
                 resume_path: str = None, debug: bool = False):
        self.config = config
        self.device = device
        self.debug  = debug

        # Putanje
        self.model_dir   = Path(config["paths"]["model_dir"])
        self.results_dir = Path(config["paths"]["results_dir"])
        self.log_dir     = Path(config["paths"]["log_dir"])
        for d in [self.model_dir, self.results_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Log u datoteku
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(self.log_dir / f"train_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)

        # Abeceda
        self.alphabet = Alphabet(config["data"]["alphabet"])

        # Dataloaders
        log.info("Učitavanje dataseta ...")
        self.train_loader, self.dev_loader, self.test_loader = \
            build_dataloaders(config, self.alphabet)

        if debug:
            log.warning("DEBUG mod: ograničen dataset na 20 uzoraka!")

        # Model
        n_class = len(self.alphabet)
        config["model"]["n_class"] = n_class
        self.model = build_model(config).to(device)
        log.info(f"Model: DeepSpeech2, {self.model.count_parameters():,} parametara")
        log.info(f"Veličina abecede: {n_class} (uključujući blank=0)")

        # Gubitak
        self.ctc_loss = nn.CTCLoss(blank=Alphabet.BLANK_IDX, reduction="mean",
                                    zero_infinity=True).to(device)

        # Optimizator
        train_cfg = config["training"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )

        # Scheduler
        steps_per_epoch = len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=train_cfg["learning_rate"],
            steps_per_epoch=steps_per_epoch,
            epochs=train_cfg["epochs"],
            anneal_strategy="cos"
        )

        # Praćenje najboljeg modela
        self.best_wer   = float("inf")
        self.start_epoch = 1
        self.history    = {"train_loss": [], "dev_loss": [], "dev_wer": [], "dev_cer": []}

        # Nastavi treniranje iz checkpointa
        if resume_path:
            self._load_checkpoint(resume_path)

    def _load_checkpoint(self, path: str) -> None:
        log.info(f"Učitavam checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_wer    = ckpt.get("best_wer", float("inf"))
        self.history     = ckpt.get("history", self.history)
        log.info(f"Nastavljam od epohe {self.start_epoch}, best WER={self.best_wer:.2f}%")

    def _save_checkpoint(self, epoch: int, wer: float, is_best: bool = False) -> None:
        ckpt = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_wer":        self.best_wer,
            "history":         self.history,
            "config":          self.config,
        }
        # Periodički checkpoint
        if epoch % self.config["training"].get("save_every_n_epochs", 10) == 0:
            path = self.model_dir / f"checkpoint_epoch{epoch:04d}.pth"
            torch.save(ckpt, path)
            log.info(f"  Checkpoint spreman: {path}")

        # Zadnji checkpoint (uvijek)
        torch.save(ckpt, self.model_dir / "checkpoint_last.pth")

        # Najbolji model
        if is_best:
            torch.save(ckpt, self.model_dir / "best_model.pth")
            log.info(f"  *** Novi best model spreman! WER={wer:.2f}% ***")

    def _train_epoch(self, epoch: int) -> float:
        """Jedna epoha treniranja. Vraća prosječni CTC gubitak."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        log_every  = self.config["training"].get("log_every_n_steps", 10)
        grad_clip  = self.config["training"].get("grad_clip", 5.0)

        pbar = tqdm(self.train_loader, desc=f"Epoha {epoch} [train]", leave=False)

        for step, (specs, labels, input_lens, label_lens) in enumerate(pbar, 1):
            specs  = specs.to(self.device)
            labels = labels.to(self.device)

            # Forward
            log_probs = self.model(specs)  # (T, B, n_class)

            # CTC gubitak
            # input_lens se moraju skalirati prema CNN strideovanju
            # stride=2 u CNN → T → T//2
            T_out = log_probs.shape[0]
            # Proporcionalno skaliraj duljine
            scaled_input_lens = torch.clamp(
                (input_lens.float() * T_out / specs.shape[-1]).long(),
                max=T_out
            )

            loss = self.ctc_loss(
                log_probs, labels,
                scaled_input_lens, label_lens
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

            if step % log_every == 0:
                avg = total_loss / n_batches
                lr  = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr:.2e}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader, desc: str) -> tuple:
        """
        Evaluacija na skupu. Vraća (avg_loss, WER%, CER%).
        """
        self.model.eval()
        total_loss    = 0.0
        n_batches     = 0
        all_refs      = []
        all_hyps      = []

        for specs, labels, input_lens, label_lens in tqdm(loader, desc=desc, leave=False):
            specs  = specs.to(self.device)
            labels = labels.to(self.device)

            log_probs = self.model(specs)
            T_out = log_probs.shape[0]
            scaled_input_lens = torch.clamp(
                (input_lens.float() * T_out / specs.shape[-1]).long(),
                max=T_out
            )

            loss = self.ctc_loss(log_probs, labels, scaled_input_lens, label_lens)
            total_loss += loss.item()
            n_batches  += 1

            # Greedy dekodiranje po snimci
            for b in range(log_probs.shape[1]):
                T_b    = scaled_input_lens[b].item()
                L_b    = label_lens[b].item()
                probs  = log_probs[:T_b, b, :]   # (T_b, n_class)
                ref_ids = labels[b, :L_b].tolist()

                hyp = greedy_decode(probs, self.alphabet)
                ref = self.alphabet.decode(ref_ids, remove_duplicates=False)

                all_refs.append(ref)
                all_hyps.append(hyp)

        avg_loss = total_loss / max(n_batches, 1)
        wer      = compute_wer(all_refs, all_hyps)
        cer      = compute_cer(all_refs, all_hyps)

        return avg_loss, wer, cer

    def train(self) -> None:
        """Glavni loop treniranja."""
        epochs    = self.config["training"]["epochs"]
        patience  = self.config["training"].get("early_stopping_patience", 15)
        no_improve = 0

        log.info(f"\n{'='*60}")
        log.info(f"Pokretanje treniranja: {epochs} epoha")
        log.info(f"Uređaj: {self.device}")
        log.info(f"Train batches: {len(self.train_loader)}, "
                 f"Dev batches: {len(self.dev_loader)}")
        log.info(f"{'='*60}\n")

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()

            # Treniranje
            train_loss = self._train_epoch(epoch)

            # Validacija
            dev_loss, dev_wer, dev_cer = self._evaluate(
                self.dev_loader, f"Epoha {epoch} [dev]"
            )

            elapsed = time.time() - t0
            is_best = dev_wer < self.best_wer
            if is_best:
                self.best_wer = dev_wer
                no_improve    = 0
            else:
                no_improve += 1

            # Log
            log.info(
                f"Epoha {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"dev_loss={dev_loss:.4f} | "
                f"WER={dev_wer:6.2f}% | "
                f"CER={dev_cer:6.2f}% | "
                f"{'★ BEST' if is_best else ''} | "
                f"t={elapsed:.0f}s"
            )

            # Spremi u history
            self.history["train_loss"].append(train_loss)
            self.history["dev_loss"].append(dev_loss)
            self.history["dev_wer"].append(dev_wer)
            self.history["dev_cer"].append(dev_cer)

            # Spremi checkpoint
            self._save_checkpoint(epoch, dev_wer, is_best)

            # Early stopping
            if no_improve >= patience:
                log.info(
                    f"\nEarly stopping: nema poboljšanja {patience} epoha zaredom."
                )
                break

        log.info(f"\nTreniranje završeno! Best dev WER = {self.best_wer:.2f}%")

        # Finalna evaluacija na test skupu
        log.info("\nFinalna evaluacija na test skupu ...")
        test_loss, test_wer, test_cer = self._evaluate(
            self.test_loader, "Test"
        )
        log.info(
            f"TEST | loss={test_loss:.4f} | WER={test_wer:.2f}% | CER={test_cer:.2f}%"
        )

        # Spremi rezultate
        results = {
            "best_dev_wer": self.best_wer,
            "test_wer":     test_wer,
            "test_cer":     test_cer,
            "history":      self.history,
        }
        results_path = self.results_dir / "training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"Rezultati spremljeni: {results_path}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Treniranje DeepSpeech2")
    parser.add_argument("--config", default="configs/training_config.json",
                        help="Putanja do JSON konfig datoteke")
    parser.add_argument("--resume", default=None,
                        help="Putanja do checkpointa za nastavak treniranja")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps (auto ako nije navedeno)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mod: manji dataset, manje epoha")
    return parser.parse_args()


def get_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        log.info("Apple Silicon MPS uređaj")
    else:
        dev = torch.device("cpu")
        log.info("CPU uređaj (treniranje će biti sporo!)")
    return dev


if __name__ == "__main__":
    args = parse_args()

    # Učitaj konfig
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    # Debug mod: manje epoha
    if args.debug:
        config["training"]["epochs"] = 5
        config["training"]["batch_size"] = 4

    # Uređaj
    device = get_device(args.device)

    # Treniranje
    trainer = Trainer(
        config=config,
        device=device,
        resume_path=args.resume,
        debug=args.debug,
    )
    trainer.train()
