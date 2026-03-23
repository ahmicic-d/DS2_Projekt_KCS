# DeepSpeech2 — Raspoznavanje hrvatskog govora

Projektni zadatak iz kolegija **Komunikacija čovjek-stroj**.

Tema: Učenje akustičnih modela govora za raspoznavanje pomoću alata DeepSpeech2  
Baza: Hrvatski govorni korpus (HGK) — govornik 3069 snimaka

---

## Struktura projekta

```
ds2_projekt/
├── data/
│   ├── raw/
│   │   ├── audio/          ← WAV datoteke (*.wav)
│   │   └── text/           ← TXT transkripte (*.txt)
│   └── processed/
│       ├── train/          ← generirano skriptom (80%)
│       ├── dev/            ← generirano skriptom (10%)
│       └── test/           ← generirano skriptom (10%)
├── src/
│   ├── normalize.py        ← normalizacija transkripata (ISO-8859-2 → UTF-8)
│   ├── make_manifest.py    ← kreiranje CSV manifesta za DS2
│   ├── train.py            ← pokretanje treniranja
│   ├── evaluate.py         ← evaluacija modela (WER, CER)
│   └── transcribe.py       ← transkripcija novih snimaka
├── scripts/
│   ├── setup.sh            ← instalacija ovisnosti
│   ├── prepare_data.sh     ← priprema podataka
│   └── run_all.sh          ← pokretanje cijelog pipelinea
├── configs/
│   ├── alphabet_hr.txt     ← hrvatska abeceda za DS2
│   └── training_config.json ← hiperparametri treniranja
├── notebooks/
│   └── analysis.ipynb      ← EDA i vizualizacija rezultata
├── models/                 ← ovdje se sprema naučeni model
├── results/                ← WER/CER rezultati, logovi
├── requirements.txt
└── README.md
```
## Ovisnosti

- Python 3.8+
- PyTorch ≥ 2.0
- torchaudio
- SpeechBrain (DeepSpeech2 model)
- jiwer (WER/CER metrike)
- librosa, soundfile (audio processing)
- pandas, numpy
- tqdm, matplotlib

Detaljno: `requirements.txt`


