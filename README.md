# DeepSpeech2 — Raspoznavanje hrvatskog govora

Projektni zadatak iz kolegija **Komunikacija čovjek-stroj**.

Tema: Učenje akustičnih modela govora za raspoznavanje pomoću alata DeepSpeech2  
Baza: Hrvatski govorni korpus (HGK) — govornik `sm045`, sesija `01`, 3069 snimaka

---

## Struktura projekta

```
ds2_projekt/
├── data/
│   ├── raw/
│   │   ├── audio/          ← ovdje kopiraj WAV datoteke (*.wav)
│   │   └── text/           ← ovdje kopiraj TXT transkripte (*.txt)
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
│   ├── prepare_data.sh     ← priprema podataka (korak po korak)
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

---

## Brzi start

### 1. Pripremi okruženje

```bash
# Kloniraj / otvori projekt u VS Code
# Otvori terminal (Ctrl+`)

bash scripts/setup.sh
```

### 2. Postavi podatke

Kopiraj sve WAV i TXT datoteke:

```bash
cp /putanja/do/audio/*.wav   data/raw/audio/
cp /putanja/do/text/*.txt    data/raw/text/
```

### 3. Pripremi podatke

```bash
bash scripts/prepare_data.sh
```

Ovo će:
- Normalizirati transkripcije (ISO-8859-2 → UTF-8, dijakritici)
- Podijeliti na train/dev/test skup
- Kreirati CSV manifeste

### 4. Treniraj model

```bash
python src/train.py --config configs/training_config.json
```

### 5. Evaluiraj model

```bash
python src/evaluate.py --model models/best_model.pth --test_manifest data/processed/test/manifest.csv
```

### 6. Transkribij novu snimku

```bash
python src/transcribe.py --model models/best_model.pth --audio moja_snimka.wav
```

---

## Kodiranje dijakritičkih znakova

Izvorna baza koristi ISO-8859-2 i ASCII supstitucije:

| ASCII znak | Slovo | Fonem (rječnik) |
|-----------|-------|-----------------|
| `{`       | š     | S               |
| `~`       | č     | C               |
| `` ` ``   | ž     | Z               |
| `^`       | đ     | cc              |
| `d``      | dž    | dZ              |
| `<sil>`   | —     | tišina (SIL)    |

Normalizacijska skripta automatski rješava ovaj problem.

---

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

---

## Napomene

- Audio format: WAV, 16 kHz, mono, 16-bit PCM (DS2 standard)
- Minimalni podaci za pravi ASR: ~100h govora. Za demonstraciju koristimo fine-tuning.
- Treniranje na CPU je sporo (~1h/epoch za 3069 snimaka). Preporučuje se GPU.
- Za produkciju preporučiti: Whisper (OpenAI) ili wav2vec2 fine-tuned na HR korpusu.
