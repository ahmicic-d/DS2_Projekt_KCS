import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpecAugment(nn.Module):

    def __init__(self, freq_mask_max=15, time_mask_max=50,
                 n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_max)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_max)
        self.n_freq = n_freq_masks
        self.n_time = n_time_masks

    def forward(self, x):
        for _ in range(self.n_freq):
            x = self.freq_masking(x)
        for _ in range(self.n_time):
            x = self.time_masking(x)
        return x


class CNNLayerNorm(nn.Module):
    """Layer normalizacija prilagođena za CNN output (B, C, F, T)."""
    def __init__(self, n_feats):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):

        x = x.transpose(2, 3).contiguous()  # (B, C, T, F)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (B, C, F, T)


class ResidualCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride,
                 dropout, n_feats):
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels, out_channels,
            kernel, stride=stride,
            padding=kernel // 2
        )
        self.cnn2 = nn.Conv2d(
            out_channels, out_channels,
            kernel, stride=stride,
            padding=kernel // 2
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        # Rezidualna veza: zbroji input i output
        x = x + residual
        return x

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super().__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class DeepSpeech2(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim,
                 n_class, n_feats, stride=2, dropout=0.1):
        super().__init__()
        self.n_feats = n_feats
        n_feats_after = n_feats // 2 

        # CNN slojevi
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(
                in_channels=32, out_channels=32,
                kernel=3, stride=1, dropout=dropout,
                n_feats=n_feats_after
            )
            for _ in range(n_cnn_layers)
        ])

        # Projekcija CNN → RNN
        self.fully_connected = nn.Linear(n_feats_after * 32, rnn_dim)

        # RNN slojevi
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                hidden_size=rnn_dim,
                dropout=dropout,
                batch_first=(i == 0)
            )
            for i in range(n_rnn_layers)
        ])

        # Klasifikator
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
        )

    def forward(self, x):

        # CNN
        x = self.cnn(x)            # (B, 32, n_feats/2, T)
        x = self.rescnn_layers(x)  # (B, 32, n_feats/2, T)

        # Flatten CNN output za RNN
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (B, C*F, T)
        x = x.transpose(1, 2)  # (B, T, C*F) — batch_first za RNN

        # Projekcija
        x = self.fully_connected(x)  # (B, T, rnn_dim)

        # RNN
        x = self.birnn_layers(x)     # (B, T, rnn_dim*2)

        # Klasifikator
        x = self.classifier(x)      # (B, T, n_class)

        # CTC očekuje (T, B, n_class)
        x = x.transpose(0, 1)       # (T, B, n_class)
        x = F.log_softmax(x, dim=2)

        return x

    def count_parameters(self) -> int:
        """Vrati ukupan broj parametara modela."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AudioPreprocessor(nn.Module):
    """
    Pretvara sirovi audio signal u Log-Mel spektrogram.
    Koristi se za ulaz u DeepSpeech2.
    """
    def __init__(self, sample_rate=16000, n_mels=128,
                 win_length_ms=20, hop_length_ms=10, n_fft=512):
        super().__init__()
        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
    
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)

        mean = spec.mean()
        std  = spec.std().clamp(min=1e-9)
        spec = (spec - mean) / std
        return spec


def build_model(config: dict) -> DeepSpeech2:

    m = config["model"]
    model = DeepSpeech2(
        n_cnn_layers=m["n_cnn_layers"],
        n_rnn_layers=m["n_rnn_layers"],
        rnn_dim=m["rnn_dim"],
        n_class=m["n_class"],
        n_feats=m["n_feats"],
        stride=m["stride"],
        dropout=m["dropout"],
    )
    return model


if __name__ == "__main__":
    print("Testiranje DeepSpeech2 arhitekture ...")


    batch_size = 4
    n_mels     = 128
    T          = 200
    n_class    = 30

    x = torch.randn(batch_size, 1, n_mels, T)

    model = DeepSpeech2(
        n_cnn_layers=3,
        n_rnn_layers=5,
        rnn_dim=512,
        n_class=n_class,
        n_feats=n_mels,
        stride=2,
        dropout=0.1
    )

    out = model(x)

    print(f"  Ulaz:   {x.shape}")
    print(f"  Izlaz:  {out.shape}  (T, B, n_class)")
    print(f"  Parametara: {model.count_parameters():,}")
    print("OK!")
