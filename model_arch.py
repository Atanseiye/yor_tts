import torch
import torch.nn as nn

class Tacotron2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Tacotron2, self).__init__()
        # Encoder
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Encoding
        x = self.embedding(x)
        encoder_output, _ = self.encoder_lstm(x)

        # Decoding
        decoder_output, _ = self.decoder_lstm(encoder_output)
        mel_output = self.linear(decoder_output)
        return mel_output
