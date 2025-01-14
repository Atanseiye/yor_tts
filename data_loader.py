import os
import torch
from data_prep import wav_to_mel, normalize_text
from torch.utils.data import Dataset, DataLoader

class TTSDataset(Dataset):
    def __init__(self, metadata, data_dir, sample_rate=22050, n_mels=80):
        """
        Initialize the TTS Dataset.
        Args:
        - metadata: Pandas DataFrame containing 'text' and 'wav_path'.
        - data_dir: Path to the dataset directory.
        - sample_rate: Sampling rate for audio.
        - n_mels: Number of mel bands for spectrograms.
        """
        self.metadata = metadata
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get text and audio path
        text = self.metadata.iloc[idx]["text"]
        wav_path = os.path.join(self.data_dir, self.metadata.iloc[idx]["wav_path"])

        # Preprocess text
        text_tensor = torch.tensor([ord(c) for c in normalize_text(text)], dtype=torch.long)

        # Convert audio to mel-spectrogram
        mel_spec = wav_to_mel(wav_path, sample_rate=self.sample_rate, n_mels=self.n_mels)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32)

        return text_tensor, mel_tensor


def create_dataloader(metadata, data_dir, batch_size=32, shuffle=True, num_workers=2):
    """
    Create a DataLoader for the TTS dataset.
    """
    dataset = TTSDataset(metadata, data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    """
    Collate function to pad sequences for batching.
    Args:
    - batch: List of tuples (text_tensor, mel_tensor).
    Returns:
    - Padded text and mel-spectrogram tensors.
    """
    texts, mels = zip(*batch)

    # Pad text sequences
    text_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)

    # Pad mel-spectrogram sequences
    mel_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    padded_mels = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)

    return padded_texts, text_lengths, padded_mels, mel_lengths
