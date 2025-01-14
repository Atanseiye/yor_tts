# HiFi-GAN vocoder integration (use a pre-trained model)
from hifi_gan import HiFiGAN  # Install from https://github.com/jik876/hifi-gan

vocoder = HiFiGAN(pretrained=True)

# Convert mel-spectrogram to audio
def mel_to_audio(mel_spec):
    """
    Convert mel-spectrogram to waveform using HiFi-GAN.
    """
    audio = vocoder.generate(mel_spec)
    return audio
