"""
Audio-specific loss functions for FocalCodec training.

Includes:
- Mel-Spectrogram Loss
- Multi-Resolution STFT Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram Loss for perceptual audio quality.

    Computes L1 distance between mel-spectrograms of original and reconstructed audio.
    Mel-spectrogram better represents human auditory perception than raw waveforms.

    Args:
        sample_rate: Audio sample rate (default: 16000 for speech)
        n_fft: FFT size (default: 1024)
        hop_length: Hop size for STFT (default: 256)
        n_mels: Number of mel filterbanks (default: 80 for speech)
        f_min: Minimum frequency (default: 0)
        f_max: Maximum frequency (default: 8000, half of 16kHz)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Create mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,  # Use magnitude instead of power
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram loss.

        Args:
            pred: Predicted audio [batch, samples]
            target: Target audio [batch, samples]

        Returns:
            L1 loss between mel-spectrograms
        """
        # Ensure inputs are 2D: [batch, samples]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        # Move transform to same device as input
        self.mel_transform = self.mel_transform.to(pred.device)

        # Compute mel-spectrograms
        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)

        # Add small epsilon to avoid log(0)
        pred_mel = torch.log(pred_mel + 1e-5)
        target_mel = torch.log(target_mel + 1e-5)

        # L1 loss
        loss = F.l1_loss(pred_mel, target_mel)

        return loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss.

    Computes STFT loss at multiple resolutions to capture both fine-grained
    and coarse-grained spectral information.

    Based on: "Parallel WaveGAN" (https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes: List of FFT sizes (default: [512, 1024, 2048])
        hop_sizes: List of hop sizes (default: [128, 256, 512])
        win_lengths: List of window lengths (default: [512, 1024, 2048])
    """

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_lengths: list = [512, 1024, 2048],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        # Create STFT transforms for each resolution
        self.stft_transforms = nn.ModuleList([
            torchaudio.transforms.Spectrogram(
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                power=None,  # Return complex spectrogram
                return_complex=True,
            )
            for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio [batch, samples]
            target: Target audio [batch, samples]

        Returns:
            Combined spectral convergence and magnitude loss
        """
        # Ensure inputs are 2D: [batch, samples]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        total_loss = 0.0

        for stft_transform in self.stft_transforms:
            # Move transform to same device as input
            stft_transform = stft_transform.to(pred.device)

            # Compute STFT
            pred_stft = stft_transform(pred)
            target_stft = stft_transform(target)

            # Compute magnitude
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # Spectral convergence loss
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')

            # Log magnitude loss
            log_pred_mag = torch.log(pred_mag + 1e-5)
            log_target_mag = torch.log(target_mag + 1e-5)
            mag_loss = F.l1_loss(log_pred_mag, log_target_mag)

            # Combine losses for this resolution
            total_loss += sc_loss + mag_loss

        # Average over all resolutions
        total_loss = total_loss / len(self.stft_transforms)

        return total_loss


class CombinedAudioLoss(nn.Module):
    """
    Combined loss for audio reconstruction.

    Combines:
    - Feature loss (MSE in latent space)
    - Mel-spectrogram loss (perceptual quality)
    - Multi-resolution STFT loss (spectral quality)

    Args:
        weight_feature: Weight for feature loss (default: 1.0)
        weight_mel: Weight for mel-spectrogram loss (default: 45.0)
        weight_stft: Weight for STFT loss (default: 1.0)
        sample_rate: Audio sample rate (default: 16000)
    """

    def __init__(
        self,
        weight_feature: float = 1.0,
        weight_mel: float = 45.0,
        weight_stft: float = 1.0,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.weight_feature = weight_feature
        self.weight_mel = weight_mel
        self.weight_stft = weight_stft

        # Initialize loss functions
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            pred_audio: Reconstructed audio [batch, samples]
            target_audio: Original audio [batch, samples]
            pred_features: Reconstructed features [batch, time, dim]
            target_features: Original features [batch, time, dim]

        Returns:
            (total_loss, loss_dict)
        """
        # Feature loss (MSE in latent space)
        feature_loss = F.mse_loss(pred_features, target_features)

        # Mel-spectrogram loss
        mel_loss = self.mel_loss(pred_audio, target_audio)

        # STFT loss
        stft_loss = self.stft_loss(pred_audio, target_audio)

        # Combine losses
        total_loss = (
            self.weight_feature * feature_loss +
            self.weight_mel * mel_loss +
            self.weight_stft * stft_loss
        )

        # Return detailed loss components
        loss_dict = {
            'feature_loss': feature_loss.item(),
            'mel_loss': mel_loss.item(),
            'stft_loss': stft_loss.item(),
        }

        return total_loss, loss_dict


# Example usage and testing
if __name__ == '__main__':
    # Test mel-spectrogram loss
    mel_loss_fn = MelSpectrogramLoss()

    # Create dummy audio (1 second at 16kHz)
    batch_size = 4
    audio_length = 16000
    pred_audio = torch.randn(batch_size, audio_length)
    target_audio = torch.randn(batch_size, audio_length)

    mel_loss = mel_loss_fn(pred_audio, target_audio)
    print(f"Mel Loss: {mel_loss.item():.4f}")

    # Test STFT loss
    stft_loss_fn = MultiResolutionSTFTLoss()
    stft_loss = stft_loss_fn(pred_audio, target_audio)
    print(f"STFT Loss: {stft_loss.item():.4f}")

    # Test combined loss
    combined_loss_fn = CombinedAudioLoss()

    # Dummy features
    feature_length = 50  # 50Hz for 1 second
    feature_dim = 1024
    pred_features = torch.randn(batch_size, feature_length, feature_dim)
    target_features = torch.randn(batch_size, feature_length, feature_dim)

    total_loss, loss_dict = combined_loss_fn(
        pred_audio, target_audio,
        pred_features, target_features
    )

    print(f"\nCombined Loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
