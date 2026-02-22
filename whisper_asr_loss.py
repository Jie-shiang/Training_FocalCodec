"""
Whisper Token Cross-Entropy Loss for FocalCodec training.

Based on: "From Hallucination to Articulation: Language Model-Driven Losses
for Ultra Low-Bitrate Neural Speech Coding" (ICASSP 2026)
Reference: https://github.com/stet-stet/lmloss-icassp2026

Supports two modes:
1. Online mode: Transcribes clean audio on-the-fly (slow, ~7s/batch)
2. Cached mode: Uses pre-computed tokens from cache (fast, ~0.5s/batch)

Supports bilingual (Chinese + English) by switching language tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import torchaudio


# Whisper language token IDs (multilingual model)
LANG_TOKENS = {
    'en': 50259,  # <|en|>
    'zh': 50260,  # <|zh|>
}

# Special tokens
SOT = 50258        # <|startoftranscript|>
TRANSCRIBE = 50359 # <|transcribe|>
NO_TIMESTAMPS = 50363  # <|notimestamps|>
EOT = 50257        # <|endoftext|>


class WhisperASRLoss(nn.Module):
    """
    Whisper Token Cross-Entropy Loss.

    Uses a frozen Whisper model to compute cross-entropy loss between
    the decoder's predictions on reconstructed audio and pseudo-labels
    from clean audio transcription.

    Args:
        model_name: Whisper model size ('tiny', 'small', 'base', 'medium')
        default_language: Default language if not provided ('en' or 'zh')
        download_root: Directory to cache Whisper model weights
    """

    def __init__(
        self,
        model_name: str = "small",
        default_language: str = "en",
        download_root: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.default_language = default_language

        # Load frozen Whisper model
        self.whisper_model = whisper.load_model(
            model_name, download_root=download_root
        )

        # Convert alignment_heads buffer to dense for compatibility
        alignment_heads_dense = self.whisper_model.get_buffer(
            "alignment_heads"
        ).to_dense()
        self.whisper_model.register_buffer(
            "alignment_heads", alignment_heads_dense, persistent=False
        )

        # Freeze all Whisper parameters
        self.whisper_model.requires_grad_(False)
        self.whisper_model.eval()

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()

        # Resampler: FocalCodec decoder outputs 24kHz, Whisper expects 16kHz
        self.resampler_24k_to_16k = torchaudio.transforms.Resample(24000, 16000)

        print(f"WhisperASRLoss initialized: model={model_name}, "
              f"default_lang={default_language}")

    def _get_initial_tokens(self, language: str) -> torch.Tensor:
        """Get Whisper initial tokens for a given language."""
        lang_token = LANG_TOKENS.get(language, LANG_TOKENS[self.default_language])
        return torch.tensor(
            [SOT, lang_token, TRANSCRIBE, NO_TIMESTAMPS],
            dtype=torch.long
        )

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for Whisper: ensure 16kHz mono, pad/trim, compute mel.

        Args:
            audio: [B, T] waveform at 16kHz

        Returns:
            mel: [B, 80, 3000] log-mel spectrogram
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel

    def _transcribe_clean(
        self, audio: torch.Tensor, languages: list
    ) -> list:
        """
        Transcribe clean audio to get pseudo-label tokens (online mode).
        """
        device = audio.device
        mel = self._preprocess_audio(audio)

        all_tokens = []
        for i in range(mel.shape[0]):
            lang = languages[i] if i < len(languages) else self.default_language
            result = whisper.decode(
                self.whisper_model,
                mel[i:i+1],
                language=lang,
                without_timestamps=True,
            )
            tokens = torch.tensor(result[0].tokens, device=device)
            all_tokens.append(tokens)

        return all_tokens

    def _compute_loss_from_tokens(
        self,
        target_tokens: list,
        languages: list,
        reconstructed_audio: torch.Tensor,
        reconstructed_sr: int,
        whisper_batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Core loss computation: given target tokens and reconstructed audio,
        compute cross-entropy loss via Whisper encoder+decoder.

        Processes Whisper in mini-batches to avoid OOM on large batches.
        Whisper attention matrices scale as O(batch * seq^2), so batch=256
        would need ~99GB just for attention in the encoder.

        This is shared between online and cached modes.
        """
        device = reconstructed_audio.device

        # Resample reconstructed audio to 16kHz if needed
        if reconstructed_sr != 16000:
            recon_16k = self.resampler_24k_to_16k.to(device)(
                reconstructed_audio.squeeze(1) if reconstructed_audio.dim() == 3
                else reconstructed_audio
            )
        else:
            recon_16k = (
                reconstructed_audio.squeeze(1) if reconstructed_audio.dim() == 3
                else reconstructed_audio
            )

        # Compute mel for full batch (cheap, no attention)
        mel_recon = self._preprocess_audio(recon_16k)

        # Filter valid samples (non-empty tokens)
        batch_size = len(target_tokens)
        token_lengths = [len(t) for t in target_tokens]
        valid_indices = [i for i, l in enumerate(token_lengths) if l > 0]
        if not valid_indices:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Process Whisper encoder+decoder in mini-batches to avoid OOM
        all_valid_logits = []
        all_valid_targets = []

        for chunk_start in range(0, len(valid_indices), whisper_batch_size):
            chunk_end = min(chunk_start + whisper_batch_size, len(valid_indices))
            chunk_indices = valid_indices[chunk_start:chunk_end]

            # Whisper encoder for this mini-batch
            mel_chunk = mel_recon[chunk_indices]
            encoded_chunk = self.whisper_model.encoder(mel_chunk)

            # Prepare decoder inputs for this mini-batch
            chunk_tokens = [target_tokens[i] for i in chunk_indices]
            chunk_languages = [
                languages[i] if i < len(languages) else self.default_language
                for i in chunk_indices
            ]
            chunk_lengths = [token_lengths[i] for i in chunk_indices]
            chunk_max_length = max(chunk_lengths)

            padded_tokens = [
                F.pad(t.to(device), (0, chunk_max_length - len(t)), value=EOT)
                for t in chunk_tokens
            ]

            decoder_inputs_list = []
            init_lengths = []
            for i in range(len(chunk_tokens)):
                init_tokens = self._get_initial_tokens(chunk_languages[i]).to(device)
                decoder_input = torch.cat([init_tokens, padded_tokens[i]])
                decoder_inputs_list.append(decoder_input)
                init_lengths.append(len(init_tokens))

            decoder_inputs = torch.stack(decoder_inputs_list)

            ground_truth = torch.cat(
                [decoder_inputs[:, 1:], decoder_inputs[:, -1:]], dim=-1
            )

            # Whisper decoder for this mini-batch
            decoder_out = self.whisper_model.decoder(
                decoder_inputs, encoded_chunk
            )

            # Extract valid positions
            for i in range(len(chunk_tokens)):
                init_len = init_lengths[i]
                tok_len = chunk_lengths[i]
                start = init_len - 1
                end = start + tok_len
                all_valid_logits.append(decoder_out[i, start:end])
                all_valid_targets.append(ground_truth[i, start:end])

        all_valid_logits = torch.cat(all_valid_logits, dim=0)
        all_valid_targets = torch.cat(all_valid_targets, dim=0)

        loss = self.ce_loss(all_valid_logits, all_valid_targets)
        return loss

    def forward(
        self,
        clean_audio_16k: torch.Tensor = None,
        reconstructed_audio: torch.Tensor = None,
        languages: list = None,
        reconstructed_sr: int = 24000,
        cached_tokens: list = None,
    ) -> torch.Tensor:
        """
        Compute Whisper Token Cross-Entropy Loss.

        Two modes:
        1. Online: provide clean_audio_16k → transcribes on-the-fly
        2. Cached: provide cached_tokens → skips transcription (much faster)

        Args:
            clean_audio_16k: [B, T] clean audio at 16kHz (online mode)
            reconstructed_audio: [B, T'] reconstructed audio
            languages: list of language codes ('en'/'zh') per sample
            reconstructed_sr: sample rate of reconstructed audio
            cached_tokens: list of token tensors (cached mode)

        Returns:
            loss: scalar cross-entropy loss
        """
        batch_size = reconstructed_audio.shape[0]

        if languages is None:
            languages = [self.default_language] * batch_size

        # Get target tokens: from cache or by transcription
        if cached_tokens is not None:
            target_tokens = cached_tokens
        else:
            with torch.no_grad():
                target_tokens = self._transcribe_clean(clean_audio_16k, languages)

        return self._compute_loss_from_tokens(
            target_tokens, languages, reconstructed_audio, reconstructed_sr
        )
