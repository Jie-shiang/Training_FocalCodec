#!/usr/bin/env python3
"""
Configuration loader for FocalCodec 25Hz training.

Loads settings from config.yaml and provides easy access to all parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for FocalCodec training."""

    def __init__(self, config_path: Optional[str] = None):
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default: config.yaml in project root
            config_path = Path(__file__).parent.parent / "config.yaml"

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Resolve path variables
        self._resolve_paths()

    def _resolve_paths(self):
        """Resolve ${paths.xxx} variables in config."""
        paths = self._config.get('paths', {})
        base_dir = paths.get('base_dir', '')

        # Resolve data paths
        data = self._config.get('data', {})
        for key, value in data.items():
            if isinstance(value, str) and '${paths.base_dir}' in value:
                data[key] = value.replace('${paths.base_dir}', base_dir)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation.

        Args:
            key: Dot-separated key (e.g., 'paths.base_dir', 'training.batch_size')
            default: Default value if key not found

        Returns:
            Config value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    # Path shortcuts
    @property
    def base_dir(self) -> str:
        return self.get('paths.base_dir')

    @property
    def focalcodec_dir(self) -> str:
        return self.get('paths.focalcodec_dir')

    @property
    def model_cache_dir(self) -> str:
        return self.get('paths.model_cache_dir')

    @property
    def asr_cache_dir(self) -> str:
        return self.get('paths.asr_cache_dir')

    @property
    def output_dir(self) -> str:
        return self.get('paths.output_dir')

    @property
    def inference_dir(self) -> str:
        return self.get('paths.inference_dir')

    @property
    def audio_base_path(self) -> str:
        return self.get('data.audio_base_path')

    @property
    def train_csv(self) -> str:
        return self.get('data.train_csv')

    @property
    def val_csv(self) -> str:
        return self.get('data.val_csv')

    @property
    def test_csv(self) -> str:
        return self.get('data.test_csv')

    # Model shortcuts
    @property
    def base_model(self) -> str:
        return self.get('model.base_model')

    @property
    def teacher_model(self) -> str:
        return self.get('model.teacher_model')

    @property
    def codebook_size(self) -> int:
        return self.get('model.codebook_size')

    @property
    def frame_rate(self) -> int:
        return self.get('model.frame_rate')

    # Training shortcuts
    @property
    def stage(self) -> int:
        return self.get('training.stage')

    @property
    def gpu_id(self) -> int:
        return self.get('training.gpu_id')

    @property
    def batch_size(self) -> int:
        return self.get('training.batch_size')

    @property
    def num_workers(self) -> int:
        return self.get('training.num_workers')

    @property
    def num_epochs(self) -> int:
        return self.get('training.num_epochs')

    @property
    def learning_rate(self) -> float:
        return self.get('training.learning_rate')

    @property
    def weight_decay(self) -> float:
        return self.get('training.weight_decay')

    @property
    def gradient_clip(self) -> float:
        return self.get('training.gradient_clip')

    @property
    def chunk_duration(self) -> float:
        return self.get('training.chunk_duration')

    @property
    def overlap(self) -> float:
        return self.get('training.overlap')

    @property
    def patience(self) -> int:
        return self.get('training.patience')

    @property
    def min_delta(self) -> float:
        return self.get('training.min_delta')

    # Loss shortcuts
    @property
    def weight_feature(self) -> float:
        return self.get('loss.weight_feature')

    @property
    def weight_asr(self) -> float:
        return self.get('loss.weight_asr')

    @property
    def weight_hubert(self) -> float:
        return self.get('loss.weight_hubert')

    def get_checkpoint_dir(self, stage: Optional[int] = None) -> str:
        """Get checkpoint directory for a stage."""
        if stage is None:
            stage = self.stage
        return os.path.join(self.output_dir, f"stage_{stage}")

    def get_inference_dir(self, stage: Optional[int] = None) -> str:
        """Get inference output directory for a stage."""
        if stage is None:
            stage = self.stage
        return os.path.join(self.inference_dir, f"stage_{stage}")

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Config object
    """
    return Config(config_path)


# For shell scripts: print config values
if __name__ == '__main__':
    import sys

    config = load_config()

    if len(sys.argv) > 1:
        # Print specific key
        key = sys.argv[1]
        value = config.get(key)
        if value is not None:
            print(value)
        else:
            print(f"Key not found: {key}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print all config
        print(yaml.dump(config._config, default_flow_style=False))
