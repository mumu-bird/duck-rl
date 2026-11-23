"""Task-oriented audio for WonderDuck; no simulator dependencies."""
__version__ = "1.0.0"

from .client import AudioClient, AudioError, Playback

__all__ = ["AudioClient", "AudioError", "Playback"]
