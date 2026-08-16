from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Common interface so the pipeline doesn't care which STT vendor is behind it."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Takes raw audio bytes, returns transcript text. Raises STTError on failure."""
        raise NotImplementedError


class STTError(Exception):
    pass
