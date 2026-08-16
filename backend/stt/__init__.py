from backend.config import settings
from backend.stt.base import STTProvider, STTError


def get_stt_provider() -> STTProvider:
    if settings.STT_PROVIDER == "elevenlabs":
        from backend.stt.elevenlabs_stt import ElevenLabsSTT
        return ElevenLabsSTT()
    elif settings.STT_PROVIDER == "sarvam":
        from backend.stt.sarvam_stt import SarvamSTT
        return SarvamSTT()
    else:
        raise STTError(f"Unknown STT_PROVIDER: {settings.STT_PROVIDER}")
