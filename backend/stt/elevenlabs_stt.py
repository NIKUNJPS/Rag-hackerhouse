import requests
from backend.stt.base import STTProvider, STTError
from backend.config import settings

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsSTT(STTProvider):
    def __init__(self, api_key: str | None = None, model_id: str = "scribe_v1"):
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.model_id = model_id
        if not self.api_key:
            raise STTError("ELEVENLABS_API_KEY is not set")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        headers = {"xi-api-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model_id": self.model_id}

        try:
            resp = requests.post(
                ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=15
            )
        except requests.RequestException as e:
            raise STTError(f"ElevenLabs request failed: {e}") from e

        if resp.status_code != 200:
            raise STTError(f"ElevenLabs STT failed ({resp.status_code}): {resp.text}")

        payload = resp.json()
        text = payload.get("text", "").strip()
        if not text:
            raise STTError("ElevenLabs returned an empty transcript")
        return text
