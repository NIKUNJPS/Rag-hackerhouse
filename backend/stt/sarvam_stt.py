import requests
from backend.stt.base import STTProvider, STTError
from backend.config import settings

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT(STTProvider):
    def __init__(self, api_key: str | None = None, language_code: str = "en-IN"):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.language_code = language_code
        if not self.api_key:
            raise STTError("SARVAM_API_KEY is not set")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        headers = {"api-subscription-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"language_code": self.language_code, "model": "saarika:v2"}

        try:
            resp = requests.post(
                SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=15
            )
        except requests.RequestException as e:
            raise STTError(f"Sarvam request failed: {e}") from e

        if resp.status_code != 200:
            raise STTError(f"Sarvam STT failed ({resp.status_code}): {resp.text}")

        payload = resp.json()
        text = payload.get("transcript", "").strip()
        if not text:
            raise STTError("Sarvam returned an empty transcript")
        return text
