"""Provider layer. The rest of the pipeline only uses two methods:

    llm.complete(system, user, temperature=..., max_tokens=..., reasoning=...) -> str
    tts.synthesize(text, voice=..., language=...) -> bytes  (any audio ffmpeg can read)

Bodhan's hosted APIs use OpenAI-compatible request shapes, so both clients follow
the OpenAI format. The TTS request matches the Bodhan API reference (Text to Speech).
"""
import io
import json
import math
import struct
import time
import wave

import requests


class ProviderError(RuntimeError):
    pass


def _post(url: str, key: str, body: dict, timeout: int = 180, retries: int = 6) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last_error = None
    for attempt in range(retries):
        wait = 2 * 2 ** attempt  # 2s, 4s, 8s ...
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in (408, 429, 500, 502, 503, 504):
                raise ProviderError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
            last_error = ProviderError(f"{resp.status_code}: {resp.text[:300]}")
            if resp.status_code == 429:  # Bodhan's quota is per minute (~8 TTS calls), so wait out the window
                wait = max(wait, float(resp.headers.get("Retry-After") or 20))
        time.sleep(wait)
    raise ProviderError(f"{url} failed after {retries} attempts: {last_error}")


class OpenAICompatibleLLM:
    """Any chat endpoint that speaks the OpenAI /chat/completions format."""

    def __init__(self, base_url: str, api_key: str, model: str):
        if not base_url or not model:
            raise ProviderError("Set LLM_BASE_URL and LLM_MODEL in .env (see .env.example)")
        self.url = f"{base_url}/chat/completions"
        self.key = api_key
        self.model = model

    def complete(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 4000,
                 reasoning: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Thinking models reason by default; at full effort DeepSeek once spent 16k tokens on one
        # verify call. So: no reasoning for writing, low effort for edit and verify.
        # ponytail: OpenRouter's field; other hosts keep their default.
        if "openrouter.ai" in self.url:
            body["reasoning"] = {"effort": "low"} if reasoning else {"enabled": False}
        data = _post(self.url, self.key, body).json()
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"Unexpected LLM response shape: {str(data)[:300]}") from exc
        if not content and choice.get("finish_reason") == "length":
            raise ProviderError(f"The LLM used all {max_tokens} tokens without answering")
        return content or ""  # an occasional empty reply is retried by ask_json


class BodhanTTS:
    """Bodhan Indic-Speak: POST /v1/audio/speech, returns WAV (PCM16, 24 kHz, mono).
    voice is required (any voice reads any language, e.g. Kavya, Arun), and the language
    goes in `instructions` as a JSON string: '{"lang": "en"}'."""

    def __init__(self, base_url: str, api_key: str, model: str):
        if not api_key:
            raise ProviderError("Set BODHAN_TTS_KEY in .env (Bodhan issues a separate key per model)")
        self.url = f"{base_url}/audio/speech"
        self.key = api_key
        self.model = model

    def synthesize(self, text: str, voice: str, language: str) -> bytes:
        body = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "instructions": json.dumps({"lang": language}),
        }
        resp = _post(self.url, self.key, body)
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            raise ProviderError(f"Expected audio but got JSON: {resp.text[:300]}")
        return resp.content


class MockTTS:
    """Offline stand-in that costs nothing: a short beep, then silence roughly as long
    as the line would take to speak (150 words per minute). Host A beeps higher than
    host B, so you can check caption sync by ear before spending real TTS credit."""

    def synthesize(self, text: str, voice: str, language: str) -> bytes:
        rate = 24000
        seconds = max(0.8, len(text.split()) / 2.5)
        freq = 660 if voice.endswith("A") else 440
        frames = bytearray()
        for i in range(int(seconds * rate)):
            t = i / rate
            sample = 0.25 * math.sin(2 * math.pi * freq * t) if t < 0.15 else 0.0
            frames += struct.pack("<h", int(sample * 32767))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(frames))
        return buf.getvalue()


def build_llm(settings) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model)


def build_tts(settings, mock: bool = False):
    """Returns (tts_client, voices) where voices maps speaker 'A'/'B' to a voice id."""
    if mock:
        return MockTTS(), {"A": "mock-A", "B": "mock-B"}
    if not (settings.voice_a and settings.voice_b):
        raise ProviderError("Set BODHAN_VOICE_A and BODHAN_VOICE_B in .env (e.g. Kavya and Arun)")
    if settings.voice_a == settings.voice_b:
        print("Warning: BODHAN_VOICE_A and BODHAN_VOICE_B are the same; the hosts will sound identical.")
    tts = BodhanTTS(settings.tts_base_url, settings.tts_api_key, settings.tts_model)
    return tts, {"A": settings.voice_a, "B": settings.voice_b}
