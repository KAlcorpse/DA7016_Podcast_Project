"""Settings come from environment variables or a local .env file.

Credentials stay on the server side only. Never ship them to the browser
and never commit .env to git (it is already in .gitignore).
"""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env reader so the project has no extra dependency for it."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    tts_base_url: str
    tts_api_key: str
    tts_model: str
    voice_a: str
    voice_b: str
    tts_language: str
    script_language: str
    host_a_name: str
    host_b_name: str
    tts_max_chars: int
    tts_workers: int


def load_settings() -> Settings:
    _load_dotenv()
    env = os.environ.get
    return Settings(
        llm_base_url=env("LLM_BASE_URL", "").rstrip("/"),
        llm_api_key=env("LLM_API_KEY", ""),
        llm_model=env("LLM_MODEL", ""),
        tts_base_url=env("BODHAN_TTS_BASE_URL", "https://api.bodhan.ai/v1").rstrip("/"),
        tts_api_key=env("BODHAN_TTS_KEY", ""),
        tts_model=env("BODHAN_TTS_MODEL", "indic-speak"),
        voice_a=env("BODHAN_VOICE_A", ""),
        voice_b=env("BODHAN_VOICE_B", ""),
        tts_language=env("PODCAST_TTS_LANGUAGE", "en"),
        script_language=env("PODCAST_SCRIPT_LANGUAGE", "English"),
        host_a_name=env("HOST_A_NAME", "Meera"),
        host_b_name=env("HOST_B_NAME", "Arjun"),
        tts_max_chars=int(env("TTS_MAX_CHARS", "250")),
        tts_workers=int(env("TTS_WORKERS", "3")),
    )
