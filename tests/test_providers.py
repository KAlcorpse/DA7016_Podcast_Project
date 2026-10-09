import json

import pytest

import providers


def test_bodhan_request_matches_api_reference(monkeypatch):
    sent = {}

    class Resp:
        headers = {"Content-Type": "audio/wav"}
        content = b"RIFF..."

    def fake_post(url, key, body):
        sent.update(url=url, key=key, body=body)
        return Resp()

    monkeypatch.setattr(providers, "_post", fake_post)
    tts = providers.BodhanTTS("https://api.bodhan.ai/v1", "k", "indic-speak")
    assert tts.synthesize("Hello.", voice="Kavya", language="en") == b"RIFF..."
    assert sent["url"] == "https://api.bodhan.ai/v1/audio/speech"
    assert sent["body"] == {"model": "indic-speak", "input": "Hello.", "voice": "Kavya", "instructions": '{"lang": "en"}'}
    assert json.loads(sent["body"]["instructions"]) == {"lang": "en"}


def _llm_reply(monkeypatch, choice):
    class Resp:
        def json(self):
            return {"choices": [choice]}

    sent = {}
    monkeypatch.setattr(providers, "_post", lambda url, key, body: sent.update(body) or Resp())
    return providers.OpenAICompatibleLLM("https://openrouter.ai/api/v1", "k", "m"), sent


def test_llm_reasoning_flag_and_empty_replies(monkeypatch):
    llm, sent = _llm_reply(monkeypatch, {"message": {"content": None}, "finish_reason": "stop"})
    assert llm.complete("s", "u", reasoning=True) == ""  # empty reply -> ask_json retries
    assert sent["reasoning"] == {"effort": "low"}

    llm, _ = _llm_reply(monkeypatch, {"message": {"content": None}, "finish_reason": "length"})
    with pytest.raises(providers.ProviderError, match="tokens"):
        llm.complete("s", "u")


def test_post_waits_out_rate_limits(monkeypatch):
    class Resp:
        def __init__(self, status, headers=None):
            self.status_code, self.headers, self.text = status, headers or {}, "x"

    replies = iter([Resp(429, {"Retry-After": "7"}), Resp(429), Resp(503), Resp(200)])
    sleeps = []
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: next(replies))
    monkeypatch.setattr(providers.time, "sleep", sleeps.append)
    assert providers._post("u", "k", {}).status_code == 200
    assert sleeps == [7.0, 20.0, 8]

    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: Resp(401))
    with pytest.raises(providers.ProviderError, match="401"):
        providers._post("u", "k", {})
