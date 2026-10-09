import json

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
