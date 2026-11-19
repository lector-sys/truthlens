"""OpenAI calls. Model ID from env because the Luna ID isn't confirmed: set TRUTHLENS_MODEL=gpt-5.6-luna (or gpt-6-luna).
Returns None when no key is set, so the CPU pipeline still runs offline."""
import base64, json, mimetypes, os

MODEL = os.getenv("TRUTHLENS_MODEL", "gpt-5.6-luna")
_client = None


def client():
    global _client
    if _client is None and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        _client = OpenAI()
    return _client


def image_part(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def ask_json(system, user, images=()):
    """One call, JSON object out. `images` are file paths."""
    c = client()
    if not c:
        return None
    content = [{"type": "text", "text": user}] + [image_part(p) for p in images]
    r = c.chat.completions.create(model=MODEL, response_format={"type": "json_object"},
                                  messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])
    return json.loads(r.choices[0].message.content)


def transcribe(path):
    """OWNER B: audio -> segments with timestamps. Luna can't take audio; this uses OpenAI's transcription endpoint."""
    c = client()
    if not c:
        return None
    with open(path, "rb") as f:
        r = c.audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json",
                                          timestamp_granularities=["segment"])
    return [{"start": s.start, "end": s.end, "text": s.text} for s in r.segments]
