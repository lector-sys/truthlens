"""Public demo mode (TRUTHLENS_PUBLIC=1): access code, per-session cases under cases_public/, upload limits, 24 h
auto-delete, one analysis at a time, and scrubbing of host names / addresses / paths from anything shown.
Off (the default), nothing here is used. Display and intake only: no analysis, weight or label changes here."""
import hashlib, hmac, os, re, secrets, shutil, stat, tempfile, threading, time
from pathlib import Path
from . import llm  # loads .env first
from .analyzers.media_meta import ffprobe
from .vault import CASES, PUBLIC_CASES as ROOT, case_dir

PUBLIC = os.getenv("TRUTHLENS_PUBLIC") == "1"
ACCESS_CODE = os.getenv("TRUTHLENS_ACCESS_CODE", "")
DEMOS = ("demo1", "demo2", "demo3")
MAX_MB, MAX_FILES, MAX_SECONDS = 50, 3, 120
PER_SESSION_HOUR, PER_IP_DAY = 5, 30
MIN_FREE_GB, KEEP_H = 10, 24
CODE_TRIES, CODE_WINDOW_S = 5, 600
LIMITS = (f"Up to {MAX_FILES} files per case, {MAX_MB} MB each · video and audio up to {MAX_SECONDS} s · "
          f"{PER_SESSION_HOUR} cases per hour per visitor · files are deleted after {KEEP_H} hours")
NOTICE = ("Demo service. Don't upload private or sensitive material. Files are deleted after 24 hours. "
          "Results are automated indicators, not proof.")

# ponytail: in-memory counters, reset on restart and per process; a shared store if this ever runs as several processes
_lock = threading.Lock()
_tries, _made_ip, _made_sid = {}, {}, {}


def _recent(d, key, window):
    now = time.time()
    d[key] = [t for t in d.get(key, []) if now - t < window]
    return d[key]


def check_code(given, ip):
    """(ok, message). Constant-time compare; at most CODE_TRIES tries per IP per CODE_WINDOW_S."""
    with _lock:
        tries = _recent(_tries, ip, CODE_WINDOW_S)
        if len(tries) >= CODE_TRIES:
            return False, "Too many tries. Wait 10 minutes and try again."
        tries.append(time.time())
    if hmac.compare_digest(given.encode(), ACCESS_CODE.encode()):
        return True, ""
    return False, "That code is not right."


def new_session_id():
    return secrets.token_urlsafe(24)


def _tag(sid):
    return hashlib.sha256(sid.encode()).hexdigest()[:16]


def new_case_id(sid):
    return f"pub-{_tag(sid)}-{time.strftime('%Y%m%d_%H%M%S')}{secrets.token_hex(2)}"


def visible(case_id, sid):
    """Can this session see the case? Demo cases, and cases this session created; nothing else."""
    return case_id in DEMOS or case_id.startswith(f"pub-{_tag(sid)}-")


def session_cases(sid):
    d = ROOT / _tag(sid)
    return sorted((p.name for p in d.iterdir() if (p / "result.json").exists()), reverse=True) if d.exists() else []


def upload_problems(files, sid, ip):
    """Reasons to refuse an upload ([] = accept). files: [(name, bytes)]."""
    out = []
    if len(files) > MAX_FILES:
        out.append(f"At most {MAX_FILES} files per case (you added {len(files)}).")
    for name, data in files:
        if len(data) > MAX_MB * 2**20:
            out.append(f"{name} is {len(data) / 2**20:.0f} MB; the limit is {MAX_MB} MB.")
            continue
        if Path(name).suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            secs = duration(data, Path(name).suffix)
            if secs is None:
                out.append(f"{name} could not be read as video or audio.")
            elif secs > MAX_SECONDS:
                out.append(f"{name} is {secs:.0f} s long; video and audio may be at most {MAX_SECONDS} s.")
    if shutil.disk_usage(ROOT if ROOT.exists() else CASES.parent).free < MIN_FREE_GB * 2**30:
        out.append("The demo is not accepting uploads right now (storage is full). Please try again later.")
    with _lock:
        if len(_recent(_made_sid, sid, 3600)) >= PER_SESSION_HOUR:
            out.append(f"At most {PER_SESSION_HOUR} cases per hour. Please try again later.")
        if len(_recent(_made_ip, ip, 86400)) >= PER_IP_DAY:
            out.append(f"At most {PER_IP_DAY} cases per day from one address. Please try again tomorrow.")
    return out


def count_case(sid, ip):
    with _lock:
        _made_sid.setdefault(sid, []).append(time.time())
        _made_ip.setdefault(ip, []).append(time.time())


def duration(data, suffix):
    """Seconds of a video/audio upload by ffprobe (via a temp copy), or None if unreadable."""
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        d = ffprobe(tmp).get("format", {}).get("duration")
        return float(d) if d else None
    finally:
        os.remove(tmp)


def created(d):
    return d.stat().st_ctime if os.name == "nt" else min(d.stat().st_mtime, (d / "log.jsonl").stat().st_mtime
                                                           if (d / "log.jsonl").exists() else time.time())


def expires(case_id):
    d = case_dir(case_id)
    return created(d) + KEEP_H * 3600 if case_id.startswith("pub-") and d.exists() else None


def cleanup(now=None):
    """Delete public cases older than KEEP_H hours (evidence is read-only, so unlock first). Returns the ids deleted."""
    now, gone = now or time.time(), []
    for d in ROOT.glob("*/pub-*"):
        if now - created(d) > KEEP_H * 3600:
            for p in d.rglob("*"):
                if p.is_file():
                    os.chmod(p, stat.S_IWRITE)
            shutil.rmtree(d, ignore_errors=True)
            gone.append(d.name)
    for s in ROOT.glob("*"):
        if s.is_dir() and not any(s.iterdir()):
            s.rmdir()
    return gone


def start_cleanup():
    """Cleanup now and then every hour, in a daemon thread (call once per process)."""
    def loop():
        while True:
            cleanup()
            time.sleep(3600)
    threading.Thread(target=loop, daemon=True, name="public-cleanup").start()


# one analysis at a time: a FIFO of job ids
_turn = threading.Condition()
_queue = []


def wait_turn(jid):
    with _turn:
        _queue.append(jid)
        _turn.wait_for(lambda: _queue[0] == jid)


def done(jid):
    with _turn:
        if jid in _queue:
            _queue.remove(jid)
        _turn.notify_all()


def position(jid):
    """0 = running now, n = n analyses ahead in the queue, None = not queued."""
    with _turn:
        return _queue.index(jid) if jid in _queue else None


def _hosts():
    names = set()
    for u in (llm.OLLAMA_URL, llm.WHISPER_URL, llm.DETECTORS_URL):
        h = llm.urlparse(u).hostname if u else None
        if h:
            names |= {h, h.split(".")[0]}
    return sorted(names, key=len, reverse=True)


_PATTERNS = [(re.compile(r"https?://[^\s<>\"')]+"), "[server]"), (re.compile(r"\b[\w-]+(\.[\w-]+)*\.ts\.net\b"), "[server]"),
             (re.compile(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b"), "[address]"),
             (re.compile(r"[A-Za-z]:\\[^\s<>\"']+"), "[path]"), (re.compile(r"(?<![\w.])/(?:home|Users|tmp|var|root|mnt|srv)/[^\s<>\"']+"), "[path]")]


def scrub(text):
    """Text with server URLs, host names, IP addresses and file-system paths replaced (public mode only)."""
    for rx, sub in _PATTERNS:
        text = rx.sub(sub, text)
    for h in _hosts():
        text = re.sub(rf"\b{re.escape(h)}\b", "[server]", text)
    return text
