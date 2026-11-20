"""Evidence vault: hash on ingest, read-only working copy, append-only case log."""
import hashlib, json, os, re, shutil, stat, time
from pathlib import Path

CASES = Path(__file__).resolve().parent.parent / "cases"
PUBLIC_CASES = CASES.parent / "cases_public"  # public demo mode: cases_public/<session tag>/pub-<session tag>-<time>


def case_dir(case_id):
    m = re.fullmatch(r"pub-([0-9a-f]{16})-[0-9_a-f]+", case_id)
    return PUBLIC_CASES / m[1] / case_id if m else CASES / case_id


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Case:
    def __init__(self, case_id=None):
        self.id = case_id or time.strftime("case_%Y%m%d_%H%M%S")
        self.dir = case_dir(self.id)
        (self.dir / "evidence").mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        self.log_path = self.dir / "log.jsonl"

    def log(self, action, **details):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "action": action, **details}) + "\n")

    def ingest(self, name, data: bytes):
        """Store an uploaded file read-only and record its hash. Returns the stored path."""
        dest = self.dir / "evidence" / Path(name).name
        dest.write_bytes(data)
        os.chmod(dest, stat.S_IREAD)
        digest = sha256(dest)
        self.log("ingest", file=dest.name, sha256=digest, bytes=len(data))
        return dest

    def evidence(self):
        return sorted((self.dir / "evidence").iterdir())

    def verify(self):
        """Re-hash evidence and compare with the ingest log. {file: ok}"""
        logged = {e["file"]: e["sha256"] for e in self.entries() if e["action"] == "ingest"}
        return {p.name: sha256(p) == logged.get(p.name) for p in self.evidence()}

    def entries(self):
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines()]

    def artifact_path(self, name):
        return self.dir / "artifacts" / name

    def delete(self):
        for p in self.evidence():
            os.chmod(p, stat.S_IWRITE)
        shutil.rmtree(self.dir)
