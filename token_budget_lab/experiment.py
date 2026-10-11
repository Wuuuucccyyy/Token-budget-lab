"""Fingerprints and paths for the current plugin evaluation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hash():
    return stable_hash({p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})

