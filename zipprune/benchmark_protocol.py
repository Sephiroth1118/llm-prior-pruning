"""Provenance used to reject stale evaluation checkpoints."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_signature(path, files=None):
    """Hash a local artifact manifest, not gigabytes of model weights.

    Small files/config JSON are hashed. Large weights use size/mtime/ctime;
    this assumes weight files are immutable or their metadata changes on write.
Remote identifiers remain recorded by name in the enclosing protocol.
"""
    if path is None or not Path(path).is_dir():
        return None
    root = Path(path)
    if files is None:
        files = [p.relative_to(root) for p in root.iterdir()
                 if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".pth"}]
    manifest = []
    for relative in sorted(str(p) for p in files):
        item = root / relative
        stat = item.stat()
        manifest.append((relative, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                         file_sha256(item) if item.suffix == ".json" or stat.st_size <= 1024 * 1024 else None))
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def checkpoint_protocol_matches(checkpoint, protocol):
    existing = checkpoint.get("protocol")
    return isinstance(existing, dict) and existing == protocol
