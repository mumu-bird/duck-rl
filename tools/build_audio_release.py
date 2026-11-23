#!/usr/bin/env python3
"""Build a source-only board release: no desktop venv or architecture-specific binaries."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = []
    for folder in ("duckaudio", "deploy/audio", "assets/audio", "docs/audio", "examples/audio"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files += [ROOT / "pyproject.toml"]
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / "wonderduck-audio-1.0.0.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for path in sorted(files):
            out.add(path, arcname="wonderduck-audio-1.0.0/" + str(path.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / (archive.name + ".sha256")).write_text(digest + "  " + archive.name + "\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest, "files": len(files)}, indent=2))

if __name__ == "__main__":
    main()
