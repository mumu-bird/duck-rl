"""Import bundled phrases once; never overwrite the user's replacements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .client import AudioClient
from .errors import AudioError


def seed(client, directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    results = []
    for item in manifest["clips"]:
        if client.has_clip(item["key"]):
            results.append({"key": item["key"], "action": "kept"})
            continue
        source = directory / item["file"]
        if source.parent != directory or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise AudioError("默认提示音文件校验失败: " + item["key"])
        try:
            client.import_file(source, key=item["key"], name=item["name"])
        except AudioError as exc:
            if exc.code != "duplicate_key": raise
        results.append({"key": item["key"], "action": "imported"})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="unix:/run/wonderduck-audio/audio.sock")
    parser.add_argument("--token-file")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent.parent / "assets/audio")
    args = parser.parse_args(argv)
    print(json.dumps(seed(AudioClient(args.endpoint, token_file=args.token_file), args.directory), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
