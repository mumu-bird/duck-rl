"""Build a standalone read-only retrieval field package using frozen assets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    src = Path(__file__).resolve().parent
    shared = src.parent / "selfcatch_field"
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    data = {name: (args.bundle / name).read_bytes() for name in manifest["files"]}
    for name, raw in data.items():
        if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
            raise ValueError("input export checksum mismatch: " + name)
    data.update({name: (src / name).read_bytes() for name in ("shadow.py", "field_check.py")})
    data.update({name: (shared / name).read_bytes()
                 for name in ("readonly_preflight.py", "camera_probe.py", "camera_board.py")})
    data["__main__.py"] = b"import field_check\nraise SystemExit(field_check.main())\n"
    manifest["files"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
    data["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in data.items():
            archive.writestr(name, raw)
    print(args.out, hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
