"""Build the consumer's portable stdlib-only field entry point. No model-slot installer."""

import argparse
import hashlib
from pathlib import Path
import zipfile

if __package__:
    from . import accept
else:
    import accept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    dest = parser.parse_args().out
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    data = {'__main__.py': b'import accept\nraise SystemExit(accept.main())\n',
            'accept.py': (src / 'accept.py').read_bytes(),
            'readonly_preflight.py': (src / 'readonly_preflight.py').read_bytes(),
            'board-replay.py': accept.program_bytes()}
    with zipfile.ZipFile(dest, 'x') as z:
        for name, content in data.items():
            z.writestr(zipfile.ZipInfo(name), content, compress_type=zipfile.ZIP_DEFLATED)
    print(dest, hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
