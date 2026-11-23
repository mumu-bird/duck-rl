"""Build a no-actuation stdlib-only board parity test from the original frozen actor."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

if __package__:
    from . import board_shadow
    from .contract import FROZEN, SHA, TransferEnv, load_actor, torch
else:
    import board_shadow
    from contract import FROZEN, SHA, TransferEnv, load_actor, torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    actor = load_actor()
    results = json.loads((FROZEN.parent / 'assessment/paired.json').read_text())['after']
    if results['sha256'] != SHA:
        raise ValueError('original frozen reference differs')
    corpus = dict(source_sha256=SHA, phases=[], cases=[])
    env = TransferEnv('trained_5ms')
    with torch.inference_mode():
        for network in actor.actors:
            corpus['phases'].append([[network[n].weight.tolist(), network[n].bias.tolist()] for n in (0, 2)])
        for row in sorted(results['rows'], key=lambda r: r['seed']):
            obs = env.reset(row['seed'])
            for phase in (0, 1):
                action = actor.dist(torch.from_numpy(obs), phase).loc.tanh().numpy()
                corpus['cases'].append([row['seed'], phase, obs.tolist(), action.tolist()])
                obs, _, _, info = env.step(action)
            if {'seed': row['seed'], **info} != row:
                raise ValueError(f"physics replay differs for seed {row['seed']}")
    parity = board_shadow.verify(corpus, False)
    if parity['code'] != 'HOST_PARITY_ONLY':
        raise ValueError('export parity check failed')
    raw = json.dumps(corpus, separators=(',', ':'), allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    template = Path(board_shadow.__file__).read_text()
    script = template.replace('PAYLOAD_B85 = b""', 'PAYLOAD_B85 = ' + repr(base64.b85encode(zlib.compress(raw, 9))))
    script = script.replace('PAYLOAD_SHA256 = ""', f'PAYLOAD_SHA256 = "{digest}"')
    dest = args.out / 'board-replay.py'
    dest.write_bytes(script.encode('utf-8'))
    manifest = dict(source_sha256=SHA, script_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                    payload_sha256=digest, on_board=False, replayed_episodes=100, local=parity)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
