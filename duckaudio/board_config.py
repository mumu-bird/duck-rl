"""Conservative TOML edit for the legacy player's shared amplifier wrapper."""
from __future__ import annotations

import json
import re

from .config import tomllib
from .errors import AudioError


def shared_player_config(text, player):
    parsed = tomllib.loads(text)
    if parsed.get("audio", {}).get("player") == player:
        return text
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.fullmatch(r"\s*\[audio\]\s*(?:#.*)?\n?", line)), None)
    assignment = "player = " + json.dumps(player, ensure_ascii=False) + "  # shared amplifier lock\n"
    if start is None:
        if "audio" in parsed:
            raise AudioError("原厂 audio 使用非标准 TOML 写法，请人工设置 player", "unsupported_config")
        result = text.rstrip() + "\n\n[audio]\n" + assignment
    else:
        end = next((i for i in range(start+1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        position = next((i for i in range(start+1, end) if re.match(r"\s*player\s*=", lines[i])), None)
        if position is None:
            if start == len(lines)-1 and not lines[start].endswith("\n"):
                lines[start] += "\n"
            lines.insert(start+1, assignment)
        else:
            lines[position] = assignment
        result = "".join(lines)
    expected = dict(parsed)
    expected["audio"] = {**parsed.get("audio", {}), "player": player}
    if tomllib.loads(result) != expected:
        raise AudioError("原厂配置编辑影响了其他字段，已拒绝修改", "unsupported_config")
    return result
