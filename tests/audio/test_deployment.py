from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from duckaudio.board_config import shared_player_config
from duckaudio.config import tomllib

ROOT = Path(__file__).resolve().parents[2]


class DeploymentTests(unittest.TestCase):
    def test_legacy_config_only_changes_player(self):
        before = '# keep comments\n[robot]\nx = 4\n[audio]\nenabled = true\nplayer = "aplay" # old\n[vision]\nmode = "normal"\n'
        after = shared_player_config(before, "/robot/scripts/audio-output.py")
        expected = tomllib.loads(before)
        expected["audio"]["player"] = "/robot/scripts/audio-output.py"
        self.assertEqual(tomllib.loads(after), expected)
        self.assertIn("# keep comments", after)
        self.assertEqual(shared_player_config(after, expected["audio"]["player"]),after)
        self.assertEqual(tomllib.loads(shared_player_config("[robot]\nx=4", "/x"))["audio"]["player"], "/x")

    def test_staged_install_update_rollback_and_uninstall_preserves_data(self):
        with tempfile.TemporaryDirectory(prefix="duckaudio-install-") as tmp:
            directory = Path(tmp)
            package = directory / "package"
            for folder in ("duckaudio", "deploy/audio", "assets/audio", "docs/audio", "examples/audio"):
                shutil.copytree(ROOT / folder, package / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            stage = directory / "stage"
            original = '# original board config\n[audio]\nenabled=true\n[robot]\nname="duck"\n'
            board = stage / "etc/robot/robotd.toml"
            board.parent.mkdir(parents=True);board.write_text(original)
            script = package / "deploy/audio/install.py"
            def run(action, *options):
                result = subprocess.run([sys.executable,str(script),action,"--root",str(stage),*options],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return result.stdout
            run("install")
            current = stage / "opt/wonderduck-audio/current"
            first = current.resolve()
            self.assertTrue((current/"duckaudio/web/index.html").is_file())
            self.assertEqual(tomllib.loads(board.read_text())["audio"]["player"],"/home/wonderduck/wonderduck/scripts/audio-output.py")
            state = stage / "var/lib/wonderduck-audio"
            (state / "keep-user-audio.wav").write_bytes(b"user file")
            (package / "duckaudio/__init__.py").write_text((package / "duckaudio/__init__.py").read_text()+"\n# deployment test second version\n")
            run("install")
            second = current.resolve()
            self.assertNotEqual(first,second)
            run("rollback")
            self.assertEqual(current.resolve(),first)
            run("uninstall")
            self.assertEqual(board.read_text(),original)
            self.assertEqual((state/"keep-user-audio.wav").read_bytes(),b"user file")
            self.assertFalse((stage / "etc/systemd/system/wonderduck-audio.service").exists())
            self.assertFalse((stage / "etc/systemd/system/wonderduck-prompt-sounds.service.d/60-duckaudio.conf").exists())
            self.assertTrue((current/"duckaudio/__init__.py").exists())

    def test_uninstall_keeps_edited_legacy_config(self):
        with tempfile.TemporaryDirectory(prefix="duckaudio-install-edited-") as tmp:
            stage = Path(tmp)
            board = stage/"etc/robot/robotd.toml"
            board.parent.mkdir(parents=True);board.write_text("[audio]\nenabled=true\n")
            script = ROOT/"deploy/audio/install.py"
            result=subprocess.run([sys.executable,str(script),"install","--root",str(stage)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            changed=board.read_text()+"\n[robot]\ncustom=true\n";board.write_text(changed)
            result=subprocess.run([sys.executable,str(script),"uninstall","--root",str(stage)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(board.read_text(),changed)
            self.assertTrue((stage/"var/lib/wonderduck-audio/robotd.toml.before-audio").exists())


if __name__ == "__main__":
    unittest.main()
