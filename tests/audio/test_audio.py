from __future__ import annotations

import array
import fcntl
import types
import sys
import concurrent.futures
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import wave

from duckaudio.client import AudioClient, AudioError
from duckaudio.config import Config
from duckaudio.engine import PlaybackEngine
from duckaudio.media import MediaLibrary, run_process, terminate_process
from duckaudio.recording import Recorder
from duckaudio.server import Application
from duckaudio.store import Store
from duckaudio.tasks import LogAudioClient, MouthFollower, TaskAudio

ROOT = Path(__file__).resolve().parents[2]
TTS = ROOT / "vendor/audio-tools/usr/bin/espeak-ng"


def tone(path, seconds=.25, rate=48000):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(b"".join(struct.pack("<h", int(5000 * math.sin(i * 440 * math.tau / rate))) for i in range(int(seconds*rate))))
    return path


def eventually(fn, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = fn()
        if result:
            return result
        time.sleep(.015)
    raise AssertionError("condition did not become true")


class AudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="duckaudio-test-")
        self.path = Path(self.temp.name)
        self.config = Config(data_dir=str(self.path), socket=str(self.path/"audio.sock"),
                             lock_file=str(self.path/"pcm.lock"), backend="null", host="127.0.0.1", port=0,
                             tts_program=str(TTS) if TTS.exists() else "espeak-ng").validate()
        self.app = Application(self.config).start()
        self.client = AudioClient("unix:"+self.config.socket)
        self.http = AudioClient("http://127.0.0.1:"+str(self.app.http.server_port), token=self.app.token)
        self.source = tone(self.path/"source.wav")

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def clip(self, key="test", seconds=.25):
        source=tone(self.path/(key+".wav"), seconds)
        return self.client.import_file(source, key=key, name="测试音频")

    def test_authentication_origin_and_hardware_acceptance(self):
        unauth = AudioClient(self.http.endpoint)
        with self.assertRaises(AudioError) as caught:
            unauth.health()
        self.assertEqual(caught.exception.status,401)
        self.assertTrue(self.http.health()["simulated"])
        with self.assertRaises(AudioError) as caught:
            self.http.request("POST","/api/acceptance",{"id":"speaker","result":"pass"})
        self.assertEqual(caught.exception.code,"hardware_unverified")
        result=self.http.request("POST","/api/acceptance",{"id":"library","result":"pass","notes":"真实导入与下载"})
        self.assertEqual(next(x for x in result["items"] if x["id"]=="library")["result"],"pass")

    def test_import_rename_download_delete_and_unicode(self):
        clip=self.http.import_file(self.source,key="task.target_found",name="找到目标 🦆")
        self.assertTrue(self.client.has_clip("task.target_found"))
        self.assertEqual(self.client.rename(clip["id"],"新的名称",key="renamed")["name"],"新的名称")
        target=self.path/"download.wav"
        self.http.download("renamed",target)
        self.assertEqual(target.read_bytes(),self.source.read_bytes())
        self.client.delete("renamed")
        self.assertFalse(self.client.has_clip("renamed"))

    def test_duplicate_key_has_no_orphan_file(self):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(self.client.import_file,self.source,key="same") for _ in range(2)]
            results=[]
            for f in futures:
                try:results.append(f.result())
                except AudioError as exc:results.append(exc)
        self.assertEqual(sum(isinstance(x,AudioError) for x in results),1)
        self.assertEqual(len(list(self.app.store.clips.glob("*.wav"))),1)
        self.assertEqual(len(self.client.clips()),1)

    def test_invalid_file_keys_truncation_and_quota(self):
        for key in ("../escape","a/b","a\\b",".",".."):
            with self.assertRaises(AudioError):self.client.import_file(self.source,key=key)
        broken=self.path/"broken.wav"
        broken.write_bytes(self.source.read_bytes()[:-100])
        with self.assertRaises(AudioError) as caught:self.client.import_file(broken,key="bad")
        self.assertEqual(caught.exception.code,"truncated_audio")
        self.config.max_library_bytes=10
        with self.assertRaises(AudioError) as caught:self.client.import_file(self.source,key="quota")
        self.assertEqual(caught.exception.code,"quota_exceeded")

    def test_mp3_conversion_uses_real_ffmpeg(self):
        mp3=self.path/"encoded.mp3"
        run_process([self.config.ffmpeg,"-v","error","-i",str(self.source),"-c:a","libmp3lame",str(mp3)],10)
        clip=self.client.import_file(mp3,key="mp3")
        with wave.open(str(self.app.library.path(clip)),"rb") as handle:
            self.assertEqual((handle.getnchannels(),handle.getsampwidth(),handle.getframerate()),(1,2,48000))
        self.assertGreater(clip["duration"],.2)

    def test_real_chinese_tts_and_cache(self):
        if not self.client.health()["tts"]:self.skipTest("eSpeak NG unavailable")
        first=self.client.say("你好，小鸭子的任务已经完成。",play=False)
        second=self.client.say("你好，小鸭子的任务已经完成。",play=False)
        self.assertEqual(first["clip"]["id"],second["clip"]["id"])
        self.assertGreater(first["clip"]["duration"],1)
        with wave.open(str(self.app.library.path(first["clip"])),"rb") as handle:
            values=array.array("h",handle.readframes(handle.getnframes()))
        self.assertGreater(max(values),1000)

    def test_idempotency_and_conflict(self):
        self.clip(seconds=.5)
        one=self.client.play("test",request_id="one",task_id="a")
        two=self.client.play("test",request_id="one",task_id="a")
        self.assertEqual(one.id,two.id)
        with self.assertRaises(AudioError) as caught:self.client.play("test",request_id="one",task_id="b")
        self.assertEqual(caught.exception.code,"idempotency_conflict")
        self.assertEqual(one.wait(3)["state"],"finished")

    def test_pause_seek_resume_and_repeat(self):
        self.clip(seconds=.7)
        job=self.client.play("test",repeat=2)
        eventually(lambda:job.status()["state"]=="playing")
        job.pause()
        before=job.status()["position"]
        time.sleep(.12)
        self.assertLess(abs(job.status()["position"]-before),.03)
        job.seek(.4)
        job.resume()
        result=job.wait(4)
        self.assertEqual((result["state"],result["iteration"]),("finished",2))

    def test_task_cancellation_isolation_and_delete_busy(self):
        self.clip("long",seconds=2)
        self.clip("short",seconds=.1)
        first=self.client.play("long",task_id="A")
        second=self.client.play("short",task_id="B")
        with self.assertRaises(AudioError) as caught:self.client.delete("long")
        self.assertEqual(caught.exception.code,"clip_in_use")
        self.client.stop_task("A")
        self.assertEqual(first.wait(3)["state"],"cancelled")
        self.assertEqual(second.wait(3)["state"],"finished")
        self.client.delete("long")

    def test_priority_preemption_and_remote_restriction(self):
        self.clip("long",seconds=2)
        self.clip("alert",seconds=.1)
        first=self.client.play("long",task_id="A")
        eventually(lambda:first.status()["state"]=="playing")
        with self.assertRaises(AudioError) as caught:self.http.play("alert",priority=100)
        self.assertEqual(caught.exception.code,"system_priority_denied")
        alert=self.client.play("alert",priority=100,task_id="system")
        self.assertEqual(alert.wait(3)["state"],"finished")
        self.assertEqual(first.wait(3)["error_code"],"preempted")

    def test_queue_expiration_and_capacity(self):
        self.clip("long",seconds=1.2)
        self.clip("short",seconds=.1)
        first=self.client.play("long")
        eventually(lambda:first.status()["state"]=="playing")
        queued=self.client.play("short",ttl=1)
        first.wait(3)
        self.assertEqual(queued.wait(2,False)["state"],"expired")
        self.config.max_queue=1
        running=self.client.play("long")
        eventually(lambda:running.status()["state"]=="playing")
        self.client.play("short")
        with self.assertRaises(AudioError) as caught:self.client.play("short")
        self.assertEqual(caught.exception.code,"queue_full")

    def test_disable_stop_muting_and_settings_persistence(self):
        self.clip(seconds=1)
        job=self.client.play("test",repeat=0)
        self.client.settings(volume=27,muted=True,enabled=False)
        self.assertEqual(job.wait(3)["state"],"cancelled")
        with self.assertRaises(AudioError):self.client.play("test")
        self.app.close()
        self.app=Application(self.config).start()
        self.assertEqual(self.client.settings()["volume"],27)
        self.assertTrue(self.client.settings()["muted"])
        self.assertFalse(self.client.settings()["enabled"])
        self.assertTrue(self.client.has_clip("test"))

    def test_stop_paused_and_reject_invalid_options(self):
        self.clip(seconds=1)
        job=self.client.play("test")
        eventually(lambda:job.status()["state"]=="playing")
        job.pause();job.stop()
        self.assertEqual(job.wait(2)["state"],"cancelled")
        for values in ({"repeat":-1},{"repeat":True},{"gain":2},{"ttl":0},{"priority":101},{"policy":"unknown"}):
            with self.assertRaises(AudioError):self.client.play("test",**values)

    def test_event_deduplication_and_simulation_backend(self):
        log=LogAudioClient(self.path/"task-events.jsonl")
        task=TaskAudio(log,task_id="demo")
        first=task.emit("target_found")
        self.assertIs(task.emit("target_found"),first)
        self.assertEqual(first.result(2).status()["backend"],"log")
        task.emit("succeeded").result(2)
        task.close()
        events=[json.loads(x) for x in log.path.read_text().splitlines()]
        self.assertEqual(len(events),2)
        self.assertTrue(all(not x["audible_verified"] for x in events))

    def test_task_api_nonblocking_and_failure_visible(self):
        class SlowClient:
            def play(self,*args,**kwargs):
                time.sleep(.15)
                raise AudioError("device unavailable","test_offline",503)
        task=TaskAudio(SlowClient(),task_id="slow")
        started=time.monotonic()
        future=task.emit("started")
        self.assertLess(time.monotonic()-started,.05)
        with self.assertRaises(AudioError):future.result(2)
        self.assertTrue(task.errors)
        task.close()

    def test_mouth_envelope_callback_and_final_close(self):
        clip=self.clip(seconds=.3)
        job=self.client.play("test")
        values=[]
        follower=MouthFollower(job,clip["metadata"]["envelope"],values.append,interval=.02).start()
        job.wait(3)
        follower.close()
        self.assertIsNone(follower.error)
        self.assertTrue(any(x>0 for x in values))
        self.assertEqual(values[-1],0)
        self.assertTrue(all(0<=x<=.65 for x in values))

    def test_local_recording_rejects_simulation_hardware(self):
        with self.assertRaises(AudioError) as caught:self.client.request("POST","/api/recording/start",{})
        self.assertEqual(caught.exception.code,"hardware_unavailable")

    def test_native_player_exit_and_output_lock(self):
        self.clip(seconds=.2)
        fake=self.path/"fake_player.py"
        captured=self.path/"captured.pcm"
        fake.write_text(f"import sys\nfrom pathlib import Path\nPath({str(captured)!r}).write_bytes(sys.stdin.buffer.read())\n")
        cfg=replace(self.config,backend="alsa",player=str(fake))
        engine=PlaybackEngine(cfg,self.app.store,self.app.library)
        try:
            job=engine.submit("test",task_id="native")
            eventually(lambda:engine.status(job["id"])["state"] in ("finished","failed"))
            self.assertEqual(engine.status(job["id"])["state"],"finished")
            self.assertEqual(captured.stat().st_size,int(.2*48000*2))
            fake.write_text("import sys\nsys.stderr.write('deliberate hardware error')\nsys.exit(23)\n")
            bad=engine.submit("test",task_id="bad")
            eventually(lambda:engine.status(bad["id"])["state"]=="failed")
            self.assertIn(engine.status(bad["id"])["error_code"],("player_failed","playback_error"))
        finally:engine.close()

    def test_recording_pipeline_with_subprocess_fixture(self):
        fake=self.path/"fake_arecord"
        fake.write_text("#!/usr/bin/python3\nimport sys,wave,struct,signal,time\np=sys.argv[-1]\nw=wave.open(p,'wb');w.setparams((2,2,48000,0,'NONE','not compressed'))\nw.writeframes(struct.pack('<hh',2000,2000)*4800)\nw.close()\n")
        fake.chmod(0o755)
        cfg=replace(self.config,backend="alsa",capture_program=str(fake))
        record=Recorder(cfg,self.app.store,self.app.library,self.app.engine)
        try:
            record.start(name="fixture recording",seconds=1)
            result=eventually(lambda:record.status() if record.status()["state"] in ("idle","failed") else None)
            self.assertEqual(result["state"],"idle",result)
            self.assertEqual(self.app.store.get_clip(result["clip_id"])["source"],"recording")
        finally:record.close()

    def test_recovery_marks_incomplete_jobs_failed(self):
        self.clip(seconds=.1)
        job=self.client.play("test")
        completed=job.wait(2)
        self.app.close()
        # Persist the state a process killed mid-play would have left behind.
        store=Store(self.path)
        completed["state"]="playing"
        store.save_job(completed)
        store.close()
        self.app=Application(self.config).start()
        recovered=self.client.request("GET","/api/jobs/"+job.id)
        self.assertEqual(recovered["state"],"failed")
        self.assertEqual(recovered["error_code"],"service_restarted")
        self.assertIsNone(self.app.engine.current)

    def test_shared_pcm_lock_wait_and_cancel(self):
        self.clip(seconds=.1)
        with open(self.config.lock_file, "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cfg = replace(self.config, backend="alsa", player="/missing-player")
            engine = PlaybackEngine(cfg, self.app.store, self.app.library)
            try:
                job = engine.submit("test")
                eventually(lambda:engine.status(job["id"])["state"]=="waiting")
                engine.stop(job["id"])
                eventually(lambda:engine.status(job["id"])["state"]=="cancelled")
                expired = engine.submit("test", ttl=1)
                eventually(lambda:engine.status(expired["id"])["state"]=="expired")
                self.assertEqual(engine.status(expired["id"])["error_code"], "queue_expired")
            finally: engine.close()

    def test_board_wrapper_receives_inherited_lock_and_warmup(self):
        self.clip(seconds=.1)
        captured = self.path / "board.pcm"
        player = self.path / "board.py"
        player.write_text("import sys,fcntl\nfrom pathlib import Path\nfd=int(sys.argv[sys.argv.index('--lock-fd')+1])\nfcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nPath(" + repr(str(captured)) + ").write_bytes(sys.stdin.buffer.read())\n")
        cfg = replace(self.config, backend="board", player=str(player), warmup_ms=100)
        engine = PlaybackEngine(cfg, self.app.store, self.app.library)
        try:
            job = engine.submit("test")
            result = eventually(lambda:engine.status(job["id"]) if engine.status(job["id"])["state"] in ("finished","failed") else None)
            self.assertEqual(result["state"], "finished", result)
            data = captured.read_bytes()
            self.assertEqual(len(data), 19200)
            self.assertEqual(data[:9600], bytes(9600))
            self.assertTrue(any(data[9600:]))
        finally: engine.close()

    def test_user_interrupt_cannot_preempt_system_prompt(self):
        self.clip(seconds=.5)
        system = self.client.play("test", priority=100)
        eventually(lambda:system.status()["state"] == "playing")
        other = self.http.play("test", policy="interrupt", priority=10)
        self.assertEqual(system.wait(2)["state"], "finished")
        self.assertEqual(other.wait(2)["state"], "finished")

    def test_recording_excludes_playback_and_yields_to_system(self):
        self.clip(seconds=.15)
        fake = self.path / "slow_arecord"
        fake.write_text("#!/usr/bin/python3\nimport sys,wave,struct,time,signal\nw=wave.open(sys.argv[-1],'wb');w.setparams((2,2,48000,0,'NONE','none'));w.writeframes(struct.pack('<hh',2000,2000)*4800);w.close()\nsignal.signal(signal.SIGINT,lambda *_:sys.exit(0))\ntime.sleep(10)\n")
        fake.chmod(0o755)
        record = Recorder(replace(self.config, backend="alsa", capture_program=str(fake)), self.app.store, self.app.library, self.app.engine)
        try:
            record.start(seconds=10)
            time.sleep(.1)
            with self.assertRaises(AudioError) as caught: self.client.play("test")
            self.assertEqual(caught.exception.code, "recording_busy")
            system = self.client.play("test", priority=100)
            self.assertEqual(system.wait(3)["state"], "finished")
            self.assertEqual(eventually(lambda:record.status() if record.status()["state"] == "idle" else None)["error"], None)
            self.assertIsNone(self.app.engine.capture_interrupt)
        finally: record.close()

    def test_factory_prompt_bridge_and_offline_fallback(self):
        from duckaudio.prompt_runner import attach
        self.clip(seconds=2)
        task = self.client.play("test")
        tone(self.path / "low_battery.wav", seconds=.15)
        class Factory:
            def launch(self, kind, now): self.fallback = True
            def maximize_volume(self, now): self.maximized = True
            def sync_timing_marker(self, now): self.marked = True
            def warn(self, *args): pass
        module = types.SimpleNamespace(PromptService=Factory, PLAYER_TIMEOUT=5)
        attach(module, self.client.endpoint)
        factory = Factory(); factory.sound_dir = self.path
        factory.launch("low_battery", time.monotonic())
        self.assertEqual(factory.child.wait(timeout=5), 0)
        self.assertTrue(factory.marked)
        self.assertEqual(task.wait(2)["error_code"], "preempted")
        self.assertTrue(any(j["task_id"] == "system.low_battery" and j["state"] == "finished" for j in self.app.engine.list_jobs()))
        class OfflineFactory:
            def launch(self, kind, now): self.fallback = True
        offline_module = types.SimpleNamespace(PromptService=OfflineFactory, PLAYER_TIMEOUT=5)
        attach(offline_module, "unix:" + str(self.path / "missing.sock"))
        fallback = OfflineFactory()
        fallback.launch("startup", time.monotonic())
        self.assertTrue(fallback.fallback)

    def test_system_volume_override_restores_latest_settings(self):
        self.clip(seconds=.5)
        mixer_log = self.path / "mixer.jsonl"
        mixer = self.path / "fake_amixer"
        # Write a real executable fixture; no host mixer is accessed.
        mixer.write_text("#!/usr/bin/python3\nimport json,sys\nwith open(" + repr(str(mixer_log)) + ", 'a') as f:\n    print(json.dumps(sys.argv[1:]), file=f)\n")
        mixer.chmod(0o755)
        player = self.path / "reader.py"
        player.write_text("import sys\nsys.stdin.buffer.read()\n")
        engine = PlaybackEngine(replace(self.config, backend="alsa", player=str(player), mixer_program=str(mixer)),self.app.store,self.app.library)
        try:
            job = engine.submit("test", priority=100)
            eventually(lambda:engine.status(job["id"])["state"]=="playing")
            self.assertTrue(engine.settings()["system_prompt_override"])
            engine.set_settings(volume=23, muted=True)
            self.assertEqual(len(mixer_log.read_text().splitlines()),2)
            eventually(lambda:engine.status(job["id"])["state"]=="finished")
            calls = [json.loads(x) for x in mixer_log.read_text().splitlines()]
            self.assertEqual(calls[0][-1], "100%,100%")
            self.assertEqual(calls[-1][-1], "off,off")
            self.assertFalse(engine.settings()["system_prompt_override"])
            engine.set_settings(muted=False)
            self.assertEqual(json.loads(mixer_log.read_text().splitlines()[-2])[-1],"23%,23%")
        finally: engine.close()

    def test_shutdown_disconnects_incomplete_upload(self):
        connection = socket.create_connection(self.app.http.server_address)
        header = ("POST /api/clips?key=partial HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer " + self.app.token + "\r\nContent-Length: 1000\r\n\r\nabc").encode()
        connection.sendall(header)
        time.sleep(.05)
        started=time.monotonic()
        self.app.close()
        connection.close()
        self.assertLess(time.monotonic()-started,2)
        self.assertEqual(list((self.path/"tmp").glob("*.upload")), [])

    def test_restart_cleans_only_owned_orphan_media(self):
        self.app.close()
        temp=self.path/"tmp"/("a"*32+".wav");temp.write_bytes(b"partial")
        orphan=self.path/"clips"/("b"*32+".wav");orphan.write_bytes(b"orphan")
        foreign=self.path/"tmp"/"user-notes.txt";foreign.write_text("keep")
        self.app=Application(self.config).start()
        self.assertFalse(temp.exists())
        self.assertFalse(orphan.exists())
        self.assertTrue(foreign.exists())

    def test_supported_compressed_import_formats(self):
        for extension, codec in (("m4a", "aac"), ("ogg", "libvorbis"), ("flac", "flac")):
            with self.subTest(format=extension):
                converted = self.path / ("source." + extension)
                run_process([self.config.ffmpeg,"-v","error","-y","-i",str(self.source),"-c:a",codec,str(converted)],10)
                clip=self.client.import_file(converted,key="format."+extension)
                self.assertEqual(clip["metadata"]["format"],"PCM16/48000/mono")
                self.assertGreater(clip["duration"],.2)

    def test_cli_control_resource_and_diagnostics(self):
        self.clip(seconds=1)
        def cli(*arguments):
            result=subprocess.run([sys.executable,"-m","duckaudio","--endpoint",self.client.endpoint,*arguments],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)
        self.assertEqual(cli("doctor")["backend"],"null")
        job=cli("play","test","--repeat","0","--request-id","cli-id")
        eventually(lambda:self.app.engine.status(job["id"])["state"]=="playing")
        self.assertEqual(cli("pause",job["id"])["state"],"paused")
        cli("seek",job["id"],".4")
        cli("resume",job["id"])
        cli("stop","--job-id",job["id"])
        eventually(lambda:cli("status",job["id"])["state"]=="cancelled")
        self.assertEqual(cli("rename","test","命令行改名")["name"],"命令行改名")
        cli("download","test",str(self.path/"cli.wav"))
        self.assertTrue((self.path/"cli.wav").is_file())
        cli("delete","test")
        self.assertFalse(self.client.has_clip("test"))

    def test_cleanup_kills_child_after_wrapper_exits(self):
        childfile=self.path/"child.pid"
        code="import subprocess,sys\nsubprocess.Popen([sys.executable,'-c',"+repr("import signal,time,os\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath("+repr(str(childfile))+").write_text(str(os.getpid()))\ntime.sleep(30)")+ "])\n"
        process=subprocess.Popen([sys.executable,"-c",code],start_new_session=True)
        try:
            eventually(childfile.exists)
            process.wait(timeout=2)
            child=int(childfile.read_text())
            terminate_process(process,timeout=.1)
            def stopped():
                try:return Path(f"/proc/{child}/stat").read_text().split()[2]=="Z"
                except FileNotFoundError:return True
            eventually(stopped)
        finally:terminate_process(process,timeout=.1)


if __name__ == "__main__":unittest.main()
