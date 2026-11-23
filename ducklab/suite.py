"""Checkpointed batch runner for calibrated Wonder Duck MuJoCo episodes."""
import csv
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from .arena import Arena

ROOT = Path(__file__).resolve().parents[1]
CORE = ('stand','walk_forward','walk_backward','sidestep_left','sidestep_right',
        'turn_left','turn_right','sit_stand','ground_pick_rollout','kick_left','kick_right',
        'push_recovery')
EXPLORATORY = ('rgb_locate','table_fixture','table_contact','table_push')
TASKS = CORE+EXPLORATORY+('penalty_baseline','penalty_approach')
SUPPORTED_TASKS = TASKS + ('table_push_feedback', 'table_push_initial_image')
FIELDS = ('task','episode','seed','success','task_criterion','fell','nonfinite',
          'joint_limit_violations','max_exceed_deg','sim_s','setup','metrics','video')


class BatchAlreadyRunning(RuntimeError):
    """Another process owns the given results directory."""


def aggregate(output):
    output = Path(output)
    groups = {}
    with (output/'episodes.csv').open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            item = groups.setdefault(row['task'],{'episodes':0,'successes':0,'falls':0,
                'joint_limit_episodes':0,'joint_limit_ticks':0,'nonfinite':0,'scored':0,
                'task_criterion_passes':0,'max_exceed_deg':0.})
            item['episodes'] += 1
            item['falls'] += int(row['fell'])
            item['joint_limit_ticks'] += int(row['joint_limit_violations'])
            item['joint_limit_episodes'] += int(int(row['joint_limit_violations'])>0)
            item['nonfinite'] += int(row['nonfinite'])
            item['max_exceed_deg'] = max(item['max_exceed_deg'],float(row['max_exceed_deg']))
            item['task_criterion_passes'] += int(row['task_criterion'])
            if row['success']:
                item['successes'] += int(row['success']);item['scored'] += 1
    for row in groups.values():
        row['success_rate'] = round(row['successes']/row['scored'],4) if row['scored'] else None
    videos = sorted(str(p.resolve()) for p in (output/'videos').glob('*.mp4') if '.partial.' not in p.name)
    report = {'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'tasks':groups,
              'videos':videos,'episodes_csv':str((output/'episodes.csv').resolve()),
              'scope':'Official ONNX baseline rollouts; scripted interaction prototypes are separately named. No PPO training or hardware deployment.'}
    save_json(output/'summary.json',report)
    return report


def save_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def run_suite(output, *, episodes=None, seed, duration_hours=0, video_every=0,
              task_filter=None, exploratory=False, pace=0, deadline=None, video_cap=64,
              audio_endpoint=None, audio_token_file=None):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    selected = list(task_filter or (CORE+EXPLORATORY if exploratory else CORE))
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(SUPPORTED_TASKS):
        raise ValueError('unknown or empty task list')
    with (output/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BatchAlreadyRunning(f'Batch is already active: {output}') from None
        return _run_locked(output,selected,seed,episodes,duration_hours,video_every,pace,deadline,video_cap,audio_endpoint,audio_token_file)


def _run_locked(output,selected,seed,episodes,duration_hours,video_every,pace,deadline,video_cap,audio_endpoint=None,audio_token_file=None):
    from duckaudio import AudioClient
    from duckaudio.tasks import LogAudioClient, TaskAudio
    import uuid
    audio_client = AudioClient(audio_endpoint, token_file=audio_token_file) if audio_endpoint else LogAudioClient(output / "audio-events.jsonl")
    audio_run = uuid.uuid4().hex[:8]
    end = time.monotonic()+duration_hours*3600 if duration_hours else float('inf')
    approach = json.loads((ROOT/'configs/penalty-approach.json').read_text()) if 'penalty_approach' in selected else None
    identity = {'version':2,'tasks':selected,'start_seed':seed,'scheme':'same seed per task in a round',
                'approach':approach}
    csv_path = output/'episodes.csv'
    manifest = output/'batch.json'
    if manifest.is_file():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Task list, seed or configuration changed. Use a new output directory.')
    elif csv_path.exists():
        raise ValueError('Existing study has no matching batch manifest. Use a new output directory.')
    else:
        save_json(manifest,identity)
    offset = 0
    if csv_path.exists():
        with csv_path.open(newline='') as saved:
            reader = csv.DictReader(saved)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError('Incompatible or incomplete episode log')
            for offset,row in enumerate(reader,1):
                i = offset-1
                if row['video'] is None or row['task'] != selected[i%len(selected)] or int(row['seed'])!=seed+i//len(selected):
                    raise ValueError(f'Incomplete or unexpected episode row {offset}')
    count, recorded = 0,len([p for p in (output/'videos').glob('*.mp4') if '.partial.' not in p.name])
    reason = 'complete'
    def status(state):
        save_json(output/'status.json',{'state':state,'episodes':offset+count,'videos':recorded,
                  'pid':os.getpid(),'updated':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'deadline_unix':deadline})
    arena = Arena()
    try:
        fresh = not csv_path.exists()
        with csv_path.open('a',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=FIELDS)
            if fresh:
                writer.writeheader();stream.flush();os.fsync(stream.fileno())
            status('running')
            try:
                while (episodes is None or count<episodes) and time.monotonic()<end:
                    if deadline is not None and time.time() >= deadline:
                        reason = 'night_finished'
                        break
                    if (output/'STOP').is_file():
                        reason = 'stopped_by_file'
                        break
                    if count%25==0 and shutil.disk_usage(output).free<2*1024**3:
                        reason = 'low_disk_space'
                        break
                    index = offset+count
                    task = selected[index%len(selected)]
                    episode_seed = seed+index//len(selected)
                    capture = video_every>0 and recorded<video_cap and (index<len(selected) or index%video_every==0)
                    video = output/'videos'/f'{task}-{episode_seed}.mp4' if capture else None
                    audio = TaskAudio(audio_client, task_id=f"suite:{audio_run}:{task}:{episode_seed}")
                    audio.emit("started")
                    try:
                        result = arena.run(task,episode_seed,video_path=video,approach=approach)
                        audio.emit("succeeded" if result.get("success") else "failed")
                    except BaseException:
                        audio.close(cancel=True)
                        raise
                    else:
                        audio.close()
                    row = {key:result[key] for key in FIELDS if key in result}
                    row.update(setup=json.dumps(result['setup'],ensure_ascii=False,separators=(',',':')),
                               metrics=json.dumps(result['metrics'],ensure_ascii=False,separators=(',',':')),
                               video=str(video.relative_to(output)) if video else '')
                    writer.writerow(row);stream.flush();os.fsync(stream.fileno())
                    recorded += int(capture)
                    count += 1
                    status('running')
                    if count==1 or (index+1)%250==0:
                        aggregate(output)
                    if pace and (episodes is None or count<episodes):
                        budget = min(pace, end - time.monotonic())
                        if deadline is not None:
                            budget = min(budget, deadline - time.time())
                        time.sleep(max(0,budget))
            except KeyboardInterrupt:
                reason = 'interrupted'
    except BaseException:
        status('error')
        raise
    finally:
        arena.close()
    report = aggregate(output)
    status(reason)
    return report
