"""Encode a real-time 10 fps MuJoCo replay. The inset is the actual head camera."""
import subprocess
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import mujoco
from PIL import Image, ImageDraw, ImageFont

from .bench import view_options


@lru_cache
def font(size):
    for path in ('/usr/share/fonts/Alibaba-PuHuiTi/Alibaba-PuHuiTi-Regular.otf',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Video:
    def __init__(self, path, renderer, head, label, seed, *, arena_view=None):
        self.arena_view = arena_view
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.path.with_suffix("").with_name(
            self.path.stem + "-frames"
        )
        self.temp = self.path.with_suffix('.partial.mp4')
        self.renderer, self.head = renderer, head
        self.label, self.seed = label, seed
        self.camera, self.options = mujoco.MjvCamera(), view_options()
        self.frame = None
        self.snapshots = []
        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s:v','768x416','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2',
            '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(self.temp)
        ], stdin=subprocess.PIPE)

    def _render_canvas(self, data, lines, *, kick=False):
        from .assets import GOAL_X
        if self.arena_view is not None:
            self.camera.lookat[:], self.camera.distance = self.arena_view
            self.camera.elevation, self.camera.azimuth = -36, 112
        elif self.head:
            self.camera.lookat[:] = [.067,0,.161]
            self.camera.distance, self.camera.elevation, self.camera.azimuth = .43,-25,139
        elif kick:
            self.camera.lookat[:] = [GOAL_X*.45, 0, .102]
            self.camera.distance, self.camera.elevation, self.camera.azimuth = .68,-21,114
        else:
            self.camera.lookat[:] = [*data.qpos[:2], .128]
            self.camera.distance, self.camera.elevation, self.camera.azimuth = .52,-17,130
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
        canvas = Image.new('RGB',(768,416),'#13232b')
        draw = ImageDraw.Draw(canvas)
        draw.text((14,5),self.label,font=font(23),fill='#f4f6f4')
        draw.text((547,12),'Wonder Duck  /  MuJoCo',font=font(16),fill='#9ed9cf')
        canvas.paste(Image.fromarray(self.renderer.render()),(8,43))
        if self.head:
            self.head.update_scene(data,camera='head_rgb',scene_option=self.options)
            head_image = Image.fromarray(self.head.render())
            if self.arena_view is not None:
                head_image.thumbnail((200,150), Image.Resampling.LANCZOS)
            canvas.paste(head_image,(556,66))
            draw.text((556,42),'头部 RGB',font=font(15),fill='#b6e4dd')
        top = 224 if self.head else 70
        for index, line in enumerate(lines):
            draw.text((556,top+26*index),line,font=font(15),fill='#eaf2ee')
        draw.text((15,381),f'种子 {self.seed}     仿真时间 {data.time:.2f} s',font=font(17),fill='#deebe8')
        return canvas

    def capture(self, data, lines, *, kick=False):
        canvas = self._render_canvas(data, lines, kick=kick)
        self.process.stdin.write(canvas.tobytes())
        self.frame = canvas

    def snapshot(self, data, tag, lines=(), *, kick=False):
        """Persist a labeled PNG without changing the encoded video timeline."""
        canvas = self._render_canvas(data, lines, kick=kick)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(tag)
        ).strip("._") or "frame"
        path = self.snapshot_dir / f"{len(self.snapshots):02d}-{safe_tag}.png"
        canvas.save(path)
        self.snapshots.append(str(path))
        self.frame = canvas
        return path

    def finish(self, verdict=None):
        try:
            if verdict and self.frame is not None:
                draw = ImageDraw.Draw(self.frame)
                draw.text((460,378),verdict,font=font(19),fill='#ffe39a')
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                final_path = self.snapshot_dir / f"{len(self.snapshots):02d}-final.png"
                self.frame.save(final_path)
                self.snapshots.append(str(final_path))
                for _ in range(8):
                    self.process.stdin.write(self.frame.tobytes())
        finally:
            self.process.stdin.close()
            code = self.process.wait(timeout=45)
        if code:
            raise RuntimeError(f'ffmpeg exited with {code}')
        if verdict is not None:
            self.temp.replace(self.path)
