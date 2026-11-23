"""Create display-only meshes from the official render rig and build two MJCF scenes."""
import json
import fcntl
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import fast_simplification
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / 'vendor/wonderduck/simulator/robot/wonderduck'
CACHE = ROOT / 'runs/ducklab-assets'
TABLE_Z = .179
BALL_RADIUS = .05
GOAL_X = .35
GOAL_HALF_WIDTH = .115


def display_meshes():
    directory = CACHE / 'display'
    directory.mkdir(parents=True, exist_ok=True)
    data = (ROBOT / 'robot_visual.glb').read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    gltf = json.loads(data[20:20 + size]); base = 28 + size

    def accessor(i):
        desc = gltf['accessors'][i]; view = gltf['bufferViews'][desc['bufferView']]
        dtype = np.dtype({5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[desc['componentType']])
        columns = {'VEC3': 3, 'SCALAR': 1}[desc['type']]
        return np.ndarray((desc['count'], columns), dtype, buffer=data,
                          offset=base + view.get('byteOffset', 0) + desc.get('byteOffset', 0),
                          strides=(view.get('byteStride', columns * dtype.itemsize), dtype.itemsize))

    for mesh in gltf['meshes']:
        path = directory / (mesh['name'] + '.obj')
        if path.is_file():
            continue
        prim = mesh['primitives'][0]
        vertices = accessor(prim['attributes']['POSITION']).copy()
        faces = accessor(prim['indices']).reshape(-1, 3).astype(np.int32)
        # STL-style coincident vertices must be shared for edge-collapse simplification.
        points, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces].astype(np.int32)
        if len(faces) > 750:
            points, faces = fast_simplification.simplify(points, faces, target_count=750)
        with path.with_suffix('.tmp').open('w') as out:
            np.savetxt(out, points, fmt='v %.8g %.8g %.8g')
            np.savetxt(out, faces + 1, fmt='f %d %d %d')
        path.with_suffix('.tmp').replace(path)
    return directory


def node(parent, tag, **attrs):
    def txt(v):
        return ' '.join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else str(v)
    return ET.SubElement(parent, tag, {k: txt(v) for k, v in attrs.items()})


def scene_xml(bench):
    display = display_meshes()
    xml = ET.parse(ROBOT / 'robot.xml').getroot()
    world, assets = xml.find('worldbody'), xml.find('asset')
    xml.find('compiler').attrib.pop('meshdir')
    for mesh in assets.findall('mesh'):
        mesh.set('file', str(ROBOT / 'meshes' / mesh.get('file')))
    for geom in world.iter('geom'):
        geom.set('group', '3')
    bodies = {b.get('name'): b for b in world.iter('body')}
    for body in json.loads((ROBOT / 'kinematics.json').read_text())['bodies']:
        for geom in body['geoms']:
            name = geom['mesh']
            if assets.find(f"mesh[@name='{name}']") is None:
                node(assets, 'mesh', name=name, file=display / (name + '.obj'))
            node(bodies[body['name']], 'geom', type='mesh', mesh=name, pos=geom['pos'],
                 quat=geom['quat'], rgba=geom['color'], contype=0, conaffinity=0, group=2, density=0)
    node(bodies['camera'], 'camera', name='head_rgb', xyaxes='0 -1 0  .60 0 .80', fovy=100)
    visual = node(xml, 'visual')
    node(visual, 'global', offwidth=720, offheight=480)
    node(visual, 'map', znear=.004, zfar=35)
    node(visual, 'headlight', ambient='.45 .45 .45', diffuse='.48 .48 .48', specular='.08 .08 .08')
    node(assets, 'texture', name='tiles', type='2d', builtin='checker', rgb1='.18 .25 .28',
         rgb2='.22 .31 .34', width=64, height=64)
    node(assets, 'texture', type='skybox', builtin='gradient', rgb1='.35 .49 .59',
         rgb2='.13 .20 .23', width=64, height=384)
    node(assets, 'material', name='floor_mat', texture='tiles', texrepeat='4 4', texuniform='true', reflectance=.04)
    node(world, 'light', pos='.3 -.5 1.6', dir='-.2 .25 -1', diffuse='.78 .78 .78', castshadow='false')
    node(world, 'geom', type='plane', size='0 0 .04', material='floor_mat', name='floor',
         conaffinity=5, friction='1 .005 .0001')
    # Inactive scene props are parked far from the duck at reset.
    if bench:
        eq = node(xml, 'equality')
        node(eq, 'weld', name='bench_support', body1='trunk_base', solref='.005 1')
        node(world, 'geom', name='table', type='box', pos=(.155, 0, TABLE_Z-.005),
             size='.091 .102 .005', rgba='.45 .35 .27 1', friction='.85 .01 .001')
        for y in [-.086, .086]:
            node(world, 'geom', type='box', pos=(.19, y, TABLE_Z/2), size=(.007,.007,TABLE_Z/2-.005), rgba='.28 .22 .19 1')
        # The pad is a visual marking, not a physical obstacle.
        node(world, 'geom', name='pad', type='cylinder', pos=(.092,.028,TABLE_Z+.0001),
             size='.015 .0002', rgba='.12 .90 .30 1', contype=0, conaffinity=0)
        for name, color in [('parcel', '.96 .05 .07 1'), ('spare', '.04 .18 .98 1')]:
            body = node(world,'body', name=name, pos=(.096,-.02,TABLE_Z+.007))
            node(body,'freejoint',name=name+'_free')
            node(body, 'inertial', pos='0 0 0', mass=.003, diaginertia=(7.2e-8,)*3)
            node(body,'geom',name=name,type='box',size='.006 .006 .006',mass=.003,
                 rgba=color, friction='.88 .01 .001', condim=4)
        tip = xml.find(".//site[@name='mouth_tip']").get('pos')
        # Ball/socket grasp surrogate, activated only after measured contact near the tip.
        for name in ('parcel', 'spare'):
            node(eq, 'connect', name='hold_'+name, body1='head_roll', body2=name,
                 anchor=tip, active='false', solref='.008 1')
        node(bodies['head_roll'],'geom',name='tip_pad',type='sphere',pos=tip,size=.003,
             rgba='.9 .47 .12 1',mass=0, friction='.9 .01 .001')
    elif bench is False:
        ball = node(world,'body',name='ball',pos='2 1 .03'); node(ball,'freejoint',name='ball_free')
        node(ball, 'inertial', pos='0 0 0', mass=.03, diaginertia=(.00003,)*3)
        node(ball,'geom',name='ball',type='sphere', size=BALL_RADIUS, mass=.03, condim=6,
             rgba='.96 .39 .14 1', friction='.4 .01 .003', solref='.03 .4', conaffinity=5)
        for i in range(6):
            node(world,'geom',name=f'mark{i}', type='cylinder',pos=(3+i,1,.001),
                 size='.025 .0008',rgba='.13 .71 .65 .9',contype=0,conaffinity=0)
        for i,y in enumerate((-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH)):
            node(world,'geom',name=f'post{i}',type='capsule',size='.006 .055',pos=(GOAL_X,y,.055),rgba='.9 .92 .83 1')
        node(world, 'geom', name='goal_line', type='box', pos=(GOAL_X,0,.0005),
             size=(.002,GOAL_HALF_WIDTH,.0004), rgba='.70 .86 .77 1', contype=0, conaffinity=0)
    return xml


def build_scene(bench):
    return mujoco.MjModel.from_xml_string(ET.tostring(scene_xml(bench), encoding='unicode'))


def load_scene(bench):
    # Invalidate the native cache after scene-code, vendor model or MuJoCo version changes.
    path = CACHE / f'{"bench" if bench else "pitch"}-{mujoco.__version__}.mjb'
    sources = (Path(__file__), ROBOT/'robot.xml', ROBOT/'kinematics.json', ROBOT/'robot_visual.glb')
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and path.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
            return mujoco.MjModel.from_binary_path(str(path))
        model = build_scene(bench)
        temp = path.with_suffix('.partial'); mujoco.mj_saveModel(model, str(temp), None); temp.replace(path)
        return model
