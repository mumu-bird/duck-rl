# Official Wonder Duck Assets

Retrieved on 2026-11-21 from Hiwonder's public Wonder Duck browser simulator and WonderRL API. Vendor assets are kept under `vendor/`, which is excluded from Git because the public pages do not state a redistribution license.

## Downloaded Package

```text
vendor/wonderduck/
  policies/
    velocity.onnx
    stand.onnx
    sitstand.onnx
    ground_pick.onnx
    ball_kick_left.onnx
    ball_kick_right.onnx
  simulator/
    active-models.json
    robot/wonderduck/
      robot.xml
      kinematics.json
      robot_visual.glb
      visual_updates.glb
      meshes/                 19 files referenced by robot.xml
```

The raw MJCF declares 15 joint records: 14 actuated hinge joints and one floating root joint; it has 14 actuators. All 19 referenced mesh files are present. It is the model currently loaded by the official online simulator, not yet confirmed by Hiwonder as the correct trainable model for the user's exact Wonder Duck SKU.

The two GLB files are for the browser simulator's visual representation; MuJoCo's physical model uses `robot.xml` and the referenced mesh files. The six ONNX files are published inference policies for walking, standing, sit/stand, ground pickup, and left/right ball kicking. They are useful simulation baselines, but are not the editable PPO source project or training checkpoints.

## Version And Compatibility Checks

- The public Wonder Duck product/forum description says the robot has 15 degrees of freedom. The current online simulator's `robot.xml` contains 14 actuated hinge joints and one floating root joint, with 14 actuators. Whether the public "15 DOF" claim counts a different mechanism or indicates a model revision remains unresolved; ask Hiwonder to confirm the exact SKU and whether an updated MJCF exists before training for hardware deployment.
- The WonderRL 0.1.32 Windows x64 installer is saved at `vendor/wonderrl/0.1.32/WonderRL-0.1.32-windows-x64.exe` (SHA-256 `7f5f0d7f73953e972b95c4373cecb8d60b3c6ec7fce5362f5a492dc3a216ed29`). It was not run. InnoExtract 1.9 reports an unsupported setup loader revision; 7-Zip 26.06 cannot open it. This does not prove the installer is corrupt, but its contents and whether it bundles the editable training project remain unverified.
- The host Python interpreter does not have MuJoCo preinstalled. For verification only, MuJoCo Python 4.0.0 and its dependencies were installed under `/tmp`; the model compiled, all 19 meshes loaded, and one simulation step completed (`nbody=23`, `njnt=15`, `nu=14`, `nmesh=19`). No system Python or user MuJoCo installation was changed.
- The online simulator's endpoint currently publishes six baseline ONNX policies. Their file URLs and returned policy metadata are recorded in `simulator/active-models.json`.
- The official online simulator extends the robot MJCF at runtime with the floor, ball, goal, and arena. Those scene additions are separate from the robot model; use the provided WonderRL training environment for matching scenes and policy contracts.
- No editable PPO training source/configuration, matching training checkpoint, or documented robot-side ONNX deployment contract has been found in the public assets downloaded here. Request those items from Hiwonder or the seller before fine-tuning or deploying a new policy.
- The official November 12 deep-dive says a MuJoCo simulation environment, PPO training project, and robot deployment process are provided. The November 18 sales post says learning materials are being opened progressively and directs owners to official customer service or the technical after-sales group for tutorials and files. Neither post provides a direct public download link to the editable training project, so it may be available to purchasers through support rather than an anonymous public download. When requesting it, ask for the matching Wonder Duck SKU's MJCF plus meshes, editable PPO project/configs, checkpoints, Linux/GPU requirements, and ONNX observation/action deployment contract.

## Sources

- Hiwonder product forum, Wonder Duck and WonderRL introduction: [official post](https://forum.hiwonder.com.cn/t/wonderduck/1560). The post describes MuJoCo/PPO training and says learning resources are being released progressively.
- Hiwonder Wonder Duck online simulator: [simulator](https://wonderrl.hiwonder.com.cn/WonderDuck/Simulator/).
- Raw physical model: `https://wonderrl.hiwonder.com.cn/WonderDuck/Simulator/robot/wonderduck/robot.xml`.
- Referenced meshes: `https://wonderrl.hiwonder.com.cn/WonderDuck/Simulator/robot/wonderduck/meshes/<filename>`.
- Kinematics and visualization files use the same `/WonderDuck/Simulator/robot/wonderduck/` directory.
- Current simulator policy metadata: `https://wonderrl.hiwonder.com.cn/api/rl/simulator/WonderDuck`.
- Policy binaries: `https://wonderrl.hiwonder.com.cn/download/WonderDuck/<filename>`; filenames and metadata are preserved in `vendor/wonderduck/simulator/active-models.json`.
- WonderStudio download page: [official WonderStudio page](https://hiwonder.com.cn/product-detail/WonderStudio.html).

## Integrity Check

From the repository root, check all downloaded Wonder Duck files against the retrieval-time manifest:

```bash
sha256sum -c docs/wonderduck-assets.sha256
```

The installer has a separate checksum in the first section above.
