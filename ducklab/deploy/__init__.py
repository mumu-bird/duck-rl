"""Deployable WonderDuck training: policies that match the on-robot policy contract.

Everything in this package is built around the contract the real robot's ``robotd``
enforces (transcribed into :mod:`ducklab.deploy.contract`):

* observation ``obs[1, 61]`` — proprioception + command block, no camera;
* action ``actions[1, 14]`` — joint offsets from the home pose, mouth excluded;
* 50 Hz policy over 5 ms (200 Hz) MuJoCo physics, action scale 1.0, pure-P servo
  firmware gain 15, no low-pass, official 3–6 step command delay;
* the mouth (bus servo 34) is *outside* the policy and driven by a separate
  rule controller, exactly as on the robot.

The physics model is the official identified ``robot.xml`` shipped in the
simulator handoff package under ``vendor/`` (git-ignored, see docs).
"""
