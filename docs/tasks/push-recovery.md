# Push-Resistant Walking and Recovery

## Goal

Train a policy to remain controllable after a bounded disturbance, or to return to a stable standing/walking state after a controlled loss of balance.

## Constraints

- Use only sensor observations already present in the official deployable policy contract, such as its existing posture and joint-state inputs.
- Apply disturbance forces through the official MuJoCo environment/task API; do not invent an observation channel or actuator interface.
- Define the disturbance range from the official model and a baseline sweep. Increase it gradually instead of selecting arbitrary force magnitudes.
- Record whether each episode was a keep-balance or recover-from-fall scenario so metrics remain interpretable.

## Reward structure

Reward stable posture and recovery toward the official standing/walking target. Penalize terminal uncontrolled falls, joint-limit violations, nonfinite states, and unnecessarily abrupt actions. Keep the original stand/recovery policy as the baseline.

## Acceptance

Evaluate at least 100 holdout seeds. The initial project target is 80% task success, terminal-fall rate below 5%, and no joint-limit or nonfinite episodes. Report keep-balance and recovery scenarios separately; a combined score must not hide a failure in either subset.
