"""Standard-library high-level actor replay. No hardware or networking imports."""
import math

SCHEMA = "wonderduck.retrieval.shadow.v1"
OBS_SIZE = 29
ACTION_SIZE = 6
STAGES = {"pick": 0, "delivery": 1}


def validate_observation(stage, obs):
    if stage not in STAGES:
        raise ValueError("unknown retrieval stage")
    if not isinstance(obs, list) or len(obs) != OBS_SIZE:
        raise ValueError("expected 29 observation values")
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1.000001 for x in obs):
        raise ValueError("observation must be finite and normalized")
    if obs[:2] != ([1.0, 0.0] if stage == "pick" else [0.0, 1.0]):
        raise ValueError("stage and observation flags disagree")


def infer(model, stage, obs):
    validate_observation(stage, obs)
    if model.get("schema") != SCHEMA or model.get("motion_permitted") is not False:
        raise ValueError("not a supported shadow actor")
    weights = model["weights"]
    def layer(values, name):
        rows, bias = weights[name + ".weight"], weights[name + ".bias"]
        if len(rows) != len(bias) or any(len(row) != len(values) for row in rows):
            raise ValueError("actor layer dimensions disagree")
        result = [math.tanh(math.fsum(a*b for a, b in zip(row, values)) + offset)
                  for row, offset in zip(rows, bias)]
        if not all(math.isfinite(v) for v in result):
            raise ValueError("non-finite actor output")
        return result
    values = layer(obs, "body.0")
    values = layer(values, "body.2")
    actions = layer(values, "actors." + str(STAGES[stage]))
    if len(actions) != ACTION_SIZE:
        raise ValueError("expected six high-level parameters")
    return actions


def verify_cases(model, cases, tolerance=2e-6):
    if not cases:
        raise ValueError("no replay cases")
    error = 0.0
    for case in cases:
        actual = infer(model, case["stage"], case["obs"])
        expected = case["actions"]
        if len(expected) != ACTION_SIZE or not all(math.isfinite(v) for v in expected):
            raise ValueError("invalid reference actions")
        error = max(error, max(abs(a-b) for a, b in zip(actual, expected)))
    return dict(code="SHADOW_REPLAY_PASS" if error <= tolerance else "SHADOW_REPLAY_FAIL",
                cases=len(cases), max_abs_error=error, tolerance=tolerance,
                hardware_tested=False, motion_permitted=False)
