"""Connect envelope values to an existing task owner's mouth command."""
from duckaudio.tasks import MouthFollower


def speak_with_owned_mouth(client, clip_key, set_normalized_mouth, still_owns_mouth, task_id):
    """set_normalized_mouth receives 0..0.65; caller supplies calibrated motor conversion.

    Call from the task orchestrator. The callback should queue to its motor loop,
    or send through the already-owned motor session; it must not seize control.
    Returns handles the task must close/stop on cancellation or ownership loss.
    """
    envelope = client.request("GET", "/api/clips/" + clip_key + "/envelope")
    playback = client.play(clip_key, task_id=task_id)
    follower = MouthFollower(playback, envelope, set_normalized_mouth, allowed=still_owns_mouth).start()
    return playback, follower
