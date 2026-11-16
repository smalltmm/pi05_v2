"""Mixer-only observation/action adapter for the local visual servo."""

import argparse

import numpy as np

try:
    from .final_policy import FinalPolicy
    from .pi05_assists.options import parse_options
    from .pi05_assists.runtime import clip_actions
    from .pi05_assists.mixer.mixer_servo import (
        MixerServo, add_arguments, config_from_args,
    )
except ImportError:
    from final_policy import FinalPolicy
    from pi05_assists.options import parse_options
    from pi05_assists.runtime import clip_actions
    from pi05_assists.mixer.mixer_servo import (
        MixerServo, add_arguments, config_from_args,
    )


def parse_mixer_options(tokens):
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    mixer_args, remaining = parser.parse_known_args(tokens)
    args = parse_options(remaining)
    vars(args).update(vars(mixer_args))
    config_from_args(args)
    return args


class MixerFinalPolicy(FinalPolicy):
    """Keep hardware execution in the official adapter, at 10 Hz."""

    def __init__(self, client, args):
        super().__init__(client, args)
        self.mixer = MixerServo(config_from_args(args))
        self.right_hold_command = None

    def _step_mixer(self, state, encoded_obs):
        return self.mixer.step(
            state, encoded_obs["observation/left_wrist_image"],
            frame_id=self.runtime.steps, now=self.runtime.clock(),
        )

    def next_action(self, encoded_obs, prompt=None, *, feedback=None):
        runtime = self.runtime
        if runtime.pending is not None:
            raise RuntimeError("Call commit_action before requesting another action")
        if runtime.steps >= 1000:
            raise StopIteration("RoboSyn episode reached 1000 executed frames")
        state = encoded_obs["observation/state"]
        if self.mixer.owns_motion and not self.args.mixer_servo_observe_only:
            target = self._step_mixer(state, encoded_obs)
            if target is not None:
                target = self._apply_right_hold(target)
                runtime.pending = dict(
                    current=state.copy(), target=target.copy(),
                    mixer_servo_only=True, record={"mixer_servo": self.mixer.record()},
                )
                return target.copy()
            # A released visual plan must not resume stale model actions.
            runtime.row = len(runtime.control)

        target = super().next_action(encoded_obs, prompt, feedback=feedback)
        runtime = self.runtime
        right_step = runtime.pending.get("right")
        if right_step is not None:
            self.mixer.observe_right_grip(right_step.record)
            self.right_hold_command = (
                right_step.record["hold_command"] if right_step.force_close else None
            )
        servo_target = self._step_mixer(state, encoded_obs)
        if not self.args.mixer_servo_observe_only:
            if servo_target is not None:
                target = servo_target
            elif self.mixer.release_sequence_enabled and self.mixer.phase == "model":
                if self.mixer.wait_left_target is not None:
                    target = target.copy()
                    target[:7] = self.mixer.wait_left_target
            target = self._apply_right_hold(target)
        runtime.pending["target"] = target.copy()
        runtime.pending.setdefault("record", {})["mixer_servo"] = self.mixer.record()
        return target.copy()

    def _apply_right_hold(self, target):
        target = np.asarray(target, dtype=np.float32).copy()
        if self.right_hold_command is not None:
            target[13] = self.right_hold_command
        return clip_actions(target[None, :])[0]

    def commit_action(self):
        runtime = self.runtime
        if runtime.pending is not None and runtime.pending.get("mixer_servo_only"):
            runtime.previous = runtime.pending["target"].copy()
            runtime.last_record = dict(
                runtime.pending["record"], frame=runtime.steps, cycle=runtime.cycle,
            )
            runtime.steps += 1
            runtime.pending = None
            return
        super().commit_action()

    def reset(self):
        super().reset()
        self.mixer.reset()
        self.right_hold_command = None
