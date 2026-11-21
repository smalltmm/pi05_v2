"""Legacy mixer option parsing and model-only policy compatibility."""

import argparse


try:
    from .final_policy import FinalPolicy
    from .pi05_assists.options import parse_options
    from .pi05_assists.mixer.mixer_servo import (
        add_arguments,
    )
except ImportError:
    from final_policy import FinalPolicy
    from pi05_assists.options import parse_options
    from pi05_assists.mixer.mixer_servo import (
        add_arguments,
    )


def parse_mixer_options(tokens):
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    mixer_args, remaining = parser.parse_known_args(tokens)
    args = parse_options(remaining)
    vars(args).update(vars(mixer_args))
    # Retain legacy option syntax without loading unused servo calibration.
    return args


class MixerFinalPolicy(FinalPolicy):
    """Compatibility name for model-only mixer chunk execution.

    Legacy servo options remain parseable, but no visual/grip helper owns motion.
    """
