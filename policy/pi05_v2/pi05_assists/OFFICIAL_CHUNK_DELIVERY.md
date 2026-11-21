# Use the organizer's original ChunkBridge

The policy's public `deploy_policy.eval()` is now directly accepted by
`rsc_real.chunks.intercept_eval`. No custom bridge subclass or modified organizer
source is required.

## Execution

At the beginning of eval, `chunk_delivery.prepare_eval_chunk()` consumes the
boundary observation once, runs one model inference and completes all auxiliary
planning. It copies the entire executable block before entering this loop:

```python
for action_tensor in action_tensors:
    final_obs, reward, terminated, truncated, info = env.step(action_tensor)
    info, truncated_done, truncated = block.after_step(terminated, truncated, info)
    if truncated:
        break
```

The unchanged official `intercept_eval()` rewrites the loop iterator to
`env.action_chunk(action_tensors)`. Thus one station infer request receives the
whole precomputed block. `env.step()` then verifies and acknowledges each row
from the real feedback queue. The loop never encodes observations, invokes the
model, or changes future actions. `block.after_step()` commits only executed
frames and proposes the next cached command. A new inference happens only at
the next block boundary.

Auxiliary failures in the six assisted tasks use the valid model prefix as a
complete fallback block and retry at the next boundary. All blocks respect the
remaining 1000-frame episode budget. Existing sampling lengths are unchanged.

## Units and feedback

The normal simulator and original finalist serial bridge continue to receive
finger displacements in metres, produced by `encode_action()` (`opening * .05`).

The generic official ChunkBridge publishes declared values unchanged, so it
needs normalized station commands. Its constructor sets the exact metadata
`adaptation.kind = "precomputed-action-loop-v1"`. On this path only,
`prepare_eval_chunk()` applies `station_tensor()` to the metre commands: it
checks the [0, .05] range, copies them, and divides fingers 6 and 13 by .05.
This preserves the organizer's serial conversion without applying it twice.
Internal policy commands and measured observations keep their original units.

Real feedback is still required for each executed frame. "One observation per
block" means one observation consumed by policy inference/planning, not removing
station measurement messages. No commanded state is substituted for measured
feedback.

## Local/server construction

With the supplied, unchanged rsc_real package on PYTHONPATH:

```python
from rsc_real.chunks import ChunkBridge
from policy.pi05_v2 import deploy_policy as adapter

# The organizer must validate source/checkpoint receipts before construction.
model = adapter.get_model(verified_args)
bridge = ChunkBridge(adapter, model, instruction)
```

The object returned by the organizer's prepared-policy context can also be used
as `adapter`: its eval points to the compatible public callback.

This is an implementation ready for the provided chunk entry point, not a
change to the live station binding. The frozen Click Bell configuration still
selects the serial adapter. If the service continues to instantiate that serial
adapter, delivery remains one frame per request. The attachment lacks the full
server dispatcher; the organizers still need to select their chunk entry point
for deployment. We have not altered their files, hashes, receipts or registry.

The unmodified ChunkBridge's `finish()` requires the declared block to be fully
acknowledged. For an interrupted block, do not send fictitious feedback to fill
its suffix. The local tests submit only feedback for executed frames, wait for
those acknowledgements, and use `close()` to abort the rest. Full station abort
handling must follow its actual client/service lifecycle.

## Reproducible checks

From the submission repository root, with the organizer attachment unzipped
and an evaluator Python containing torch, numpy, scipy, OpenCV and pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/policy/pi05_v2" \
RSC_BASELINE_DIR=/path/to/finalist_10-adaptation-baseline \
python -m pytest -q -p no:cacheprovider policy/pi05_v2/tests
```

These CPU checks use actual organizer classes and synthetic model/observation
fixtures. They do not load checkpoints or reproduce the reported real episode.
The bridge tests require `RSC_BASELINE_DIR`; without it that module is skipped.
If the variable is provided but points to an invalid directory, collection fails.

- All six organizer source hashes remain equal to BASELINE.json.
- `intercept_eval(deploy_policy.eval)` accepts the callback and returns its
  original `precomputed-action-loop-v1` metadata. The bridge type is exactly
  `rsc_real.chunks.ChunkBridge`, not a replacement/subclass.
- Click Bell: 446 synthetic executed frames require 446 serial requests versus
  45 chunk requests. Both receive 446 measured-feedback fixtures. The last four
  declared but unexecuted rows are aborted, never acknowledged with fake state.
- All ten task profiles execute/acknowledge 1000 frames through the original
  ChunkBridge and ChunkGuard, with 100/125/167 requests for 10/8/6-frame blocks.
- A 123-frame synthetic assist path is delivered once. Failed auxiliary planning
  also delivers a full model block and retries only at the next boundary.
- Unit conversion, measured-state propagation, missing feedback, action mutation,
  full-block finish, prefix abort, serial compatibility and simulator early-success
  behavior are checked. Declaration alone does not increment executed frames.

Worker read/wait behavior is unchanged: a worker that
never replies can still block the policy read. A worker exception or invalid
model output now raises `PolicyFailure`, latched until reset, instead of sending
holding actions. Tests verify the unchanged serial and chunk bridges promptly
forward the error. The attachment does not include the outer RPC failure schema,
so this verifies the bridge error channel only, not the live service response.

All ten tasks were tested locally for ten simulation episodes each. Timing
results and local checkpoint bindings are supplied separately with the correction
report. These runs do not reproduce the reported station episode and predate
the error-only `PolicyFailure` patch, which has separate fault-injection coverage.
Pipette planning still has long responses (maximum 26.503 s); no worker-read or
global planning wall-clock deadline was added.
