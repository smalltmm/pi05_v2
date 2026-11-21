# Assisted action blocks

`FinalPolicy` uses `ChunkRuntime` in `chunk_runtime.py`. At a block boundary it
encodes one observation, performs one inference, and plans the executable block.
Subsequent `next_action(None)` calls return cached commands. The caller still
acknowledges each successfully executed action with `commit_action()`.
This contract applies to every task, including inserted assist actions. Block
length is not restricted to 10-20 frames: execute the complete planned block
before consuming another observation, subject to the remaining episode budget
and early success/termination.

The public eval callback precomputes the complete action list and exposes the
simple step loop required by the organizer's unchanged `ChunkBridge`. Its AST
adapter inserts `env.action_chunk` once at the block boundary. No custom bridge
is required. Measured feedback still acknowledges each executed frame; with the
original serial entry point, delivery remains one frame per request. See
`OFFICIAL_CHUNK_DELIVERY.md` for units, integration and request-count checks.

The planner starts from the block's measured state and advances a **planned
command cursor**. This cursor is never treated as evidence of contact, a grasp,
or physical arrival. The numerical FK/IK and trajectory builders are reused;
the old feedback-gated runtime is not used to execute commands.

Model prefixes remain 10 control frames for click_bell/items_handover/sample_loading,
6 for drawer_open_place, and 8 for the other default profiles. Assistance can
insert additional frames or replace the remainder with a complete scripted path.
Motion keeps the existing 10 Hz sampling and speed settings. Scripted pose
changes end the block instead of replaying stale model targets afterward.

- Bell: detect descent in the planned trajectory, insert the existing press/hold,
  and retain press/rearm state between blocks.
- Basket: retain left/right grip holding. Left holding requires a predicted
  open -> close transition; initial closed -> open is not treated as release.
  Release does not insert forward motion or hold arm poses. All arm joints
  continue to follow the model block.
- Table: grip holding can latch only when the predicted arm pose meets that
  side's enabled grasp-dip height range and downward tilt limit. Existing holds
  persist during lifting; model release inserts the lowering path and fixed
  dwell before opening. Outside the gate, model gripper commands are preserved.
- Handover: trigger from predicted handover geometry and gripper intent, retain
  a fixed 0.4 s (4 additional frames at 10 Hz) dwell after the right-grip close
  command latches in items_handover, before subsequent motion or transfer;
  both arms retain that command during the dwell. No feedback is checked.
  Retain
  relative approach for sample_loading, close the left gripper for 3 frames
  (0.3 s), then ramp the right gripper open. There is no contact confirmation.
  Sample loading gates initial right grip holding by its grasp-dip height and
  downward tilt limits, using the predicted pose. Its closing threshold also
  respects the dip closing threshold so holding cannot skip preparation.
  Scripted pickup closure immediately latches grip holding; resumed model rows
  cannot reopen it before transfer. After either scripted or model-triggered
  right pickup closure, add 4 fixed frames (0.4 s) holding both arms before
  lifting or subsequent motion. The same closure does not add another dwell.
  At transfer, hold the receiver while commanding the donor
  pose, dwell for relative-handover stable_frames (default 5, or 0.5 s), then
  approach with only the left arm; donor joint commands stay fixed. These are
  timed commands, not physical stability confirmations.
  Sample placement follows its planned approach, descent,
  dwell and release.
- Assembly: model commands only; no dip, grip hold, lift, or insertion assistance,
  including when legacy assist flags are supplied.
- Pipette: retain predicted right-arm trajectory stability as the press trigger,
  with a model-command grip latch gated by the right grasp-dip height range and
  downward tilt limit. Once latched, grip holding persists during lifting and
  pressing. Consume the complete planned press path.
- Pouring: record planned pickup poses on closing commands, trigger in the
  predicted pouring region, and schedule alignment, tilt, hold, return, release.
- After pouring or sample placement completes, the next observation starts a
  fresh model block instead of `completed_hold`. Completed assistance stays
  latched until episode reset and no longer alters joints or gripper commands;
  standard sampling, gripper-unit compensation and limits still apply. The
  scripted release dwell within its original block is retained.
- Drawer and mixer: model commands only (plus standard sampling/action encoding).
  Mixer visual servo and grip assistance are disabled, including legacy flags.

Planning is transactional. The planner is copied before generating a block;
only a fully acknowledged block publishes its future helper state. Episode
reset discards all unexecuted commands. The executor's frame count advances
only on acknowledgements and includes inserted frames; blocks are capped by
the remaining 1000-frame budget. In click_bell, sample_loading, items_handover,
manipulate_pipette, table_rearrangement, and water_pouring, auxiliary planning
failure discards the speculative helper changes and executes the current model
prefix without geometric or grip assistance. Sampling, configured gripper-unit
compensation and joint limits still apply. The whole fallback block is cached
and acknowledged normally; the next observation starts a new planning attempt.
Records identify these frames as model_fallback with the assist error. Invalid
model output cannot be used for fallback. Nonfinite/malformed outputs and errors
that escape recoverable auxiliary planning raise `PolicyFailure` immediately.
The failed episode stays failed until reset; no holding action is substituted.
The organizer bridge forwards this exception through its existing error channel.

The policy does not consume intermediate observations. The simulator's existing
`env.step()` may still generate them internally; this change does not modify
the simulator's rendering/observation pipeline. Success and truncation checks
in the evaluator still stop execution early without consuming the remainder.

Evaluation timing samples count block generation, not cached action delivery.
