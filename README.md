# interactor-motion-retarget

Puts one SOMA30 motion clip onto a humanoid rig, in Lean 4, in a workspace that never grows.

## What it is for

A sandbox guest that makes walk-mode clips must fit in a fixed memory arena, also in a browser. The per-clip step here reads a clip from a workspace buffer and writes the rig's keyframes back into the same buffer. `Retarget.clip_size` proves that the step never changes the buffer's size. A buffer that only one reference holds is updated in place, so the guest's memory stays the same for any number of clips.

The float steps follow skin-tokens-ggml's C++ (`retarget_soma30_motion_to_generated`, `add_animations_glb_memory` and `quat_from_rows`) in `Float32`, operation for operation. The once-per-rig work (reading the rig and matching its humanoid joints) stays in the guest's C++. That code writes the plan the clip step reads.

## Build and run

`lake build` checks the proofs and writes `.lake/build/ir/Retarget.c`. A guest compiles that C file with its own sources and the Lean runtime C that it calls, using its own compiler and flags. Nothing is linked from a prebuilt Lean library. The exported entry point is `retarget_lean_clip(plan, frames, workspace)`.

The plan and workspace layouts are in the comment at the top of `Retarget.lean`.

## Licence

MIT. See `LICENSE`.
