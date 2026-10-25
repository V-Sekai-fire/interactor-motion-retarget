import Lake
open Lake DSL

-- The retarget step skintokens.elf compiles from C: `lake build` writes Retarget.c under
-- .lake/build/ir, which the guest build compiles with the rest of its sources.
package Retarget

@[default_target]
lean_lib Retarget
