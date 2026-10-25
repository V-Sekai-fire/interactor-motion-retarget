/-
Retarget: one SOMA30 walk clip onto a rig, in a workspace that never grows.

The C++ side (guest/skintokens/retarget_lean.cpp) does the once-per-rig work: it reads the rig,
matches its humanoid joints and writes the plan below. Every clip then comes through `clip`,
which reads the clip from the workspace and writes the rig's keyframes back into the same
workspace. `clip` changes values only; `clip_size` proves it never changes the workspace's size,
so a uniquely held workspace is updated in place and the guest's memory stays the same for any
number of clips.

The float steps follow the C++ they replace (skin-tokens-ggml retarget.cpp
retarget_soma30_motion_to_generated, glb.cpp add_animations_glb_memory and motion_clip.cpp
quat_from_rows) in Float32, operation for operation.

Plan `p` (integers are stored exactly as floats):
  0 J target joints · 1 S clip joints · 2 root needs the parent transform (0/1) · 3 target height
  4..6 target root rest · 7..22 to-parent matrix (column major, as glb.cpp's matrix4)
  23 head · 24 left toe · 25 right toe (clip joint indices) · 26.. the clip's S parents, each plus one (0: the root)
  then J offsets, one per target, each to: pf(4) of(4) path length n, path (n clip joints).
Workspace `w`: 0 travel (written) · 1.. the clip, F*S rows of 12 · then the keyframes:
  F root positions (3 each), then per target joint F rotations (4 each, x y z w).
-/
namespace Retarget

@[inline] def rd (a : FloatArray) (i : Nat) : Float32 := (a.get! i).toFloat32
@[inline] def ri (a : FloatArray) (i : Nat) : Nat := (a.get! i).toUInt64.toNat

structure Q where
  x : Float32
  y : Float32
  z : Float32
  w : Float32

@[inline] def qmul (a b : Q) : Q :=
  ⟨a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
   a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z⟩

@[inline] def qconj (a : Q) : Q := ⟨-a.x, -a.y, -a.z, a.w⟩

/-! Float32 constants, bit for bit as the C++ literals (an OfNat or OfScientific literal would call
Init's conversion code, which the guest does not compile). -/
@[inline] def c0 : Float32 := Float32.ofBits 0x00000000
@[inline] def c1 : Float32 := Float32.ofBits 0x3f800000
@[inline] def c2 : Float32 := Float32.ofBits 0x40000000
@[inline] def cQuarter : Float32 := Float32.ofBits 0x3e800000
@[inline] def cHalf : Float32 := Float32.ofBits 0x3f000000
/-- 1.0e-12F -/
@[inline] def tiny : Float32 := Float32.ofBits 0x2b8cbccc

@[inline] def qnorm (a : Q) : Q :=
  let l := Float32.sqrt (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w)
  if l > tiny then ⟨a.x / l, a.y / l, a.z / l, a.w / l⟩ else ⟨c0, c0, c0, c1⟩

/-- motion_clip.cpp quat_from_rows: the rotation of one 3x3 row-major block at `b`. -/
@[inline] def fromRows (w : FloatArray) (b : Nat) : Q :=
  let m00 := rd w b; let m01 := rd w (b + 1); let m02 := rd w (b + 2)
  let m10 := rd w (b + 3); let m11 := rd w (b + 4); let m12 := rd w (b + 5)
  let m20 := rd w (b + 6); let m21 := rd w (b + 7); let m22 := rd w (b + 8)
  let tr := m00 + m11 + m22
  let q : Q :=
    if tr > c0 then
      let s := Float32.sqrt (tr + c1) * c2
      ⟨(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, cQuarter * s⟩
    else if m00 > m11 && m00 > m22 then
      let s := Float32.sqrt (c1 + m00 - m11 - m22) * c2
      ⟨cQuarter * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s⟩
    else if m11 > m22 then
      let s := Float32.sqrt (c1 + m11 - m00 - m22) * c2
      ⟨(m01 + m10) / s, cQuarter * s, (m12 + m21) / s, (m02 - m20) / s⟩
    else
      let s := Float32.sqrt (c1 + m22 - m00 - m11) * c2
      ⟨(m02 + m20) / s, (m12 + m21) / s, cQuarter * s, (m10 - m01) / s⟩
  let n := Float32.sqrt (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
  ⟨q.x / n, q.y / n, q.z / n, q.w / n⟩

/-- Writes four values at `i`. -/
@[inline] def put4 (w : FloatArray) (i : Nat) (q : Q) : FloatArray :=
  ((((w.set! i q.x.toFloat).set! (i + 1) q.y.toFloat).set! (i + 2) q.z.toFloat).set! (i + 3) q.w.toFloat)

@[inline] def put3 (w : FloatArray) (i : Nat) (x y z : Float32) : FloatArray :=
  ((w.set! i x.toFloat).set! (i + 1) y.toFloat).set! (i + 2) z.toFloat

/-- Where clip row (frame f, joint j) starts. -/
@[inline] def row (s f j : Nat) : Nat := 1 + (f * s + j) * 12

/-- The local rotation of one target: the clip's local rotations along its path, multiplied in
order (retarget.cpp: local = qnorm(qmul(local, source))), carried as four scalars. -/
def path (p w : FloatArray) (s f : Nat) : Nat → Nat → Float32 → Float32 → Float32 → Float32 → Q
  | 0, _, x, y, z, ww => ⟨x, y, z, ww⟩
  | n + 1, at_, x, y, z, ww =>
    let q := qnorm (qmul ⟨x, y, z, ww⟩ (fromRows w (row s f (ri p at_))))
    path p w s f n (at_ + 1) q.x q.y q.z q.w

/-- Every target joint's rotation for frame f, in the joint's own frame (glb.cpp:
q_norm(q_mul(q_mul(q_conj(parent_frame), local), own_frame))). -/
def targets (p : FloatArray) (s nf f out : Nat) : Nat → Nat → FloatArray → FloatArray
  | 0, _, w => w
  | k + 1, t, w =>
    let o := ri p (26 + s + t)
    let pf : Q := ⟨rd p o, rd p (o + 1), rd p (o + 2), rd p (o + 3)⟩
    let of : Q := ⟨rd p (o + 4), rd p (o + 5), rd p (o + 6), rd p (o + 7)⟩
    let lr := path p w s f (ri p (o + 8)) (o + 9) c0 c0 c0 c1
    let r := qnorm (qmul (qmul (qconj pf) lr) of)
    targets p s nf f out k (t + 1) (put4 w (out + 3 * nf + (t * nf + f) * 4) r)

structure V where
  x : Float32
  y : Float32
  z : Float32

/-- A clip joint's rest position: its parents' frame-0 offsets added root first (motion_clip.cpp
soma30_motion_from_transforms). `fuel` is the clip's joint count, the longest parent chain. -/
def rest (p w : FloatArray) (s : Nat) : Nat → Nat → V
  | 0, _ => ⟨c0, c0, c0⟩
  | fuel + 1, j =>
    let b := row s 0 j + 9
    let off : V := ⟨rd w b, rd w (b + 1), rd w (b + 2)⟩
    -- the plan stores parent + 1, so 0 is the root
    let parent := ri p (26 + j)
    if parent == 0 then off
    else
      let r := rest p w s fuel (parent - 1)
      ⟨r.x + off.x, r.y + off.y, r.z + off.z⟩

@[inline] def dist (a b : V) : Float32 :=
  let dx := a.x - b.x; let dy := a.y - b.y; let dz := a.z - b.z
  Float32.sqrt (dx * dx + dy * dy + dz * dz)

/-- Root travel scale for this clip, or 0 for a degenerate skeleton (1.0e-8F as in retarget.cpp). -/
def travel (p w : FloatArray) (s : Nat) : Float32 :=
  let head := rest p w s s (ri p 23)
  let lt := rest p w s s (ri p 24)
  let rt := rest p w s s (ri p 25)
  let feet : V := ⟨(lt.x + rt.x) * cHalf, (lt.y + rt.y) * cHalf, (lt.z + rt.z) * cHalf⟩
  let h := dist head feet
  let th := rd p 3
  let eps := Float32.ofBits 0x322bcc77
  if h ≤ eps || th ≤ eps then c0 else th / h

/-- The hips position for frame f: the target root rest plus the clip's root travel since frame 0,
scaled; below another joint it goes into that joint's space (glb.cpp transform_point). -/
@[inline] def root (p w : FloatArray) (s f : Nat) (k : Float32) : V :=
  let b := row s f 0 + 9; let b0 := row s 0 0 + 9
  let x := rd p 4 + (rd w b - rd w b0) * k
  let y := rd p 5 + (rd w (b + 1) - rd w (b0 + 1)) * k
  let z := rd p 6 + (rd w (b + 2) - rd w (b0 + 2)) * k
  if ri p 2 == 0 then ⟨x, y, z⟩
  else
    ⟨rd p 7 * x + rd p 11 * y + rd p 15 * z + rd p 19, rd p 8 * x + rd p 12 * y + rd p 16 * z + rd p 20,
     rd p 9 * x + rd p 13 * y + rd p 17 * z + rd p 21⟩

def frames (p : FloatArray) (s j nf out : Nat) (k : Float32) : Nat → Nat → FloatArray → FloatArray
  | 0, _, w => w
  | n + 1, f, w =>
    let r := root p w s f k
    let w := put3 w (out + f * 3) r.x r.y r.z
    frames p s j nf out k n (f + 1) (targets p s nf f out j 0 w)

/-- One clip of `nf` frames, already in the workspace: its keyframes replace the workspace's
keyframe region, and slot 0 gets the travel scale (0: the clip was refused). -/
def clip (p : FloatArray) (nf : Nat) (w : FloatArray) : FloatArray :=
  let s := ri p 1
  let j := ri p 0
  let k := travel p w s
  let w := w.set! 0 k.toFloat
  if k == c0 then w else frames p s j nf (1 + nf * s * 12) k nf 0 w

@[export retarget_lean_clip]
def clipC (p : FloatArray) (nf : USize) (w : FloatArray) : FloatArray := clip p nf.toNat w

/-! ## The workspace never grows -/

theorem size_set! (a : FloatArray) (i : Nat) (v : Float) : (a.set! i v).size = a.size := by
  cases a; simp [FloatArray.set!, FloatArray.size]

theorem put4_size (w : FloatArray) (i : Nat) (q : Q) : (put4 w i q).size = w.size := by
  simp [put4, size_set!]

theorem put3_size (w : FloatArray) (i : Nat) (x y z : Float32) : (put3 w i x y z).size = w.size := by
  simp [put3, size_set!]

theorem targets_size (p : FloatArray) (s nf f out k t : Nat) (w : FloatArray) :
    (targets p s nf f out k t w).size = w.size := by
  induction k generalizing t w with
  | zero => simp [targets]
  | succ k ih => simp only [targets]; rw [ih, put4_size]

theorem frames_size (p : FloatArray) (s j nf out : Nat) (k : Float32) (n f : Nat) (w : FloatArray) :
    (frames p s j nf out k n f w).size = w.size := by
  induction n generalizing f w with
  | zero => simp [frames]
  | succ n ih => simp only [frames]; rw [ih, targets_size, put3_size]

/-- Retargeting a clip never changes the workspace's size: for any plan, clip length and
workspace, the buffer that comes back is exactly as long as the one that went in. -/
theorem clip_size (p : FloatArray) (nf : Nat) (w : FloatArray) : (clip p nf w).size = w.size := by
  simp only [clip]
  split <;> simp only [frames_size, size_set!]

end Retarget
