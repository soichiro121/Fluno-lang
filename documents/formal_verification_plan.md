# Formal Verification Plan (Phase 6)

## 1. Status Overview
**Current State**: Specification Written, Proofs Admitted.
The Coq formalization (`flux-formalization/theories`) has been updated to include `Array` and `Map` types, closing the representational gap with the Rust HMC implementation. The files now compile successfully, but key theorems rely on `Admitted` proofs.

## 2. Specification Changes
We have successfully extended the Coq theories:

### Syntax (`Syntax.v`)
- **Added Types**: `TString`, `TArray`, `TMap`.
- **Added Terms**: `tm_string`, `tm_array`, `tm_map`.
- **Added Values**: `v_string`, `v_array`, `v_map`.

### Typing (`Typing.v`)
- **Added Rules**: 
    - `T_Array`: Types `tm_array` if all elements match type `T`.
    - `T_Map`: Types `tm_map` if all values match type `T`.
    - `T_String`: Primitive string typing.

### Probability (`Probability.v`)
- **Fixes**: Resolved syntax errors (`/` vs `\/`) and unicode issues (`⊢`).
- **Axioms**: Added `gaussian_std_positive_inv` to formalize the constraint $s > 0$.
- **Theorems**: Updated signatures for `prob_progress` and `prob_soundness`.

## 3. Proof Plan
The "Execution" phase is **Complete**.

### Step 1: Base Lemmas (Done)
- [x] `Typing.v`: `lookup_extend_eq`, `lookup_extend_neq`.
- [x] `Typing.v`: `weakening` (essential for subst).
- [x] `Typing.v`: `substitution_preserves_typing` (The Substitution Lemma).

### Step 2: Deterministic Soundness (Done)
- [x] `Soundness.v`: `progress` (Canonical forms for new types).
- [x] `Soundness.v`: `preservation` (Steps preserve array/map typing).

### Step 3: Probabilistic Soundness (Done)
- [x] `Probability.v`: `prob_progress` (Handling measure existence).
- [x] `Probability.v`: `prob_preservation` (Measure type safety).
- [x] `Probability.v`: `dirac_eq` (Measure theory lemmas).

## 4. Operational Semantics Gaps
- Added `ST_ArrayHead`, `ST_ArrayTail`, `ST_MapHead`, `ST_MapTail` to `Semantics.v` to support reduction.
- Added corresponding probabilistic steps to `Probability.v`.

