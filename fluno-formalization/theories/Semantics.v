(* ================================================================= *)
(* Semantics.v: 小ステップ意味論（決定的部分）                          *)
(* ================================================================= *)

From Flux Require Import Base Syntax Typing.
From Coq Require Import Reals.
Open Scope R_scope.

(* ================================================================= *)
(* Gaussian演算（抽象化）                                             *)
(* ================================================================= *)

Parameter gaussian_sample : R -> R -> R.

Axiom gaussian_sample_bounds : forall m s r,
  r = gaussian_sample m s ->
  exists k, Rabs (r - m) <= k * s.

(* ================================================================= *)
(* ステップ関係                                                       *)
(* ================================================================= *)

Reserved Notation "t '-->' t'" (at level 40).

Inductive step : tm -> tm -> Prop :=
  | ST_AddGaussian : forall m1 s1 m2 s2,
      tm_add (tm_gaussian m1 s1) (tm_gaussian m2 s2) -->
      tm_gaussian (m1 + m2) (sqrt (s1 * s1 + s2 * s2))
  
  | ST_Sample : forall m s,
      tm_sample (tm_gaussian m s) -->
      tm_float (gaussian_sample m s)
  
  | ST_AppAbs : forall x T t v,
      value v ->
      tm_app (tm_abs x T t) v --> [x := v] t
  
  (* 合同規則 *)
  | ST_Add1 : forall t1 t1' t2,
      t1 --> t1' ->
      tm_add t1 t2 --> tm_add t1' t2
  
  | ST_Add2 : forall v1 t2 t2',
      value v1 ->
      t2 --> t2' ->
      tm_add v1 t2 --> tm_add v1 t2'
  
  | ST_Sample1 : forall t t',
      t --> t' ->
      tm_sample t --> tm_sample t'
  
  | ST_App1 : forall t1 t1' t2,
      t1 --> t1' ->
      tm_app t1 t2 --> tm_app t1' t2
  
  | ST_App2 : forall v1 t2 t2',
      value v1 ->
      t2 --> t2' ->
      tm_app v1 t2 --> tm_app v1 t2'

  | ST_ArrayHead : forall t t' ts,
      t --> t' ->
      tm_array (t :: ts) --> tm_array (t' :: ts)

  | ST_ArrayTail : forall v ts ts',
      value v ->
      tm_array ts --> tm_array ts' ->
      tm_array (v :: ts) --> tm_array (v :: ts')

  | ST_MapHead : forall k t t' ps,
      t --> t' ->
      tm_map ((k, t) :: ps) --> tm_map ((k, t') :: ps)

  | ST_MapTail : forall k v ps ps',
      value v ->
      tm_map ps --> tm_map ps' ->
      tm_map ((k, v) :: ps) --> tm_map ((k, v) :: ps')

where "t '-->' t'" := (step t t').

Hint Constructors step : core.

(* ================================================================= *)
(* 多ステップ簡約                                                     *)
(* ================================================================= *)

Inductive multi {X : Type} (R : X -> X -> Prop) : X -> X -> Prop :=
  | multi_refl : forall x, multi R x x
  | multi_step : forall x y z,
      R x y ->
      multi R y z ->
      multi R x z.

Notation "t '-->*' t'" := (multi step t t') (at level 40).

Hint Constructors multi : core.

Theorem multi_trans : forall t1 t2 t3,
  t1 -->* t2 ->
  t2 -->* t3 ->
  t1 -->* t3.
Proof.
  intros. induction H.
  - assumption.
  - eapply multi_step; eauto.
Qed.
