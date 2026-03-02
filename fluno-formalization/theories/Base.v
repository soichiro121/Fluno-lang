(* ================================================================= *)
(* Base.v: 基礎的な定義と補題                                          *)
(* ================================================================= *)

From Coq Require Import Arith.Arith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
From Coq Require Import Logic.FunctionalExtensionality.
From Coq Require Import Lia.
Import ListNotations.

(* ================================================================= *)
(* 識別子                                                             *)
(* ================================================================= *)

Definition id := nat.

Definition beq_id (x y : id) : bool := Nat.eqb x y.

Theorem beq_id_refl : forall x, beq_id x x = true.
Proof. intros. unfold beq_id. apply Nat.eqb_refl. Qed.

Theorem beq_id_true_iff : forall x y,
  beq_id x y = true <-> x = y.
Proof.
  intros. unfold beq_id.
  rewrite Nat.eqb_eq. reflexivity.
Qed.

Theorem beq_id_false_iff : forall x y,
  beq_id x y = false <-> x <> y.
Proof.
  intros. unfold beq_id.
  rewrite Nat.eqb_neq. reflexivity.
Qed.

Theorem eq_id_dec : forall x y : id, {x = y} + {x <> y}.
Proof.
  intros. destruct (Nat.eq_dec x y).
  - left. assumption.
  - right. assumption.
Defined.

(* ================================================================= *)
(* 総称的な補題                                                       *)
(* ================================================================= *)

Ltac solve_by_invert :=
  match goal with
  | H : _ |- _ => solve [inversion H]
  end.

Ltac solve_by_invert_step :=
  match goal with
  | H : _ |- _ => inversion H; subst; clear H
  end.

(* ================================================================= *)
(* 時間                                                               *)
(* ================================================================= *)

Definition Time := nat.

(* ================================================================= *)
(* Option と Result                                                   *)
(* ================================================================= *)

(* 標準ライブラリのOptionを使用 *)
(* List.option を使う *)

(* Result型 *)
Inductive result (A E : Type) : Type :=
  | Ok : A -> result A E
  | Error : E -> result A E.

Arguments Ok {A E}.
Arguments Error {A E}.
