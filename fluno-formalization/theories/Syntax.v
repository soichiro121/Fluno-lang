(* ================================================================= *)
(* Syntax.v: 構文の定義                                               *)
(* ================================================================= *)

From Flux Require Import Base.
From Coq Require Import Strings.String.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
From Coq Require Import Lists.List.
From Coq Require Import Bool.Bool.
From Coq Require Import ZArith.
From Coq Require Import Reals.
Import ListNotations.

(* ================================================================= *)
(* 型                                                                 *)
(* ================================================================= *)

Inductive ty : Type :=
  | TInt : ty
  | TFloat : ty
  | TBool : ty
  | TGaussian : ty
  | TFun : ty -> ty -> ty
  | TUnit : ty
  | TString : ty
  | TArray : ty -> ty
  | TMap : ty -> ty.

(* 型の決定可能性 *)
Theorem eq_ty_dec : forall (T1 T2 : ty), {T1 = T2} + {T1 <> T2}.
Proof.
  decide equality.
Defined.

(* 型の等価性判定 *)
Fixpoint beq_ty (T1 T2 : ty) : bool :=
  match T1, T2 with
  | TInt, TInt => true
  | TFloat, TFloat => true
  | TBool, TBool => true
  | TGaussian, TGaussian => true
  | TFun t11 t12, TFun t21 t22 => 
      andb (beq_ty t11 t21) (beq_ty t12 t22)
  | TUnit, TUnit => true
  | TString, TString => true
  | TArray t1, TArray t2 => beq_ty t1 t2
  | TMap t1, TMap t2 => beq_ty t1 t2
  | _, _ => false
  end.

Lemma beq_ty_refl : forall T, beq_ty T T = true.
Proof.
  induction T; simpl; auto.
  - rewrite IHT1, IHT2. reflexivity.
Qed.

Lemma beq_ty_true_iff : forall T1 T2,
  beq_ty T1 T2 = true <-> T1 = T2.
Proof.
  split.
  - generalize dependent T2.
    induction T1; intros; destruct T2; simpl in *; 
      try discriminate; try reflexivity.
    + apply andb_true_iff in H. destruct H.
      f_equal; [apply IHT1_1 | apply IHT1_2]; auto.
    + f_equal. apply IHT1. auto.
    + f_equal. apply IHT1. auto.
  - intros. subst. apply beq_ty_refl.
Qed.

(* ================================================================= *)
(* 項（term）                                                         *)
(* ================================================================= *)

Inductive tm : Type :=
  | tm_var : nat -> tm
  | tm_int : Z -> tm
  | tm_float : R -> tm
  | tm_bool : bool -> tm
  | tm_gaussian : R -> R -> tm
  | tm_add : tm -> tm -> tm
  | tm_sample : tm -> tm
  | tm_abs : nat -> ty -> tm -> tm
  | tm_app : tm -> tm -> tm
  | tm_unit : tm
  | tm_string : string -> tm
  | tm_array : list tm -> tm
  | tm_map : list (string * tm) -> tm.

(* Custom induction principle for nested inductive types *)
Section TmInd.
  Variable P : tm -> Prop.
  
  Variable HVar : forall n, P (tm_var n).
  Variable HInt : forall n, P (tm_int n).
  Variable HFloat : forall r, P (tm_float r).
  Variable HBool : forall b, P (tm_bool b).
  Variable HGaussian : forall m s, P (tm_gaussian m s).
  Variable HAdd : forall t1 t2, P t1 -> P t2 -> P (tm_add t1 t2).
  Variable HSample : forall t, P t -> P (tm_sample t).
  Variable HAbs : forall x T t, P t -> P (tm_abs x T t).
  Variable HApp : forall t1 t2, P t1 -> P t2 -> P (tm_app t1 t2).
  Variable HUnit : P tm_unit.
  Variable HString : forall s, P (tm_string s).
  Variable HArray : forall vs, Forall P vs -> P (tm_array vs).
  Variable HMap : forall ps, Forall (fun p => P (snd p)) ps -> P (tm_map ps).

  Fixpoint tm_ind_custom (t : tm) : P t :=
    match t with
    | tm_var n => HVar n
    | tm_int n => HInt n
    | tm_float r => HFloat r
    | tm_bool b => HBool b
    | tm_gaussian m s => HGaussian m s
    | tm_add t1 t2 => HAdd t1 t2 (tm_ind_custom t1) (tm_ind_custom t2)
    | tm_sample t => HSample t (tm_ind_custom t)
    | tm_abs x T t => HAbs x T t (tm_ind_custom t)
    | tm_app t1 t2 => HApp t1 t2 (tm_ind_custom t1) (tm_ind_custom t2)
    | tm_unit => HUnit
    | tm_string s => HString s
    | tm_array vs => HArray vs 
        ((fix list_ind (ls : list tm) : Forall P ls :=
            match ls with
            | nil => Forall_nil P
            | cons t ts => Forall_cons t (tm_ind_custom t) (list_ind ts)
            end) vs)
    | tm_map ps => HMap ps
        ((fix map_ind (qs : list (string * tm)) : Forall (fun p => P (snd p)) qs :=
            match qs with
            | nil => Forall_nil (fun p => P (snd p))
            | cons p qs' => Forall_cons p (tm_ind_custom (snd p)) (map_ind qs')
            end) ps)
    end.
End TmInd.

(* ================================================================= *)
(* 値                                                                 *)
(* ================================================================= *)

Inductive value : tm -> Prop :=
  | v_int : forall n, value (tm_int n)
  | v_float : forall r, value (tm_float r)
  | v_bool : forall b, value (tm_bool b)
  | v_gaussian : forall m s, value (tm_gaussian m s)
  | v_abs : forall x T t, value (tm_abs x T t)
  | v_unit : value tm_unit
  | v_string : forall s, value (tm_string s)
  | v_array : forall vs, Forall value vs -> value (tm_array vs)
  | v_map : forall ps, Forall (fun p => value (snd p)) ps -> value (tm_map ps).

Hint Constructors value : core.

(* ================================================================= *)
(* 自由変数                                                           *)
(* ================================================================= *)

Inductive appears_free_in : nat -> tm -> Prop :=
  | afi_var : forall x,
      appears_free_in x (tm_var x)
  | afi_add1 : forall x t1 t2,
      appears_free_in x t1 ->
      appears_free_in x (tm_add t1 t2)
  | afi_add2 : forall x t1 t2,
      appears_free_in x t2 ->
      appears_free_in x (tm_add t1 t2)
  | afi_sample : forall x t,
      appears_free_in x t ->
      appears_free_in x (tm_sample t)
  | afi_app1 : forall x t1 t2,
      appears_free_in x t1 ->
      appears_free_in x (tm_app t1 t2)
  | afi_app2 : forall x t1 t2,
      appears_free_in x t2 ->
      appears_free_in x (tm_app t1 t2)
  | afi_abs : forall x y T t,
      y <> x ->
      appears_free_in x t ->
      appears_free_in x (tm_abs y T t)
  | afi_array : forall x ts t,
      In t ts ->
      appears_free_in x t ->
      appears_free_in x (tm_array ts)
  | afi_map : forall x ps k t,
      In (k, t) ps ->
      appears_free_in x t ->
      appears_free_in x (tm_map ps).

Hint Constructors appears_free_in : core.

(* ================================================================= *)
(* 代入                                                               *)
(* ================================================================= *)

Fixpoint subst (x : nat) (s : tm) (t : tm) : tm :=
  match t with
  | tm_var y => if beq_id x y then s else t
  | tm_int n => tm_int n
  | tm_float r => tm_float r
  | tm_bool b => tm_bool b
  | tm_gaussian m std => tm_gaussian m std
  | tm_add t1 t2 => tm_add (subst x s t1) (subst x s t2)
  | tm_sample t1 => tm_sample (subst x s t1)
  | tm_app t1 t2 => tm_app (subst x s t1) (subst x s t2)
  | tm_abs y T t1 => 
      if beq_id x y then t 
      else tm_abs y T (subst x s t1)
  | tm_unit => tm_unit
  | tm_string s => tm_string s
  | tm_array ts => tm_array (map (subst x s) ts)
  | tm_map ps => tm_map (map (fun p => (fst p, subst x s (snd p))) ps)
  end.

        Notation "'[' x ':=' s ']' t" := (subst x s t) (at level 20).
