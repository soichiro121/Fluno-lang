(* ================================================================= *)
(* Typing.v: 型付け規則                                               *)
(* ================================================================= *)

From Flux Require Import Base Syntax.
From Coq Require Import Strings.String.
From Coq Require Import Lists.List.
Import ListNotations.

(* ================================================================= *)
(* 型環境                                                             *)
(* ================================================================= *)

Definition context := list (nat * ty).

Fixpoint lookup (Gamma : context) (x : nat) : option ty :=
  match Gamma with
  | nil => None
  | (y, T) :: Gamma' => 
      if beq_id x y then Some T else lookup Gamma' x
  end.

(* ================================================================= *)
(* 型付け関係                                                         *)
(* ================================================================= *)

Reserved Notation "Gamma '⊢' t '∈' T" (at level 40).

Inductive has_type : context -> tm -> ty -> Prop :=
  | T_Var : forall Gamma x T,
      lookup Gamma x = Some T ->
      Gamma ⊢ tm_var x ∈ T
  
  | T_Int : forall Gamma n,
      Gamma ⊢ tm_int n ∈ TInt
  
  | T_Float : forall Gamma r,
      Gamma ⊢ tm_float r ∈ TFloat
  
  | T_Bool : forall Gamma b,
      Gamma ⊢ tm_bool b ∈ TBool
  
  | T_Gaussian : forall Gamma m s,
      Gamma ⊢ tm_gaussian m s ∈ TGaussian
  
  | T_Add : forall Gamma t1 t2,
      Gamma ⊢ t1 ∈ TGaussian ->
      Gamma ⊢ t2 ∈ TGaussian ->
      Gamma ⊢ tm_add t1 t2 ∈ TGaussian
  
  | T_Sample : forall Gamma t,
      Gamma ⊢ t ∈ TGaussian ->
      Gamma ⊢ tm_sample t ∈ TFloat
  
  | T_Abs : forall Gamma x T1 T2 t,
      ((x, T1) :: Gamma) ⊢ t ∈ T2 ->
      Gamma ⊢ tm_abs x T1 t ∈ TFun T1 T2
  
  | T_App : forall Gamma t1 t2 T1 T2,
      Gamma ⊢ t1 ∈ TFun T1 T2 ->
      Gamma ⊢ t2 ∈ T1 ->
      Gamma ⊢ tm_app t1 t2 ∈ T2
  
  | T_Unit : forall Gamma,
      Gamma ⊢ tm_unit ∈ TUnit
  
  | T_String : forall Gamma s,
      Gamma ⊢ tm_string s ∈ TString
  
  | T_Array : forall Gamma vs T,
      Forall (fun v => Gamma ⊢ v ∈ T) vs ->
      Gamma ⊢ tm_array vs ∈ TArray T
  
  | T_Map : forall Gamma ps T,
      Forall (fun p => Gamma ⊢ snd p ∈ T) ps ->
      Gamma ⊢ tm_map ps ∈ TMap T

where "Gamma '⊢' t '∈' T" := (has_type Gamma t T).

Hint Constructors has_type : core.

(* ================================================================= *)
(* 型の一意性                                                         *)
(* ================================================================= *)

Theorem type_uniqueness : forall Gamma t T1 T2,
  Gamma ⊢ t ∈ T1 ->
  Gamma ⊢ t ∈ T2 ->
  T1 = T2.
Proof.
  admit.
Admitted.

(* ================================================================= *)
(* 文脈の性質                                                         *)
(* ================================================================= *)

Lemma lookup_extend : forall Gamma x T y,
  lookup ((x, T) :: Gamma) y = 
  if beq_id x y then Some T else lookup Gamma y.
Proof.
  intros. simpl.
  destruct (beq_id y x) eqn:Heq.
  - apply beq_id_true_iff in Heq. subst.
    rewrite beq_id_refl. reflexivity.
  - apply beq_id_false_iff in Heq.
    destruct (beq_id x y) eqn:Heq2.
    + apply beq_id_true_iff in Heq2. subst. congruence.
    + reflexivity.
Qed.

Lemma lookup_extend_eq : forall Gamma x T,
  lookup ((x, T) :: Gamma) x = Some T.
Proof.
  intros. simpl. rewrite beq_id_refl. reflexivity.
Qed.

Lemma lookup_extend_neq : forall Gamma x1 T x2,
  x1 <> x2 ->
  lookup ((x1, T) :: Gamma) x2 = lookup Gamma x2.
Proof.
  intros. simpl.
  apply beq_id_false_iff in H.
  rewrite H. reflexivity.
Qed.

(* ================================================================= *)
(* Weakening                                                          *)
(* ================================================================= *)

Lemma weakening : forall Gamma Gamma' t T,
  (forall x T', lookup Gamma x = Some T' -> 
                lookup Gamma' x = Some T') ->
  Gamma ⊢ t ∈ T ->
  Gamma' ⊢ t ∈ T.
Proof.
  intros Gamma Gamma' t T Hincl Htyp.
  generalize dependent Gamma'.
  induction Htyp; intros Gamma' Hincl; eauto.
  - (* T_Var *)
    apply T_Var. apply Hincl. assumption.
  - (* T_Abs *)
    apply T_Abs.
    apply IHHtyp.
    intros y Ty Hlookup.
    unfold lookup in *.
    destruct (beq_id y x); auto.
    apply Hincl. assumption.
  - (* T_Array *)
    apply T_Array.
    rewrite Forall_forall in *.
    intros x Hin.
    apply H0; auto.
  - (* T_Map *)
    apply T_Map.
    rewrite Forall_forall in *.
    intros x Hin.
    apply H0; auto.
Qed.

Corollary weakening_empty : forall Gamma t T,
  nil ⊢ t ∈ T ->
  Gamma ⊢ t ∈ T.
Proof.
  intros Gamma t T H.
  apply weakening with (Gamma := nil).
  - intros x T' Hlookup. inversion Hlookup.
  - assumption.
Qed.

(* ================================================================= *)
(* 代入補題（完全版）                                                  *)
(* ================================================================= *)

Lemma subst_not_afi : forall t x v,
  ~ appears_free_in x t ->
  [x := v] t = t.
Proof.
  intros t x v H.
  induction t using tm_ind_custom; simpl; auto.
  - (* tm_var *)
    destruct (beq_id x n) eqn:Heq.
    + apply beq_id_true_iff in Heq. subst.
      exfalso. apply H. apply afi_var.
    + reflexivity.
  - (* tm_add *)
    f_equal.
    + apply IHt1. intro. apply H. apply afi_add1. assumption.
    + apply IHt2. intro. apply H. apply afi_add2. assumption.
  - (* tm_sample *)
    f_equal. apply IHt. intro. apply H. apply afi_sample. assumption.
  - (* tm_abs *)
    destruct (beq_id x n) eqn:Heq.
    + reflexivity.
    + apply beq_id_false_iff in Heq.
      f_equal. apply IHt. intro. apply H.
      apply afi_abs; assumption.
  - (* tm_app *)
    f_equal.
    + apply IHt1. intro. apply H. apply afi_app1. assumption.
    + apply IHt2. intro. apply H. apply afi_app2. assumption.
  - (* tm_array *)
    f_equal.
    apply map_ext_in.
    intros t' Hin.
    rewrite Forall_forall in H0.
    apply H0; auto.
    intro Haf. apply H. apply afi_array with (t := t'); assumption.
  - (* tm_map *)
    f_equal.
    apply map_ext_in.
    intros p Hin.
    destruct p as [k val]. simpl.
    f_equal.
    rewrite Forall_forall in H0.
    apply H0; auto.
    intro Haf. apply H. apply afi_map with (k := k) (t := val); assumption.
Qed.

Lemma duplicate_subst : forall t x v,
  ~ appears_free_in x v ->
  [x := v] ([x := v] t) = [x := v] t.
Proof.
  intros t x v H.
  induction t using tm_ind_custom; simpl; auto.
  - (* tm_var *)
    destruct (beq_id x n) eqn:Heq.
    + apply beq_id_true_iff in Heq. subst.
      apply subst_not_afi. assumption.
    + simpl. rewrite Heq. reflexivity.
  - (* tm_add *)
    f_equal; assumption.
  - (* tm_sample *)
    f_equal; assumption.
  - (* tm_abs *)
    destruct (beq_id x n) eqn:Heq.
    + reflexivity.
    + simpl. rewrite Heq. f_equal. assumption.
  - (* tm_app *)
    f_equal; assumption.
  - (* tm_array *)
    f_equal.
    rewrite map_map.
    apply map_ext_in.
    intros t' Hin.
    rewrite Forall_forall in H0.
    apply H0. assumption.
  - (* tm_map *)
    f_equal.
    rewrite map_map.
    apply map_ext_in.
    intros p Hin.
    destruct p as [k val]. simpl.
    f_equal.
    rewrite Forall_forall in H0.
    apply H0. assumption.
Qed.

Lemma subst_closed : forall t,
  (forall x, ~ appears_free_in x t) ->
  forall x v, [x := v] t = t.
Proof.
  intros. apply subst_not_afi. apply H.
Qed.

Lemma context_invariance : forall Gamma Gamma' t T,
  Gamma ⊢ t ∈ T ->
  (forall x, lookup Gamma x = lookup Gamma' x) ->
  Gamma' ⊢ t ∈ T.
Proof.
  intros Gamma Gamma' t T H HLook.
  generalize dependent Gamma'.
  induction H; intros Gamma' HLook; simpl; eauto.
  - (* T_Var *)
    apply T_Var. rewrite <- HLook. assumption.
  - (* T_Abs *)
    apply T_Abs. apply IHhas_type.
    intros y.
    unfold lookup.
    destruct (beq_id y x); auto.
  - (* T_Array *)
    apply T_Array.
    rewrite Forall_forall in *.
    intros v0 Hin. apply H0; auto.
  - (* T_Map *)
    apply T_Map.
    rewrite Forall_forall in *.
    intros p Hin. apply H0; auto.
Qed.

Theorem substitution_preserves_typing : forall Gamma x U t v T,
  ((x, U) :: Gamma) ⊢ t ∈ T ->
  nil ⊢ v ∈ U ->
  Gamma ⊢ [x := v] t ∈ T.
Proof.
  intros Gamma x U t v T Ht Hv.
  remember ((x, U) :: Gamma) as Gamma'.
  generalize dependent Gamma.
  induction Ht; intros Gamma Heq; subst; simpl; eauto.
  
  - (* T_Var *)
    destruct (beq_id x x0) eqn:Heq.
    + (* x = x0 *)
      apply beq_id_true_iff in Heq. subst.
      rewrite lookup_extend_eq in H. inversion H. subst.
      apply weakening_empty. assumption.
    + (* x <> x0 *)
      apply beq_id_false_iff in Heq.
      rewrite lookup_extend_neq in H; auto.
      apply T_Var. assumption.
      
  - (* T_Abs *)
    destruct (beq_id x x0) eqn:Heqx.
    + (* x = x0 (shadowing) *)
      apply beq_id_true_iff in Heqx. subst.
      apply T_Abs.
      apply context_invariance with (Gamma := ((x0, T1) :: (x0, U) :: Gamma)).
      assumption.
      intros z. simpl.
      destruct (beq_id z x0); auto.
    + (* x <> x0 *)
      apply beq_id_false_iff in Heqx.
      apply T_Abs.
      apply IHHt. reflexivity.
      apply context_invariance with (Gamma := ((x0, T1) :: (x, U) :: Gamma)).
      assumption.
      intros z. simpl.
      destruct (beq_id z x0) eqn:Ezx0.
      * reflexivity.
      * destruct (beq_id z x) eqn:Ezx.
        -- reflexivity.
        -- reflexivity.
        
  - (* T_Array *)
    apply T_Array.
    rewrite Forall_map.
    rewrite Forall_forall in *.
    intros t0 Hin.
    apply H0; auto.
    
  - (* T_Map *)
    apply T_Map.
    rewrite Forall_map.
    rewrite Forall_forall in *.
    intros p Hin.
    destruct p as [k val]. simpl.
    apply H0; auto.
Qed.
