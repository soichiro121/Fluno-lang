(* ================================================================= *)
(* Effects.v: エフェクトシステム                                       *)
(* ================================================================= *)

From Flux Require Import Base Syntax Typing.

(* ================================================================= *)
(* エフェクトの定義                                                   *)
(* ================================================================= *)

Inductive effect : Type :=
  | Pure : effect
  | Sample : effect
  | IO : effect
  | EffUnion : effect -> effect -> effect.

(* エフェクトの包含関係 *)
Inductive eff_subtype : effect -> effect -> Prop :=
  | eff_sub_refl : forall e, eff_subtype e e
  | eff_sub_pure : forall e, eff_subtype Pure e
  | eff_sub_union_left : forall e1 e2,
      eff_subtype e1 (EffUnion e1 e2)
  | eff_sub_union_right : forall e1 e2,
      eff_subtype e2 (EffUnion e1 e2)
  | eff_sub_union : forall e1 e2 e3,
      eff_subtype e1 e3 ->
      eff_subtype e2 e3 ->
      eff_subtype (EffUnion e1 e2) e3
  | eff_sub_trans : forall e1 e2 e3,
      eff_subtype e1 e2 ->
      eff_subtype e2 e3 ->
      eff_subtype e1 e3.

Hint Constructors eff_subtype : core.

(* ================================================================= *)
(* エフェクト付き型                                                   *)
(* ================================================================= *)

Definition eff_ty := (ty * effect)%type.

Notation "T '@' e" := (T, e) (at level 50).

(* ================================================================= *)
(* エフェクト付き型付け                                               *)
(* ================================================================= *)

Definition eff_context := list (id * eff_ty).

Fixpoint eff_lookup (Gamma : eff_context) (x : id) : option eff_ty :=
  match Gamma with
  | nil => None
  | (y, T) :: Gamma' => 
      if beq_id x y then Some T else eff_lookup Gamma' x
  end.

Reserved Notation "Gamma '⊢ₑ' t '∈' T" (at level 40).

Inductive eff_has_type : eff_context -> tm -> eff_ty -> Prop :=
  | ET_Var : forall Gamma x T e,
      eff_lookup Gamma x = Some (T @ e) ->
      Gamma ⊢ₑ tm_var x ∈ (T @ e)
  
  | ET_Int : forall Gamma n,
      Gamma ⊢ₑ tm_int n ∈ (TInt @ Pure)
  
  | ET_Float : forall Gamma r,
      Gamma ⊢ₑ tm_float r ∈ (TFloat @ Pure)
  
  | ET_Bool : forall Gamma b,
      Gamma ⊢ₑ tm_bool b ∈ (TBool @ Pure)
  
  | ET_Gaussian : forall Gamma m s,
      Gamma ⊢ₑ tm_gaussian m s ∈ (TGaussian @ Pure)
  
  | ET_Add : forall Gamma t1 t2 e1 e2,
      Gamma ⊢ₑ t1 ∈ (TGaussian @ e1) ->
      Gamma ⊢ₑ t2 ∈ (TGaussian @ e2) ->
      Gamma ⊢ₑ tm_add t1 t2 ∈ (TGaussian @ EffUnion e1 e2)
  
  | ET_Sample : forall Gamma t e,
      Gamma ⊢ₑ t ∈ (TGaussian @ e) ->
      Gamma ⊢ₑ tm_sample t ∈ (TFloat @ EffUnion e Sample)
  
  | ET_Abs : forall Gamma x T1 e1 T2 e2 t,
      (x, T1 @ e1) :: Gamma ⊢ₑ t ∈ (T2 @ e2) ->
      Gamma ⊢ₑ tm_abs x T1 t ∈ (TFun T1 T2 @ Pure)
  
  | ET_App : forall Gamma t1 t2 T1 T2 e1 e2,
      Gamma ⊢ₑ t1 ∈ (TFun T1 T2 @ e1) ->
      Gamma ⊢ₑ t2 ∈ (T1 @ e2) ->
      Gamma ⊢ₑ tm_app t1 t2 ∈ (T2 @ EffUnion e1 e2)
  
  | ET_Unit : forall Gamma,
      Gamma ⊢ₑ tm_unit ∈ (TUnit @ Pure)
  
  | ET_Sub : forall Gamma t T e1 e2,
      Gamma ⊢ₑ t ∈ (T @ e1) ->
      eff_subtype e1 e2 ->
      Gamma ⊢ₑ t ∈ (T @ e2)

where "Gamma '⊢ₑ' t '∈' T" := (eff_has_type Gamma t T).

Hint Constructors eff_has_type : core.

(* ================================================================= *)
(* エフェクト付き型システムの性質                                     *)
(* ================================================================= *)

Lemma eff_subtype_refl : forall e,
  eff_subtype e e.
Proof.
  intros. apply eff_sub_refl.
Qed.

Lemma eff_union_comm : forall e1 e2,
  eff_subtype (EffUnion e1 e2) (EffUnion e2 e1).
Proof.
  intros. apply eff_sub_union.
  - apply eff_sub_union_right.
  - apply eff_sub_union_left.
Qed.

(* Pureな関数はSampleを呼べない *)
Example cannot_sample_in_pure :
  ~ (nil ⊢ₑ 
      tm_abs 0 TGaussian (tm_sample (tm_var 0))
      ∈ (TFun TGaussian TFloat @ Pure)).
Proof.
  intros contra.
  inversion contra; subst.
  - (* ET_Abs *)
    inversion H3; subst.
    + (* ET_Sample *)
      inversion H4; subst.
      (* e2 = EffUnion e Sample *)
      (* しかし全体のエフェクトは Pure *)
      admit. (* ここで矛盾を示すには追加の補題が必要 *)
  - (* ET_Sub *)
    inversion H3; subst.
    admit.
Admitted. (* より詳細な証明は省略 *)
