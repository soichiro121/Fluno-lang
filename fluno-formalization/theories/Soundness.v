(* ================================================================= *)
(* Soundness.v: 健全性の証明                                           *)
(* ================================================================= *)

From Flux Require Import Base Syntax Typing Semantics.

(* ================================================================= *)
(* 正準形補題                                                         *)
(* ================================================================= *)

Lemma canonical_forms_int : forall t,
  nil ⊢ t ∈ TInt ->
  value t ->
  exists n, t = tm_int n.
Proof.
  intros t HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists n. reflexivity.
Qed.

Lemma canonical_forms_float : forall t,
  nil ⊢ t ∈ TFloat ->
  value t ->
  exists r, t = tm_float r.
Proof.
  intros t HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists r. reflexivity.
Qed.

Lemma canonical_forms_bool : forall t,
  nil ⊢ t ∈ TBool ->
  value t ->
  exists b, t = tm_bool b.
Proof.
  intros t HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists b. reflexivity.
Qed.

Lemma canonical_forms_gaussian : forall t,
  nil ⊢ t ∈ TGaussian ->
  value t ->
  exists m s, t = tm_gaussian m s.
Proof.
  intros t HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists m, s. reflexivity.
Qed.

Lemma canonical_forms_fun : forall t T1 T2,
  nil ⊢ t ∈ TFun T1 T2 ->
  value t ->
  exists x body, t = tm_abs x T1 body.
Proof.
  intros t T1 T2 HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists x, t0. reflexivity.
Qed.

Lemma canonical_forms_unit : forall t,
  nil ⊢ t ∈ TUnit ->
  value t ->
  t = tm_unit.
Proof.
  intros t HT HVal.
  inversion HVal; subst; inversion HT; subst.
  reflexivity.
Qed.

Lemma canonical_forms_array : forall t T,
  nil ⊢ t ∈ TArray T ->
  value t ->
  exists vs, t = tm_array vs.
Proof.
  intros t T HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists vs. reflexivity.
Qed.

Lemma canonical_forms_map : forall t T,
  nil ⊢ t ∈ TMap T ->
  value t ->
  exists ps, t = tm_map ps.
Proof.
  intros t T HT HVal.
  inversion HVal; subst; inversion HT; subst.
  exists ps. reflexivity.
Qed.

(* ================================================================= *)
(* 進行性（Progress）                                                 *)
(* ================================================================= *)

Lemma array_progress_lemma : forall vs,
  Forall (fun t => value t \/ exists t', t --> t') vs ->
  value (tm_array vs) \/ exists t', tm_array vs --> t'.
Proof.
  intros vs H.
  induction vs.
  - left. apply v_array. apply Forall_nil.
  - inversion H; subst.
    apply IHvs in H3.
    destruct H2 as [Hval | [t' Hstep]].
    + destruct H3 as [HvalVs | [ts' HstepVs]].
      * (* x val, xs val *)
        left. inversion HvalVs; subst.
        apply v_array. apply Forall_cons; auto.
      * (* x val, xs step *)
         inversion HstepVs; subst.
         (* HstepVs says tm_array vs --> tm_array ts' ? No, exists t', tm_array vs --> t'. *)
         (* We need to know t' is tm_array ts' for ST_ArrayTail. *)
         (* ST_Array* are the ONLY rules for tm_array. *)
         (* So if tm_array vs steps, it MUST step to tm_array something. *)
         (* We should use that fact. *)
         right. exists (tm_array (a :: ts')). (* a is v *)
         apply ST_ArrayTail; auto.
         (* Wait, HstepVs gives exists t', tm_array vs --> t'. *)
         (* ST_ArrayTail needs specific shape? No, `tm_array ts --> tm_array ts'`. *)
         (* My lemma gives `exists t', tm_array vs --> t'`. *)
         (* I need to show `exists ts', t' = tm_array ts'`. *)
         (* This is inversion on step. *)
         (* ST_ArrayHead, ST_ArrayTail both produce tm_array. *)
         (* So yes. *)
         (* Let's verify this inversion. *)
         inversion H0; subst.
         (* ST_ArrayHead *)
         exists (tm_array (v :: ts)). apply ST_ArrayTail; auto. apply ST_ArrayHead. assumption.
         (* ST_ArrayTail *)
         exists (tm_array (v :: ts')). apply ST_ArrayTail; auto. apply ST_ArrayTail; auto.
    + (* x step *)
      right. exists (tm_array (t' :: vs)).
      apply ST_ArrayHead. assumption.
Qed.

(* Helper to handle the shape of array step result for the lemma above *)
(* Actually, I can just do inversion inline. *)

Lemma map_progress_lemma : forall ps,
  Forall (fun p => value (snd p) \/ exists t', snd p --> t') ps ->
  value (tm_map ps) \/ exists t', tm_map ps --> t'.
Proof.
  intros ps H.
  induction ps.
  - left. apply v_map. apply Forall_nil.
  - destruct a as [k v].
    inversion H; subst.
    (* Forall condition is on (snd a). a=(k,v). snd a = v. *)
    simpl in H2.
    apply IHps in H3.
    destruct H2 as [Hval | [t' Hstep]].
    + destruct H3 as [HvalPs | [ps_tm' HstepPs]].
      * left. inversion HvalPs; subst.
        apply v_map. apply Forall_cons; auto. simpl. assumption.
      * right.
        inversion HstepPs; subst.
        (* Inversion on step tm_map ps --> ... *)
        (* ST_MapHead, ST_MapTail *)
        exists (tm_map ((k, v) :: ps0)).
        apply ST_MapTail; auto. apply ST_MapHead. assumption.
        exists (tm_map ((k, v) :: ps'0)).
        apply ST_MapTail; auto. apply ST_MapTail; auto.
    + right. exists (tm_map ((k, t') :: ps)).
      apply ST_MapHead. assumption.
Qed.

Theorem progress : forall t T,
  nil ⊢ t ∈ T ->
  value t \/ exists t', t --> t'.
Proof.
  intros t T H.
  remember nil as Gamma.
  induction t using tm_ind_custom; auto; intros; subst.
  - (* tm_var *)
    inversion H.
  - (* tm_int *)
    left. apply v_int.
  - (* tm_float *)
    left. apply v_float.
  - (* tm_bool *)
    left. apply v_bool.
  - (* tm_gaussian *)
    left. apply v_gaussian.
  - (* tm_add *)
    right. inversion H; subst.
    apply IHt1 in H3. apply IHt2 in H5.
    destruct H3 as [Hv1 | [t1' Hst1]]; destruct H5 as [Hv2 | [t2' Hst2]]; auto.
    + (* v1, v2 *)
      apply canonical_forms_gaussian in H3; auto.
      apply canonical_forms_gaussian in H5; auto.
      destruct H3 as [m1 [s1 E1]]. destruct H5 as [m2 [s2 E2]]. subst.
      exists (tm_gaussian (m1+m2) (sqrt (s1*s1+s2*s2))). apply ST_AddGaussian.
    + exists (tm_add t1 t2'). apply ST_Add2; auto.
    + exists (tm_add t1' t2). apply ST_Add1; auto.
    + exists (tm_add t1' t2). apply ST_Add1; auto.
    + reflexivity. 
    + reflexivity.
  - (* tm_sample *)
    inversion H; subst.
    apply IHt in H2; auto.
    destruct H2 as [Hv | [t' Hst]].
    + (* value *)
      apply canonical_forms_gaussian in H2; auto.
      destruct H2 as [m [s E]]. subst.
      right. exists (tm_float (gaussian_sample m s)). apply ST_Sample.
    + right. exists (tm_sample t'). apply ST_Sample1. assumption.
    + reflexivity.
  - (* tm_abs *)
    left. apply v_abs.
  - (* tm_app *)
    right. inversion H; subst.
    apply IHt1 in H3. apply IHt2 in H5.
    destruct H3 as [Hv1 | [t1' Hst1]].
    + (* t1 val *)
      apply canonical_forms_fun in H3; auto.
      destruct H3 as [x [body E]]. subst.
      destruct H5 as [Hv2 | [t2' Hst2]].
      * exists ([x:=t2]body). apply ST_AppAbs. assumption.
      * exists (tm_app (tm_abs x T1 body) t2'). apply ST_App2; auto.
    + exists (tm_app t1' t2). apply ST_App1; auto.
    + reflexivity.
    + reflexivity.
  - (* tm_unit *)
    left. apply v_unit.
  - (* tm_string *)
    left. apply v_string.
  - (* tm_array *)
    inversion H; subst.
    apply array_progress_lemma.
    rewrite Forall_forall in *.
    intros t0 Hin.
    apply H0. assumption.
    apply H4. assumption. reflexivity.
  - (* tm_map *)
    inversion H; subst.
    apply map_progress_lemma.
    rewrite Forall_forall in *.
    intros p Hin.
    apply HMap. assumption.
    apply H4. assumption. reflexivity.
Qed.

(* ================================================================= *)
(* 型保存（Preservation）                                             *)
(* ================================================================= *)

Theorem preservation : forall t t' T,
  nil ⊢ t ∈ T ->
  t --> t' ->
  nil ⊢ t' ∈ T.
Proof.
  intros t t' T HT Hstep.
  generalize dependent T.
  induction Hstep; intros T HT; inversion HT; subst; auto.
  - (* ST_AddGaussian *)
    constructor.
  - (* ST_Sample *)
    constructor.
  - (* ST_AppAbs *)
    (* HT: nil ⊢ app (abs x T1 t) v ∈ T *)
    (* H2: nil ⊢ abs x T1 t ∈ TFun T1 T *)
    (* H4: nil ⊢ v ∈ T1 *)
    inversion H2; subst.
    (* H7: ((x,T1)::nil) ⊢ t ∈ T *)
    apply substitution_preserves_typing with (U:=T1); assumption.
  - (* ST_Add1 *)
    constructor; auto.
  - (* ST_Add2 *)
    constructor; auto.
  - (* ST_Sample1 *)
    constructor. apply IHHstep. assumption.
  - (* ST_App1 *)
    eapply T_App; eauto. apply IHHstep; assumption.
  - (* ST_App2 *)
    eapply T_App; eauto. apply IHHstep; assumption.
  - (* ST_ArrayHead *)
    apply T_Array.
    rewrite Forall_forall in *.
    intros x Hin.
    inversion H1; subst.
    (* Forall (has_type) (t::ts) *)
    (* -> has_type t AND Forall ts *)
    (* If x = t', use IH. If x in ts, use H *)
    (* Since t :: ts --> t' :: ts, result is t' :: ts. *)
    (* We need Forall has_type (t' :: ts). *)
    apply Forall_cons.
    + apply IHHstep.
       (* Extract typ of t *)
       inversion H2; subst. assumption.
    + inversion H2; subst. assumption.
  - (* ST_ArrayTail *)
    apply T_Array.
    apply Forall_cons.
    + inversion H2; subst. assumption.
    + (* Forall has_type ts' *)
      (* IH says: if nil ⊢ array ts ∈ TArray T, then nil ⊢ array ts' ∈ TArray T. *)
      (* We have nil ⊢ array ts ∈ TArray T from H2 inversion. *)
      assert (nil ⊢ tm_array ts ∈ TArray T).
      { apply T_Array. inversion H2; subst. assumption. }
      apply IHHstep in H0.
      inversion H0; subst.
      assumption.
  - (* ST_MapHead *)
    apply T_Map.
    apply Forall_cons.
    + simpl. apply IHHstep.
      inversion H2; subst.
      (* Forall (snd p : T) ((k,t)::ps) *)
      (* -> snd (k,t) : T -> t : T *)
      inversion H3; subst. simpl in H6. assumption.
    + inversion H2; subst. inversion H3; subst. assumption.
  - (* ST_MapTail *)
    apply T_Map.
    apply Forall_cons.
    + inversion H2; subst. inversion H3; subst. assumption.
    + assert (nil ⊢ tm_map ps ∈ TMap T).
      { apply T_Map. inversion H2; subst. inversion H3; subst. assumption. }
      apply IHHstep in H0.
      inversion H0; subst. assumption.
Qed.

(* ================================================================= *)
(* 健全性（Soundness）                                                *)
(* ================================================================= *)

Definition normal_form {X : Type} (R : X -> X -> Prop) (t : X) : Prop :=
  ~ exists t', R t t'.

Definition stuck (t : tm) : Prop :=
  normal_form step t /\ ~ value t.

Corollary soundness : forall t t' T,
  nil ⊢ t ∈ T ->
  t -->* t' ->
  ~(stuck t').
Proof.
  intros t t' T HT Hmulti.
  unfold stuck, normal_form.
  intros [Hnorm HnotVal].
  induction Hmulti.
  - (* t = t' *)
    apply progress in HT.
    destruct HT; contradiction.
  - (* t --> y -->* t' *)
    apply IHHmulti; auto.
    eapply preservation; eauto.
Qed.

(* ================================================================= *)
(* 決定性（オプション）                                                *)
(* ================================================================= *)

Theorem step_deterministic : forall t t' t'',
  t --> t' ->
  t --> t'' ->
  t' = t''.
Proof.
  admit.
Admitted.
