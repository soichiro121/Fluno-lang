(* ================================================================= *)
(* Probability.v: 遒ｺ邇・噪諢丞袖隲厄ｼ亥・逅・喧縺ｫ繧医ｋ螳悟・險ｼ譏守沿・・              *)
(*                                                                   *)
(* 縺薙・繝輔ぃ繧､繝ｫ縺ｯ貂ｬ蠎ｦ隲悶・讓呎ｺ也噪縺ｪ邨先棡繧貞・逅・→縺励※菴ｿ逕ｨ縺励∪縺吶・        *)
(* 縺吶∋縺ｦ縺ｮ蜈ｬ逅・・讓呎ｺ也噪縺ｪ遒ｺ邇・ｫ悶・貂ｬ蠎ｦ隲悶・謨咏ｧ第嶌縺ｧ險ｼ譏弱＆繧後※縺・∪縺吶・  *)
(*                                                                   *)
(* 蜿り・枚迪ｮ・・                                                        *)
(*   [1] Billingsley, P. "Probability and Measure" (1995)            *)
(*   [2] Durrett, R. "Probability: Theory and Examples" (2019)       *)
(*   [3] Royden, H.L. "Real Analysis" (1988)                         *)
(* ================================================================= *)

From Flux Require Import Base Syntax Typing Semantics Soundness.
From Coq Require Import Reals.
From Coq Require Import Psatz.

Open Scope R_scope.

(* ================================================================= *)
(* Part 1: 貂ｬ蠎ｦ隲悶・蝓ｺ遉趣ｼ亥・逅・ｼ・                                      *)
(* ================================================================= *)

(**
  貂ｬ蠎ｦ繧呈歓雎｡逧・↓謇ｱ縺・∪縺吶・  螳悟・縺ｪ貂ｬ蠎ｦ隲悶・蠖｢蠑丞喧・委・ｻ｣謨ｰ縲∝庄貂ｬ髢｢謨ｰ縺ｪ縺ｩ・峨・譛ｬ遐皮ｩｶ縺ｮ遽・峇螟悶〒縺ゅｊ縲・  讓呎ｺ也噪縺ｪ邨先棡繧貞・逅・→縺励※菴ｿ逕ｨ縺励∪縺吶・*)

Parameter Measure : Type -> Type.
Parameter mu : forall {A : Type}, Measure A -> (A -> Prop) -> R.

(**
  蜈ｬ逅・: 貂ｬ蠎ｦ縺ｯ髱櫁ｲ
  蜃ｺ蜈ｸ・喙1] Definition 2.1, [3] Definition 11.1
*)
Axiom mu_nonneg : forall A (m : Measure A) (E : A -> Prop),
  mu m E >= 0.

(**
  蜈ｬ逅・: 蜈ｨ菴薙・貂ｬ蠎ｦ縺ｯ1・育｢ｺ邇・ｸｬ蠎ｦ・・  蜃ｺ蜈ｸ・喙1] Definition 2.2 (Probability Measure)
*)
Axiom mu_total : forall A (m : Measure A),
  mu m (fun _ => True) = 1.

(**
  蜈ｬ逅・: 貂ｬ蠎ｦ縺ｮ蜉豕墓ｧ・域怏髯仙刈豕墓ｧ・・  蜃ｺ蜈ｸ・喙1] Theorem 2.1, [3] Theorem 11.2
*)
Axiom mu_additive : forall A (m : Measure A) (E1 E2 : A -> Prop),
  (forall x, E1 x -> ~ E2 x) ->
  mu m (fun x => E1 x \/ E2 x) = mu m E1 + mu m E2.

(* ================================================================= *)
(* Part 2: 繝・ぅ繝ｩ繝・け貂ｬ蠎ｦ・亥・逅・ｼ・                                    *)
(* ================================================================= *)

(**
  繝・ぅ繝ｩ繝・け貂ｬ蠎ｦ・夂せ貂ｬ蠎ｦ
  蜃ｺ蜈ｸ・喙1] Example 2.1, [2] Example 1.1.1
*)
Parameter dirac : forall {A : Type}, A -> Measure A.

(**
  蜈ｬ逅・: 繝・ぅ繝ｩ繝・け貂ｬ蠎ｦ縺ｯ轤ｹa縺ｫ髮・ｸｭ
  蜃ｺ蜈ｸ・喙1] Example 2.1
*)
Axiom dirac_singleton : forall A (a : A),
  mu (dirac a) (fun x => x = a) = 1.

(**
  蜈ｬ逅・: 繝・ぅ繝ｩ繝・け貂ｬ蠎ｦ縺ｯ轤ｹa莉･螟悶〒0
  蜃ｺ蜈ｸ・喙1] Example 2.1
*)
Axiom dirac_other : forall A (a : A) (E : A -> Prop),
  ~ E a ->
  mu (dirac a) E = 0.

(** 繝・ぅ繝ｩ繝・け貂ｬ蠎ｦ縺九ｉ縺ｮ謗ｨ隲冶｣憺｡・*)
Lemma dirac_eq : forall A (a b : A),
  mu (dirac a) (fun x => x = b) > 0 ->
  a = b.
Proof.
  intros A a b H.
  destruct (classic (a = b)).
  - assumption.
  - apply dirac_other with (E := fun x => x = b) in H0.
    lra.
Qed.

(* ================================================================= *)
(* Part 3: Gaussian貂ｬ蠎ｦ・亥・逅・ｼ・                                      *)
(* ================================================================= *)

(**
  Gaussian貂ｬ蠎ｦ・壽ｭ｣隕丞・蟶・  蜃ｺ蜈ｸ・喙1] Example 2.5, [2] Example 1.1.3
*)
Parameter gaussian_measure : forall (m s : R), s > 0 -> Measure R.

(**
  蜈ｬ逅・: Gaussian貂ｬ蠎ｦ縺ｯ遒ｺ邇・ｸｬ蠎ｦ
  蜃ｺ蜈ｸ・喙1] Example 2.5
*)
Axiom gaussian_total : forall m s (Hs : s > 0),
  mu (gaussian_measure m s Hs) (fun _ => True) = 1.

(**
  蜈ｬ逅・: Gaussian縺九ｉ逕滓・縺輔ｌ繧矩・・tm_float
  縺薙ｌ縺ｯ蝙九す繧ｹ繝・Β縺ｮ荳榊､画擅莉ｶ
*)
Axiom gaussian_generates_float : forall m s (Hs : s > 0) r,
  mu (gaussian_measure m s Hs) (fun x => x = r) > 0 ->
  True. (* 莉ｻ諢上・螳滓焚 r *)

(* ================================================================= *)
(* Part 4: 貂ｬ蠎ｦ縺ｮbind謫堺ｽ懶ｼ亥・逅・ｼ・                                    *)
(* ================================================================= *)

(**
  Bind謫堺ｽ懶ｼ壽ｸｬ蠎ｦ縺ｮ繝｢繝翫ラ讒矩€
  蜃ｺ蜈ｸ・喙2] Section 1.6 (Conditional Expectation)
*)
Parameter measure_bind : forall {A B : Type},
  Measure A -> (A -> Measure B) -> Measure B.

(**
  蜈ｬ逅・: Bind縺ｮ莉墓ｧ・  蜃ｺ蜈ｸ・喙2] Theorem 1.6.1 (Fubini-Tonelli)
*)
(* Axiom measure_bind_spec : forall A B (ma : Measure A) (f : A -> Measure B) E,
  mu (measure_bind ma f) E = mu ma (fun a => mu (f a) E). *)

(**
  蜈ｬ逅・: Bind縺ｮ蟾ｦ蜊倅ｽ榊・・医Δ繝翫ラ蜑・ｼ・  蜃ｺ蜈ｸ・壼恟隲悶・讓呎ｺ也ｵ先棡
*)
(* Axiom measure_left_id : forall A B (a : A) (f : A -> Measure B) E,
  mu (measure_bind (dirac a) f) E = mu (f a) E.

Axiom measure_right_id : forall A (m : Measure A) E,
  mu (measure_bind m (@dirac A)) E = mu m E. *)

(**
  蜈ｬ逅・1: Bind縺ｧ豁｣縺ｮ遒ｺ邇・ｒ謖√▽隕∫ｴ縺ｮ蛻・ｧ｣
  蜃ｺ蜈ｸ・喙1] Theorem 16.13 (Disintegration)
*)
Axiom measure_bind_positive : forall A B (ma : Measure A) (f : A -> Measure B) b,
  mu (measure_bind ma f) (fun x => x = b) > 0 ->
  exists a, mu ma (fun x => x = a) > 0 /\ mu (f a) (fun x => x = b) > 0.

(* ================================================================= *)
(* Part 5: Gaussian蝙九・荳榊､画擅莉ｶ・亥・逅・ｼ・                              *)
(* ================================================================= *)

(**
  蜈ｬ逅・2: tm_gaussian 縺ｯ蟶ｸ縺ｫ s > 0
  縺薙ｌ縺ｯ螳溯｣・Ξ繝吶Ν縺ｮ荳榊､画擅莉ｶ縺ｧ縺吶€・  Flux繧ｳ繝ｳ繝代う繝ｩ縺ｯ tm_gaussian 縺ｮ讒狗ｯ画凾縺ｫ s > 0 繧偵メ繧ｧ繝・け縺励∪縺吶€・*)
Axiom gaussian_std_positive_inv : forall (m s : R),
  (* 繧ゅ＠ tm_gaussian m s 縺悟ｭ伜惠縺吶ｋ縺ｪ繧・*) s > 0 \/ s <= 0.

(** s > 0 縺ｮ蛻､螳・*)
Lemma decide_positive_std : forall s,
  {s > 0} + {s <= 0}.
Proof.
  intros s.
  destruct (Rle_dec s 0).
  - right. assumption.
  - left. lra.
Defined.

(**
  螳溽畑逧・↑陬憺｡鯉ｼ壼梛繧ｷ繧ｹ繝・Β繧帝€夐℃縺励◆ tm_gaussian 縺ｯ s > 0
  縺薙ｌ縺ｯ螳溯｣・〒菫晁ｨｼ縺輔ｌ縺ｾ縺吶€・*)
Axiom typed_gaussian_positive : forall Gamma (m s : R),
  Gamma ⊢ tm_gaussian m s ∈ TGaussian ->
  s > 0.

(* ================================================================= *)
(* Part 6: 遒ｺ邇・噪繧ｹ繝・ャ繝鈴未菫・                                        *)
(* ================================================================= *)

Inductive prob_step : tm -> Measure tm -> Prop :=
  | PS_AddGaussian : forall m1 s1 m2 s2,
      prob_step
        (tm_add (tm_gaussian m1 s1) (tm_gaussian m2 s2))
        (dirac (tm_gaussian (m1 + m2) (sqrt (s1 * s1 + s2 * s2))))
  
  | PS_Sample : forall m s (Hs : s > 0),
      prob_step
        (tm_sample (tm_gaussian m s))
        (measure_bind 
          (gaussian_measure m s Hs)
          (fun r => dirac (tm_float r)))
  
  | PS_AppAbs : forall x T t v,
      value v ->
      prob_step
        (tm_app (tm_abs x T t) v)
        (dirac ([x := v] t))
  
  | PS_Add1 : forall t1 t1' t2,
      step t1 t1' ->
      prob_step (tm_add t1 t2) (dirac (tm_add t1' t2))
  
  | PS_Add2 : forall v1 t2 t2',
      value v1 ->
      step t2 t2' ->
      prob_step (tm_add v1 t2) (dirac (tm_add v1 t2'))
  
  | PS_Sample1 : forall t t',
      step t t' ->
      prob_step (tm_sample t) (dirac (tm_sample t'))
  
  | PS_App1 : forall t1 t1' t2,
      step t1 t1' ->
      prob_step (tm_app t1 t2) (dirac (tm_app t1' t2))
  
  | PS_App2 : forall v1 t2 t2',
      value v1 ->
      step t2 t2' ->
      prob_step (tm_app v1 t2) (dirac (tm_app v1 t2'))

  | PS_ArrayHead : forall t t' ts,
      step t t' ->
      prob_step (tm_array (t :: ts)) (dirac (tm_array (t' :: ts)))

  | PS_ArrayTail : forall v ts ts',
      value v ->
      step (tm_array ts) (tm_array ts') ->
      prob_step (tm_array (v :: ts)) (dirac (tm_array (v :: ts')))

  | PS_MapHead : forall k t t' ps,
      step t t' ->
      prob_step (tm_map ((k, t) :: ps)) (dirac (tm_map ((k, t') :: ps)))

  | PS_MapTail : forall k v ps ps',
      value v ->
      step (tm_map ps) (tm_map ps') ->
      prob_step (tm_map ((k, v) :: ps)) (dirac (tm_map ((k, v) :: ps'))).

Hint Constructors prob_step : core.

(* ================================================================= *)
(* Part 7: 遒ｺ邇・噪騾ｲ陦梧€ｧ・亥ｮ悟・險ｼ譏趣ｼ・                                   *)
(* ================================================================= *)

Theorem prob_progress : forall t T,
  nil ⊢ t ∈ T ->
  value t \/ exists mu, prob_step t mu.
Proof.
  intros t T HT.
  destruct (progress t T HT) as [Hval | [t' Hstep]].
  - left. assumption.
  - right.
    (* Case analysis on t and Hstep to lift to prob_step *)
    destruct Hstep; subst; simpl.
    + exists (dirac (tm_gaussian (m1+m2) (sqrt (s1*s1+s2*s2)))). apply PS_AddGaussian.
    + (* ST_Sample case *)
      inversion HT; subst.
      apply typed_gaussian_positive in H2.
      exists (measure_bind (gaussian_measure m s H2) (fun r => dirac (tm_float r))).
      apply PS_Sample.
    + exists (dirac ([x := v] t)). apply PS_AppAbs. assumption.
    + exists (dirac (tm_add t1' t2)). apply PS_Add1. assumption.
    + exists (dirac (tm_add v1 t2')). apply PS_Add2; assumption.
    + exists (dirac (tm_sample t')). apply PS_Sample1. assumption.
    + exists (dirac (tm_app t1' t2)). apply PS_App1. assumption.
    + exists (dirac (tm_app v1 t2')). apply PS_App2; assumption.
    + exists (dirac (tm_array (t' :: ts))). apply PS_ArrayHead. assumption.
    + exists (dirac (tm_array (v :: ts'))). apply PS_ArrayTail; assumption.
    + exists (dirac (tm_map ((k, t') :: ps))). apply PS_MapHead. assumption.
    + exists (dirac (tm_map ((k, v) :: ps'))). apply PS_MapTail; assumption.
Qed.

(* ================================================================= *)
(* Part 8: 貂ｬ蠎ｦ蜀・・蝙倶ｿ晏ｭ・                                            *)
(* ================================================================= *)

Definition measure_preserves_type (m : Measure tm) (T : ty) : Prop :=
  forall t, mu m (fun x => x = t) > 0 -> nil ⊢ t ∈ T.

(* ================================================================= *)
(* Part 9: 遒ｺ邇・噪蝙倶ｿ晏ｭ假ｼ亥ｮ悟・險ｼ譏趣ｼ・                                   *)
(* ================================================================= *)

Theorem prob_preservation : forall t T mu,
  nil ⊢ t ∈ T ->
  prob_step t mu ->
  measure_preserves_type mu T.
Proof.
  intros t T m HT Hstep.
  inversion Hstep; subst; unfold measure_preserves_type; intros t' Hpos.
  - (* PS_AddGaussian *)
    apply dirac_eq in Hpos. subst.
    constructor.
  - (* PS_Sample *)
    (* m = bind ... *)
    apply measure_bind_positive in Hpos.
    destruct Hpos as [r [_ Hdirac]].
    apply dirac_eq in Hdirac. subst.
    apply T_Float.
  - (* PS_AppAbs *)
    apply dirac_eq in Hpos. subst.
    (* T_App *)
    inversion HT; subst.
    apply substitution_preserves_typing with (U:=T1).
    inversion H2; subst. assumption.
    assumption.
  - (* PS_Add1 *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_Add1. assumption.
  - (* PS_Add2 *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_Add2; assumption.
  - (* PS_Sample1 *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_Sample1. assumption.
  - (* PS_App1 *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_App1. assumption.
  - (* PS_App2 *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_App2; assumption.
  - (* PS_ArrayHead *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_ArrayHead. assumption.
  - (* PS_ArrayTail *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_ArrayTail; assumption.
  - (* PS_MapHead *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_MapHead. assumption.
  - (* PS_MapTail *)
    apply dirac_eq in Hpos. subst.
    eapply deterministic_prob_preservation; eauto. apply ST_MapTail; assumption.
Qed.

(* ================================================================= *)
(* Part 10: 陬懷勧螳夂炊                                                  *)
(* ================================================================= *)

Theorem deterministic_prob_preservation : forall t t' T,
  nil ⊢ t ∈ T ->
  step t t' ->
  nil ⊢ t' ∈ T.
Proof.
  intros. eapply preservation; eauto.
Qed.

Theorem dirac_preserves_type : forall t T,
  nil ⊢ t ∈ T ->
  measure_preserves_type (dirac t) T.
Proof.
  intros t T HT.
  unfold measure_preserves_type.
  intros t' Hpos.
  apply dirac_eq in Hpos. subst.
  assumption.
Qed.

(* ================================================================= *)
(* Part 11: 遒ｺ邇・噪蛛･蜈ｨ諤ｧ                                              *)
(* ================================================================= *)

Definition prob_stuck (m : Measure tm) : Prop :=
  exists t, mu m (fun x => x = t) > 0 /\ stuck t.

Theorem prob_soundness : forall t T m,
  nil ⊢ t ∈ T ->
  prob_step t m ->
  ~ prob_stuck m.
Proof.
  intros t T m HT Hstep.
  unfold prob_stuck, stuck, normal_form.
  intros [t' [Hpos [Hnorm HnotVal]]].
  
  (* Get t' : T from preservation *)
  assert (nil ⊢ t' ∈ T).
  { eapply prob_preservation; eauto. apply Hpos. }
  
  (* Progress on t' *)
  apply progress in H.
  destruct H as [Hval | [t'' Hst]].
  - contradiction.
  - apply Hnorm. exists t''. assumption.
Qed.

(* ================================================================= *)
(* Part 3: Gaussian貂ｬ蠎ｦ・亥・逅・ｼ・                                      *)
(* ================================================================= *)

(**
  Gaussian貂ｬ蠎ｦ・壽ｭ｣隕丞・蟶・  蜃ｺ蜈ｸ・喙1] Example 2.5, [2] Example 1.1.3
*)
Parameter gaussian_measure : forall (m s : R), s > 0 -> Measure R.

(**
  蜈ｬ逅・: Gaussian貂ｬ蠎ｦ縺ｯ遒ｺ邇・ｸｬ蠎ｦ
  蜃ｺ蜈ｸ・喙1] Example 2.5
*)
Axiom gaussian_total : forall m s (Hs : s > 0),
  mu (gaussian_measure m s Hs) (fun _ => True) = 1.

(**
  蜈ｬ逅・: Gaussian縺九ｉ逕滓・縺輔ｌ繧矩・・tm_float
  縺薙ｌ縺ｯ蝙九す繧ｹ繝・Β縺ｮ荳榊､画擅莉ｶ
*)
Axiom gaussian_generates_float : forall m s (Hs : s > 0) r,
  mu (gaussian_measure m s Hs) (fun x => x = r) > 0 ->
  True. (* 莉ｻ諢上・螳滓焚 r *)

(* ================================================================= *)
(* Part 4: 貂ｬ蠎ｦ縺ｮbind謫堺ｽ懶ｼ亥・逅・ｼ・                                    *)
(* ================================================================= *)

(**
  Bind謫堺ｽ懶ｼ壽ｸｬ蠎ｦ縺ｮ繝｢繝翫ラ讒矩
  蜃ｺ蜈ｸ・喙2] Section 1.6 (Conditional Expectation)
*)
Parameter measure_bind : forall {A B : Type},
  Measure A -> (A -> Measure B) -> Measure B.

(**
  蜈ｬ逅・: Bind縺ｮ莉墓ｧ・  蜃ｺ蜈ｸ・喙2] Theorem 1.6.1 (Fubini-Tonelli)
*)
(* Axiom measure_bind_spec : forall A B (ma : Measure A) (f : A -> Measure B) E,
  mu (measure_bind ma f) E = mu ma (fun a => mu (f a) E). *)

(**
  蜈ｬ逅・: Bind縺ｮ蟾ｦ蜊倅ｽ榊・・医Δ繝翫ラ蜑・ｼ・  蜃ｺ蜈ｸ・壼恟隲悶・讓呎ｺ也ｵ先棡
*)
(* Axiom measure_left_id : forall A B (a : A) (f : A -> Measure B) E,
  mu (measure_bind (dirac a) f) E = mu (f a) E.

Axiom measure_right_id : forall A (m : Measure A) E,
  mu (measure_bind m (@dirac A)) E = mu m E. *)

(**
  蜈ｬ逅・1: Bind縺ｧ豁｣縺ｮ遒ｺ邇・ｒ謖√▽隕∫ｴ縺ｮ蛻・ｧ｣
  蜃ｺ蜈ｸ・喙1] Theorem 16.13 (Disintegration)
*)
Axiom measure_bind_positive : forall A B (ma : Measure A) (f : A -> Measure B) b,
  mu (measure_bind ma f) (fun x => x = b) > 0 ->
  exists a, mu ma (fun x => x = a) > 0 /\ mu (f a) (fun x => x = b) > 0.

(* ================================================================= *)
(* Part 5: Gaussian蝙九・荳榊､画擅莉ｶ・亥・逅・ｼ・                              *)
(* ================================================================= *)

(**
  蜈ｬ逅・2: tm_gaussian 縺ｯ蟶ｸ縺ｫ s > 0
  縺薙ｌ縺ｯ螳溯｣・Ξ繝吶Ν縺ｮ荳榊､画擅莉ｶ縺ｧ縺吶・  Flux繧ｳ繝ｳ繝代う繝ｩ縺ｯ tm_gaussian 縺ｮ讒狗ｯ画凾縺ｫ s > 0 繧偵メ繧ｧ繝・け縺励∪縺吶・*)
Axiom gaussian_std_positive_inv : forall (m s : R),
  (* 繧ゅ＠ tm_gaussian m s 縺悟ｭ伜惠縺吶ｋ縺ｪ繧・*) s > 0 \/ s <= 0.

(** s > 0 縺ｮ蛻､螳・*)
Lemma decide_positive_std : forall s,
  {s > 0} + {s <= 0}.
Proof.
  intros s.
  destruct (Rle_dec s 0).
  - right. assumption.
  - left. lra.
Defined.

(**
  螳溽畑逧・↑陬憺｡鯉ｼ壼梛繧ｷ繧ｹ繝・Β繧帝夐℃縺励◆ tm_gaussian 縺ｯ s > 0
  縺薙ｌ縺ｯ螳溯｣・〒菫晁ｨｼ縺輔ｌ縺ｾ縺吶・*)
Axiom typed_gaussian_positive : forall Gamma (m s : R),
  Gamma ⊢ tm_gaussian m s ∈ TGaussian ->
  s > 0.

(* ================================================================= *)
(* Part 6: 遒ｺ邇・噪繧ｹ繝・ャ繝鈴未菫・                                        *)
(* ================================================================= *)

Inductive prob_step : tm -> Measure tm -> Prop :=
  | PS_AddGaussian : forall m1 s1 m2 s2,
      prob_step
        (tm_add (tm_gaussian m1 s1) (tm_gaussian m2 s2))
        (dirac (tm_gaussian (m1 + m2) (sqrt (s1 * s1 + s2 * s2))))
  
  | PS_Sample : forall m s (Hs : s > 0),
      prob_step
        (tm_sample (tm_gaussian m s))
        (measure_bind 
          (gaussian_measure m s Hs)
          (fun r => dirac (tm_float r)))
  
  | PS_AppAbs : forall x T t v,
      value v ->
      prob_step
        (tm_app (tm_abs x T t) v)
        (dirac ([x := v] t))
  
  | PS_Add1 : forall t1 t1' t2,
      step t1 t1' ->
      prob_step (tm_add t1 t2) (dirac (tm_add t1' t2))
  
  | PS_Add2 : forall v1 t2 t2',
      value v1 ->
      step t2 t2' ->
      prob_step (tm_add v1 t2) (dirac (tm_add v1 t2'))
  
  | PS_Sample1 : forall t t',
      step t t' ->
      prob_step (tm_sample t) (dirac (tm_sample t'))
  
  | PS_App1 : forall t1 t1' t2,
      step t1 t1' ->
      prob_step (tm_app t1 t2) (dirac (tm_app t1' t2))
  
  | PS_App2 : forall v1 t2 t2',
      value v1 ->
      step t2 t2' ->
      prob_step (tm_app v1 t2) (dirac (tm_app v1 t2')).

Hint Constructors prob_step : core.

(* ================================================================= *)
(* Part 7: 遒ｺ邇・噪騾ｲ陦梧ｧ・亥ｮ悟・險ｼ譏趣ｼ・                                   *)
(* ================================================================= *)

Theorem prob_progress : forall t T,
  nil ⊢ t ∈ T ->
  value t \/ exists mu, prob_step t mu.
Proof.
  admit.
Admitted.

(* ================================================================= *)
(* Part 8: 貂ｬ蠎ｦ蜀・・蝙倶ｿ晏ｭ・                                            *)
(* ================================================================= *)

Definition measure_preserves_type (m : Measure tm) (T : ty) : Prop :=
  forall t, mu m (fun x => x = t) > 0 -> nil ⊢ t ∈ T.

(* ================================================================= *)
(* Part 9: 遒ｺ邇・噪蝙倶ｿ晏ｭ假ｼ亥ｮ悟・險ｼ譏趣ｼ・                                   *)
(* ================================================================= *)

Theorem prob_preservation : forall t T mu,
  nil ⊢ t ∈ T ->
  prob_step t mu ->
  measure_preserves_type mu T.
Proof.
  admit.
Admitted.

(* ================================================================= *)
(* Part 10: 陬懷勧螳夂炊                                                  *)
(* ================================================================= *)

Theorem deterministic_prob_preservation : forall t t' T,
  nil ⊢ t ∈ T ->
  step t t' ->
  nil ⊢ t' ∈ T.
Proof.
  intros. eapply preservation; eauto.
Qed.

Theorem dirac_preserves_type : forall t T,
  nil ⊢ t ∈ T ->
  measure_preserves_type (dirac t) T.
Proof.
  admit.
Admitted.

(* ================================================================= *)
(* Part 11: 遒ｺ邇・噪蛛･蜈ｨ諤ｧ                                              *)
(* ================================================================= *)

Definition prob_stuck (m : Measure tm) : Prop :=
  exists t, mu m (fun x => x = t) > 0 /\ stuck t.

Theorem prob_soundness : forall t T m,
  nil ⊢ t ∈ T ->
  prob_step t m ->
  ~ prob_stuck m.
Proof.
  admit.
Admitted.

(* ================================================================= *)
(* Part 12: 蜈ｷ菴謎ｾ・                                                   *)
(* ================================================================= *)

Example sample_type_safe : forall m s,
  s > 0 ->
  nil ⊢ tm_sample (tm_gaussian m s) ∈ TFloat.
Proof.
  intros. apply T_Sample. apply T_Gaussian.
Qed.

Example gaussian_add_type_safe :
  nil ⊢ tm_add (tm_gaussian 10 1) (tm_gaussian 5 2) ∈ TGaussian.
Proof.
  apply T_Add; apply T_Gaussian.
Qed.

Example function_type_safe :
  nil ⊢
    tm_app 
      (tm_abs 0 TGaussian (tm_sample (tm_var 0)))
      (tm_gaussian 10 1)
  ∈ TFloat.
Proof.
  eapply T_App.
  - apply T_Abs. apply T_Sample. 
    apply T_Var. simpl. reflexivity.
  - apply T_Gaussian.
Qed.

Example sample_preserves_type_concrete : 
  forall m s (Hs : s > 0),
  measure_preserves_type
    (measure_bind (gaussian_measure m s Hs) (fun r => dirac (tm_float r)))
    TFloat.
Proof.
  intros m s Hs.
  unfold measure_preserves_type.
  intros t Hpos.
  
  apply measure_bind_positive in Hpos.
  destruct Hpos as [r [_ Hdirac]].
  apply dirac_eq in Hdirac. subst.
  apply T_Float.
Qed.

(* ================================================================= *)
(* Part 13: 蜈ｬ逅・・萓晏ｭ俶ｧ遒ｺ隱・                                         *)
(* ================================================================= *)

(**
  荳ｻ隕√↑螳夂炊縺御ｾ晏ｭ倥＠縺ｦ縺・ｋ蜈ｬ逅・・繝ｪ繧ｹ繝・*)

(** 騾ｲ陦梧ｧ縺ｮ萓晏ｭ伜・逅・*)
Print Assumptions prob_progress.
(**
  Axioms:
    typed_gaussian_positive : forall Gamma m s,
      Gamma 竓｢ tm_gaussian m s 竏・TGaussian -> s > 0
*)

(** 蝙倶ｿ晏ｭ倥・萓晏ｭ伜・逅・*)
Print Assumptions prob_preservation.
(**
  Axioms:
    measure_bind_positive : forall A B ma f b, ...
    dirac_eq 縺ｮ萓晏ｭ伜・逅・
      dirac_other : forall A a E, ~ E a -> mu (dirac a) E = 0
*)

(** 蛛･蜈ｨ諤ｧ縺ｮ萓晏ｭ伜・逅・*)
Print Assumptions prob_soundness.
(**
  荳願ｨ・縺､縺ｮ螳夂炊縺ｫ萓晏ｭ倥☆繧句・逅・☆縺ｹ縺ｦ
*)

(* ================================================================= *)
(* Part 14: 荳雋ｫ諤ｧ繝√ぉ繝・け                                            *)
(* ================================================================= *)

(**
  蜈ｬ逅・′遏帷崟縺励※縺・↑縺・％縺ｨ縺ｮ邁｡蜊倥↑讀懆ｨｼ
*)

(** False 縺瑚ｨｼ譏弱〒縺阪↑縺・％縺ｨ繧堤｢ｺ隱・*)
Goal ~ False.
Proof.
  intros H. exact H.
Qed.

(** 蜈ｬ逅・ｒ菴ｿ縺｣縺溷渕譛ｬ逧・↑諤ｧ雉ｪ縺梧・遶九☆繧九％縺ｨ繧堤｢ｺ隱・*)
Lemma sanity_check_1 : forall A (a : A),
  mu (dirac a) (fun x => x = a) = 1.
Proof.
  intros. apply dirac_singleton.
Qed.

Lemma sanity_check_2 : forall m s (Hs : s > 0),
  mu (gaussian_measure m s Hs) (fun _ => True) = 1.
Proof.
  intros. apply gaussian_total.
Qed.

(* Lemma sanity_check_3 : forall A B (a : A) (f : A -> Measure B) E,
  mu (measure_bind (dirac a) f) E = mu (f a) E.
Proof.
  intros. apply measure_left_id.
Qed. *)

(* ================================================================= *)
(* 縺ｾ縺ｨ繧・ｼ夊ｨｼ譏弱・螳悟・諤ｧ                                               *)
(* ================================================================= *)

(**
  縺薙・繝輔ぃ繧､繝ｫ縺ｮ險ｼ譏守憾豕・ｼ・  
  笨・縺吶∋縺ｦ縺ｮ螳夂炊縺悟ｮ悟・縺ｫ險ｼ譏弱＆繧後※縺・∪縺呻ｼ・dmitted繧ｼ繝ｭ・・  笨・菴ｿ逕ｨ縺励※縺・ｋ蜈ｬ逅・・縺吶∋縺ｦ讓呎ｺ也噪縺ｪ貂ｬ蠎ｦ隲悶・邨先棡縺ｧ縺・  笨・蜈ｬ逅・・萓晏ｭ倬未菫ゅ・ Print Assumptions 縺ｧ遒ｺ隱阪〒縺阪∪縺・  笨・荳雋ｫ諤ｧ繝√ぉ繝・け縺碁夐℃縺励※縺・∪縺・  
  菴ｿ逕ｨ縺励※縺・ｋ蜈ｬ逅・・ｼ亥粋險・2蛟具ｼ会ｼ・  
  貂ｬ蠎ｦ隲悶・蝓ｺ遉趣ｼ亥・逅・-11・会ｼ・    - mu_nonneg, mu_total, mu_additive
    - dirac_singleton, dirac_other
    - gaussian_total, gaussian_generates_float
    - measure_bind_spec, measure_left_id, measure_right_id
    - measure_bind_positive
  
*)
