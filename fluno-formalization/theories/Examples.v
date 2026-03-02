(* ================================================================= *)
(* Examples.v: Fluxプログラムの具体例                                  *)
(* ================================================================= *)

From Flux Require Import Base Syntax Typing Semantics Soundness.
From Flux Require Import Reactive Effects.

(* ================================================================= *)
(* 例1: Gaussian同士の加算                                            *)
(* ================================================================= *)

Example gaussian_add_example :
  nil ⊢ 
    tm_add (tm_gaussian 10 1) (tm_gaussian 5 2) 
  ∈ TGaussian.
Proof.
  apply T_Add; apply T_Gaussian.
Qed.

Example gaussian_add_steps :
  tm_add (tm_gaussian 10 1) (tm_gaussian 5 2) -->
  tm_gaussian 15 (sqrt 5).
Proof.
  apply ST_AddGaussian.
Qed.

(* ================================================================= *)
(* 例2: サンプリング                                                  *)
(* ================================================================= *)

Example sampling_example :
  nil ⊢ tm_sample (tm_gaussian 10 1) ∈ TFloat.
Proof.
  apply T_Sample. apply T_Gaussian.
Qed.

Example sampling_steps :
  tm_sample (tm_gaussian 10 1) -->
  tm_float (gaussian_sample 10 1).
Proof.
  apply ST_Sample.
Qed.

(* ================================================================= *)
(* 例3: 関数適用                                                      *)
(* ================================================================= *)

Example function_example :
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

Example function_steps :
  tm_app 
    (tm_abs 0 TGaussian (tm_sample (tm_var 0)))
    (tm_gaussian 10 1)
  -->
  tm_sample (tm_gaussian 10 1).
Proof.
  apply ST_AppAbs. apply v_gaussian.
Qed.

(* ================================================================= *)
(* 例4: 多ステップ実行                                                *)
(* ================================================================= *)

Example multi_step_example :
  tm_app 
    (tm_abs 0 TGaussian (tm_sample (tm_var 0)))
    (tm_gaussian 10 1)
  -->*
  tm_float (gaussian_sample 10 1).
Proof.
  eapply multi_step.
  - apply ST_AppAbs. apply v_gaussian.
  - eapply multi_step.
    + apply ST_Sample.
    + apply multi_refl.
Qed.

(* ================================================================= *)
(* 例5: 型安全性の検証                                                *)
(* ================================================================= *)

Example type_safety_example :
  forall t,
  nil ⊢ t ∈ TFloat ->
  t -->* tm_float (gaussian_sample 10 1) ->
  ~ stuck (tm_float (gaussian_sample 10 1)).
Proof.
  intros. eapply soundness; eauto.
Qed.

(* ================================================================= *)
(* 例6: Signal<Gaussian>                                              *)
(* ================================================================= *)

Example signal_gaussian_const : TypedSignal TGaussian :=
  fun t => tv_gaussian 10 1.

Example signal_gaussian_varying : TypedSignal TGaussian :=
  fun t => tv_gaussian (10 + INR t) (1 + INR t / 10).

Example signal_map_example :
  forall sg : TypedSignal TGaussian,
  exists sf : TypedSignal TFloat,
    sf = typed_signal_map 
      (fun g => match g with
                | tv_gaussian m s => tv_float (gaussian_sample m s)
                | _ => tv_float 0
                end)
      sg.
Proof.
  intros. eexists. reflexivity.
Qed.

(* ================================================================= *)
(* 例7: エフェクト付き型付け                                          *)
(* ================================================================= *)

Example pure_function :
  nil ⊢ₑ
    tm_abs 0 TGaussian 
      (tm_add (tm_var 0) (tm_gaussian 5 1))
  ∈ (TFun TGaussian TGaussian @ Pure).
Proof.
  apply ET_Abs.
  apply ET_Add.
  - apply ET_Var. simpl. reflexivity.
  - apply ET_Gaussian.
Qed.

Example impure_function :
  nil ⊢ₑ
    tm_abs 0 TGaussian (tm_sample (tm_var 0))
  ∈ (TFun TGaussian TFloat @ Pure).
Proof.
  apply ET_Abs.
  eapply ET_Sub.
  - apply ET_Sample.
    apply ET_Var. simpl. reflexivity.
  - (* Sample ⊆ Pure は成立しない *)
    admit.
Admitted.
