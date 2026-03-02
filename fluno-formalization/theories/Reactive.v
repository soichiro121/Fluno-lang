(* ================================================================= *)
(* Reactive.v: リアクティブ型の定義と性質                              *)
(* ================================================================= *)

From Flux Require Import Base Syntax Typing.
From Coq Require Import Logic.FunctionalExtensionality.

(* ================================================================= *)
(* Signal型                                                           *)
(* ================================================================= *)

Definition Signal (A : Type) := Time -> A.

(* ================================================================= *)
(* Signalの基本操作                                                   *)
(* ================================================================= *)

Definition signal_const {A : Type} (a : A) : Signal A :=
  fun _ => a.

Definition signal_map {A B : Type} 
  (f : A -> B) (s : Signal A) : Signal B :=
  fun t => f (s t).

Definition signal_combine {A B C : Type} 
  (f : A -> B -> C) (s1 : Signal A) (s2 : Signal B) : Signal C :=
  fun t => f (s1 t) (s2 t).

(* ================================================================= *)
(* Signalの性質                                                       *)
(* ================================================================= *)

Lemma signal_map_compose : 
  forall A B C (f : A -> B) (g : B -> C) (s : Signal A),
  signal_map g (signal_map f s) = signal_map (fun x => g (f x)) s.
Proof.
  intros. apply functional_extensionality.
  intros t. unfold signal_map. reflexivity.
Qed.

Lemma signal_map_id : 
  forall A (s : Signal A),
  signal_map (fun x => x) s = s.
Proof.
  intros. apply functional_extensionality.
  intros t. unfold signal_map. reflexivity.
Qed.

Lemma signal_combine_comm : 
  forall A B (f : A -> A -> B) (s1 s2 : Signal A),
  (forall x y, f x y = f y x) ->
  signal_combine f s1 s2 = signal_combine f s2 s1.
Proof.
  intros. apply functional_extensionality.
  intros t. unfold signal_combine. apply H.
Qed.

(* ================================================================= *)
(* 内在的型付けによるSignal                                            *)
(* ================================================================= *)

Inductive typed_value : ty -> Type :=
  | tv_int : Z -> typed_value TInt
  | tv_float : R -> typed_value TFloat
  | tv_bool : bool -> typed_value TBool
  | tv_gaussian : R -> R -> typed_value TGaussian
  | tv_unit : typed_value TUnit.

Definition TypedSignal (T : ty) := Time -> typed_value T.

Definition typed_signal_map {T1 T2 : ty}
  (f : typed_value T1 -> typed_value T2)
  (s : TypedSignal T1) : TypedSignal T2 :=
  fun t => f (s t).

(* ================================================================= *)
(* Signal<Gaussian>の例                                               *)
(* ================================================================= *)

Example signal_gaussian_example : TypedSignal TGaussian :=
  fun t => tv_gaussian (10 + INR t) 1.

Example signal_gaussian_sample : 
  TypedSignal TGaussian -> TypedSignal TFloat :=
  fun sg => typed_signal_map
    (fun g => match g with
              | tv_gaussian m s => tv_float (gaussian_sample m s)
              | _ => tv_float 0 (* unreachable *)
              end)
    sg.

(* ================================================================= *)
(* Signalの型安全性                                                   *)
(* ================================================================= *)

Theorem typed_signal_type_safe : 
  forall T (s : TypedSignal T) t,
  exists v : typed_value T, s t = v.
Proof.
  intros. exists (s t). reflexivity.
Qed.

Theorem signal_map_preserves_type :
  forall T1 T2 (f : typed_value T1 -> typed_value T2) 
         (s : TypedSignal T1) (t : Time),
  exists v : typed_value T2, 
    typed_signal_map f s t = v.
Proof.
  intros. exists (f (s t)).
  unfold typed_signal_map. reflexivity.
Qed.
