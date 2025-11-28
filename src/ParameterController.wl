(* ::Package:: *)

(* ============================================================================
   ParameterController.wl - Adaptive Parameter Update Strategy
   ============================================================================
   
   Purpose:
     Implements configurable controller strategies for updating the Harmony
     Functional parameters {βH, μ, α} during HAGO optimization. Supports
     fixed, adaptive, and learning-based update schemes.
   
   Inputs:
     - currentParams: Current parameter values {βH, μ, α}
     - history: Optimization history (Γ values, gradients, etc.)
     - opts: Strategy configuration
   
   Outputs:
     - newParams: Updated parameter values
   
   Equations Implemented:
     Fixed:    params(t+1) = params(t)
     Linear:   params(t+1) = params(0) + rate * t
     Adaptive: params(t+1) = params(t) + η * gradient (Adam-inspired)
     
   References:
     - Adam optimizer (Kingma & Ba, 2014)
     - Simulated annealing parameter schedules
   
   ============================================================================ *)

BeginPackage["IRHSuite`ParameterController`"];

UpdateParameters::usage = "UpdateParameters[params, history, opts] updates the \
control parameters based on optimization history. \
Options: \"Strategy\" -> \"Fixed\"|\"Linear\"|\"Adaptive\"|\"Adam\".
Example: newParams = UpdateParameters[params, history, \"Strategy\" -> \"Adaptive\"]";

InitializeController::usage = "InitializeController[config] initializes the \
parameter controller state from configuration.
Example: ctrl = InitializeController[config]";

GetControllerState::usage = "GetControllerState[] returns the current internal state \
of the adaptive controller (momentum, variance estimates, etc.).";

ResetController::usage = "ResetController[] resets the controller internal state.";

Begin["`Private`"];

(* Controller internal state for adaptive methods *)
$controllerState = <|
  "Iteration" -> 0,
  "Momentum" -> <|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>,
  "Variance" -> <|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>,
  "Beta1" -> 0.9,
  "Beta2" -> 0.999,
  "Epsilon" -> 10^-8,
  "LearningRate" -> 0.001
|>;

(* Initialize controller from config *)
InitializeController[config_Association] := Module[
  {ctrlConfig},
  
  ctrlConfig = Lookup[config, "controller", <||>];
  
  $controllerState = <|
    "Iteration" -> 0,
    "Momentum" -> <|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>,
    "Variance" -> <|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>,
    "Beta1" -> Lookup[ctrlConfig, "beta1", 0.9],
    "Beta2" -> Lookup[ctrlConfig, "beta2", 0.999],
    "Epsilon" -> Lookup[ctrlConfig, "epsilon", 10^-8],
    "LearningRate" -> Lookup[ctrlConfig, "learningRate", 0.001],
    "Strategy" -> Lookup[ctrlConfig, "strategy", "Fixed"],
    "InitialParams" -> <|
      "betaH" -> Lookup[ctrlConfig, "betaH", 1.0],
      "mu" -> Lookup[ctrlConfig, "mu", 0.1],
      "alpha" -> Lookup[ctrlConfig, "alpha", 0.01]
    |>
  |>;
  
  $controllerState
];

(* Reset controller state *)
ResetController[] := (
  $controllerState["Iteration"] = 0;
  $controllerState["Momentum"] = <|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>;
  $controllerState["Variance"] = <|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>;
);

(* Get current state *)
GetControllerState[] := $controllerState;

(* Options for UpdateParameters *)
Options[UpdateParameters] = {
  "Strategy" -> "Fixed",
  "LearningRate" -> 0.001,
  "LinearRate" -> 0.0001,
  "MaxParams" -> <|"betaH" -> 10.0, "mu" -> 1.0, "alpha" -> 0.1|>,
  "MinParams" -> <|"betaH" -> 0.1, "mu" -> 0.001, "alpha" -> 0.0001|>
};

(* Main update function *)
UpdateParameters[params_Association, history_List, opts:OptionsPattern[]] := Module[
  {strategy, newParams},
  
  strategy = OptionValue["Strategy"];
  
  newParams = Switch[strategy,
    "Fixed",
      updateFixed[params, history, opts],
    "Linear",
      updateLinear[params, history, opts],
    "Adaptive",
      updateAdaptive[params, history, opts],
    "Adam",
      updateAdam[params, history, opts],
    _,
      Message[UpdateParameters::badstrategy, strategy];
      params
  ];
  
  (* Clamp to valid range *)
  newParams = clampParameters[newParams, opts];
  
  (* Update iteration counter *)
  $controllerState["Iteration"] += 1;
  
  newParams
];

UpdateParameters::badstrategy = "Unknown strategy: `1`. Using Fixed.";

(* Fixed strategy: parameters don't change *)
updateFixed[params_, history_, opts_] := params;

(* Linear strategy: parameters change linearly with iteration *)
updateLinear[params_, history_, opts_] := Module[
  {rate, iter, initial},
  
  rate = OptionValue[opts, "LinearRate"];
  iter = $controllerState["Iteration"];
  initial = $controllerState["InitialParams"];
  
  <|
    "betaH" -> initial["betaH"] * (1 + rate * iter),
    "mu" -> initial["mu"] * (1 + rate * iter),
    "alpha" -> initial["alpha"] * (1 + rate * iter)
  |>
];

(* Adaptive strategy: gradient-based update *)
updateAdaptive[params_, history_, opts_] := Module[
  {gradient, lr},
  
  If[Length[history] < 2,
    Return[params]
  ];
  
  lr = OptionValue[opts, "LearningRate"];
  
  (* Estimate gradient from history *)
  gradient = estimateGradient[params, history];
  
  <|
    "betaH" -> params["betaH"] + lr * gradient["betaH"],
    "mu" -> params["mu"] + lr * gradient["mu"],
    "alpha" -> params["alpha"] + lr * gradient["alpha"]
  |>
];

(* Adam strategy: momentum + adaptive learning rate *)
updateAdam[params_, history_, opts_] := Module[
  {gradient, lr, beta1, beta2, eps, m, v, mHat, vHat, iter, newParams},
  
  If[Length[history] < 2,
    Return[params]
  ];
  
  lr = $controllerState["LearningRate"];
  beta1 = $controllerState["Beta1"];
  beta2 = $controllerState["Beta2"];
  eps = $controllerState["Epsilon"];
  iter = $controllerState["Iteration"] + 1;
  
  (* Estimate gradient *)
  gradient = estimateGradient[params, history];
  
  (* Update momentum and variance *)
  m = $controllerState["Momentum"];
  v = $controllerState["Variance"];
  
  newParams = <||>;
  
  Do[
    (* Update biased first moment estimate *)
    m[key] = beta1 * m[key] + (1 - beta1) * gradient[key];
    
    (* Update biased second moment estimate *)
    v[key] = beta2 * v[key] + (1 - beta2) * gradient[key]^2;
    
    (* Compute bias-corrected estimates *)
    mHat = m[key] / (1 - beta1^iter);
    vHat = v[key] / (1 - beta2^iter);
    
    (* Update parameter *)
    newParams[key] = params[key] + lr * mHat / (Sqrt[vHat] + eps),
    
    {key, {"betaH", "mu", "alpha"}}
  ];
  
  (* Store updated state *)
  $controllerState["Momentum"] = m;
  $controllerState["Variance"] = v;
  
  newParams
];

(* Estimate gradient from history *)
estimateGradient[params_, history_] := Module[
  {recent, gammaValues, dGamma, n},
  
  n = Min[Length[history], 10];
  recent = Take[history, -n];
  
  gammaValues = If[AssociationQ[#], #["Gamma"], #] & /@ recent;
  
  If[Length[gammaValues] < 2,
    Return[<|"betaH" -> 0, "mu" -> 0, "alpha" -> 0|>]
  ];
  
  (* Simple finite difference approximation *)
  dGamma = Last[gammaValues] - First[gammaValues];
  
  (* Gradient direction: increase if improving, decrease if not *)
  (* 
     IMPLEMENTATION NOTE: This is a simplified heuristic gradient estimate.
     A proper gradient computation would require:
       1. Evaluating Γ at (params + δ) for each parameter
       2. Computing ∂Γ/∂param ≈ (Γ(param+δ) - Γ(param)) / δ
     
     Current approach uses optimization history as a proxy for gradient direction.
     This is a DOCUMENTED PROXY suitable for initial exploration but should be
     replaced with numerical differentiation for production optimization.
     
     The sign-based approach provides directional guidance without magnitude,
     relying on the learning rate for step size control.
  *)
  <|
    "betaH" -> Sign[dGamma] * 0.01,
    "mu" -> Sign[dGamma] * 0.001,
    "alpha" -> -Sign[dGamma] * 0.0001  (* alpha is subtracted in Γ, so opposite sign *)
  |>
];

(* Clamp parameters to valid range *)
clampParameters[params_, opts_] := Module[
  {maxP, minP},
  
  maxP = OptionValue[opts, "MaxParams"];
  minP = OptionValue[opts, "MinParams"];
  
  <|
    "betaH" -> Clip[params["betaH"], {minP["betaH"], maxP["betaH"]}],
    "mu" -> Clip[params["mu"], {minP["mu"], maxP["mu"]}],
    "alpha" -> Clip[params["alpha"], {minP["alpha"], maxP["alpha"]}]
  |>
];

End[];

EndPackage[];
