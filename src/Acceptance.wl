(* ::Package:: *)

(* ============================================================================
   Acceptance.wl - Acceptance Criteria for Optimization
   ============================================================================
   
   Purpose:
     Implements acceptance criteria for the ARO optimization loop. Uses a
     hybrid Metropolis-Hastings + Adam-inspired acceptance scheme that
     balances exploration and exploitation.
   
   Inputs:
     - deltaGamma: Change in Harmony Functional (Γ_new - Γ_old)
     - temperature: Current annealing temperature
     - opts: Additional configuration
   
   Outputs:
     - Boolean: True if change should be accepted
   
   Equations Implemented:
     Metropolis acceptance: P(accept) = min(1, exp(-ΔΓ/T))
     Modified acceptance: Incorporates momentum for smoother convergence
   
   References:
     - Metropolis-Hastings algorithm
     - Simulated Annealing (Kirkpatrick et al., 1983)
     - Adam optimizer momentum concepts
   
   ============================================================================ *)

BeginPackage["IRHSuite`Acceptance`"];

AcceptChange::usage = "AcceptChange[deltaGamma, temperature, opts] determines whether \
to accept a proposed change using hybrid Metropolis + momentum acceptance. \
Returns True to accept, False to reject.
Example: AcceptChange[-0.5, 1.0]  (* Always accept improvements *)";

AcceptanceRatio::usage = "AcceptanceRatio[deltaGamma, temperature] computes the \
acceptance probability without making the decision.
Example: p = AcceptanceRatio[0.1, 0.5]";

SetAcceptanceState::usage = "SetAcceptanceState[state] sets the internal momentum \
state for adaptive acceptance.";

GetAcceptanceState::usage = "GetAcceptanceState[] returns the current momentum state.";

ResetAcceptanceState::usage = "ResetAcceptanceState[] resets the momentum state.";

Begin["`Private`"];

(* Internal state for momentum-based acceptance *)
$acceptanceState = <|
  "Momentum" -> 0,
  "Beta" -> 0.9,
  "RecentDeltas" -> {},
  "MaxHistory" -> 100,
  "AcceptCount" -> 0,
  "RejectCount" -> 0
|>;

(* Set state *)
SetAcceptanceState[state_Association] := ($acceptanceState = state);

(* Get state *)
GetAcceptanceState[] := $acceptanceState;

(* Reset state *)
ResetAcceptanceState[] := (
  $acceptanceState = <|
    "Momentum" -> 0,
    "Beta" -> 0.9,
    "RecentDeltas" -> {},
    "MaxHistory" -> 100,
    "AcceptCount" -> 0,
    "RejectCount" -> 0
  |>;
);

(* Options *)
Options[AcceptChange] = {
  "UseMomentum" -> True,
  "MomentumBeta" -> 0.9,
  "MinAcceptance" -> 0.01,  (* Always have small chance to accept *)
  "Maximizing" -> False     (* Set True if maximizing Γ instead of minimizing *)
};

(* Main acceptance function *)
AcceptChange[deltaGamma_?NumericQ, temperature_?NumericQ, opts:OptionsPattern[]] := 
Module[
  {useMom, beta, minAccept, maximizing, effectiveDelta, acceptProb, accept},
  
  useMom = OptionValue["UseMomentum"];
  beta = OptionValue["MomentumBeta"];
  minAccept = OptionValue["MinAcceptance"];
  maximizing = OptionValue["Maximizing"];
  
  (* Adjust delta based on optimization direction *)
  (* For maximizing: positive delta is improvement *)
  (* For minimizing: negative delta is improvement *)
  effectiveDelta = If[maximizing, -deltaGamma, deltaGamma];
  
  (* Update momentum *)
  If[useMom,
    $acceptanceState["Momentum"] = 
      beta * $acceptanceState["Momentum"] + (1 - beta) * effectiveDelta;
    
    (* Store recent deltas *)
    $acceptanceState["RecentDeltas"] = 
      Take[Append[$acceptanceState["RecentDeltas"], effectiveDelta], 
           -$acceptanceState["MaxHistory"]];
  ];
  
  (* Compute acceptance probability *)
  acceptProb = computeAcceptanceProbability[effectiveDelta, temperature, useMom, minAccept];
  
  (* Make decision *)
  accept = RandomReal[] < acceptProb;
  
  (* Update statistics *)
  If[accept,
    $acceptanceState["AcceptCount"]++,
    $acceptanceState["RejectCount"]++
  ];
  
  accept
];

AcceptChange[_, _, ___] := (
  Message[AcceptChange::badargs];
  False
);

AcceptChange::badargs = "AcceptChange requires numeric deltaGamma and temperature.";

(* Compute acceptance probability *)
computeAcceptanceProbability[effectiveDelta_, temperature_, useMom_, minAccept_] := 
Module[
  {baseProb, momentumAdjust, finalProb},
  
  (* Base Metropolis probability *)
  If[effectiveDelta <= 0,
    (* Improvement: always accept *)
    baseProb = 1.0,
    (* Deterioration: probabilistic acceptance *)
    If[temperature <= 0,
      baseProb = 0,
      baseProb = Exp[-effectiveDelta / temperature]
    ]
  ];
  
  (* Apply momentum adjustment *)
  If[useMom && Length[$acceptanceState["RecentDeltas"]] > 5,
    (* If recent trend is improving, be more accepting of small deteriorations *)
    momentumAdjust = If[$acceptanceState["Momentum"] < 0,
      1.0 + 0.1 * Min[1, -$acceptanceState["Momentum"]],  (* Boost acceptance *)
      1.0 - 0.1 * Min[1, $acceptanceState["Momentum"]]    (* Reduce acceptance *)
    ];
    baseProb = baseProb * momentumAdjust,
    momentumAdjust = 1.0
  ];
  
  (* Apply minimum acceptance threshold *)
  finalProb = Max[minAccept, Min[1.0, baseProb]];
  
  finalProb
];

(* Acceptance ratio without decision *)
AcceptanceRatio[deltaGamma_?NumericQ, temperature_?NumericQ, opts:OptionsPattern[AcceptChange]] := 
Module[
  {maximizing, effectiveDelta},
  
  maximizing = OptionValue["Maximizing"];
  effectiveDelta = If[maximizing, -deltaGamma, deltaGamma];
  
  computeAcceptanceProbability[effectiveDelta, temperature, False, 0]
];

AcceptanceRatio[_, _] := 0;

End[];

EndPackage[];
