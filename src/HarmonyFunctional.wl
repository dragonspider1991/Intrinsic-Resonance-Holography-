(* ::Package:: *)

(* ============================================================================
   HarmonyFunctional.wl - The Harmony Functional Γ and Components
   ============================================================================
   
   Purpose:
     Implements the Harmony Functional Γ, the central objective function for
     IRH optimization. The functional combines vibrational energy, holographic
     entropy, algebraic complexity, and Lorentzian structure into a single
     measure of "spacetime quality".
   
   Inputs:
     - GraphState: A valid GraphState association
     - Parameters: {βH, μ, α} control weights
   
   Outputs:
     - Γ (Real): The harmony functional value
     - Component values: Evib, Sholo, CAlg, DLor
   
   Equations Implemented:
     Γ = βH·Evib + μ·Sholo - α·CAlg + DLor
     
     Where:
       Evib  = Σ_i λ_i^2          (Vibrational energy from eigenspectrum)
       Sholo = -Σ_i p_i log(p_i)  (Holographic entropy, p_i = |λ_i|/Σ|λ_j|)
       CAlg  = f(graph structure)  (Algebraic complexity measure)
       DLor  = g(signature)        (Lorentzian signature term)
   
   Note: The exact formulas for Sholo, CAlg, DLor are documented proxies.
   Domain experts may refine these based on theoretical requirements.
   
   References:
     - IRH Theory: Harmony functional as action principle
     - Graph spectral theory
     - Information-theoretic entropy measures
   
   ============================================================================ *)

BeginPackage["IRHSuite`HarmonyFunctional`"];

Gamma::usage = "Gamma[graphState, params] computes the Harmony Functional \
Γ = βH·Evib + μ·Sholo - α·CAlg + DLor. \
params should be <|\"betaH\"->..., \"mu\"->..., \"alpha\"->...|>.
Example: g = Gamma[gs, <|\"betaH\"->1.0, \"mu\"->0.1, \"alpha\"->0.01|>]";

Evib::usage = "Evib[graphState] computes the vibrational energy component \
Evib = Σ_i λ_i^2, where λ_i are the eigenvalues.
Example: e = Evib[gs]";

Sholo::usage = "Sholo[graphState] computes the holographic entropy component \
Sholo = -Σ_i p_i log(p_i), where p_i = |λ_i|/Σ|λ_j|.
Example: s = Sholo[gs]";

CAlg::usage = "CAlg[graphState] computes the algebraic complexity component \
based on graph structure metrics.
Example: c = CAlg[gs]";

DLor::usage = "DLor[graphState] computes the Lorentzian signature term \
based on the eigenvalue signature pattern.
Example: d = DLor[gs]";

GammaComponents::usage = "GammaComponents[graphState, params] returns all \
components as an Association: <|\"Gamma\", \"Evib\", \"Sholo\", \"CAlg\", \"DLor\"|>.
Example: comps = GammaComponents[gs, params]";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];
Needs["IRHSuite`EigenSpectrum`"];

(* Cache for eigenspectrum to avoid recomputation *)
$eigenCache = <||>;

(* Get eigenvalues with caching *)
getEigenvaluesCached[gs_] := Module[
  {key, spectrum},
  key = Hash[gs["AdjacencyMatrix"]];
  If[!KeyExistsQ[$eigenCache, key],
    spectrum = EigenSpectrum[gs, "ReturnVectors" -> False];
    $eigenCache[key] = spectrum["Eigenvalues"]
  ];
  $eigenCache[key]
];

(* Clear cache *)
ClearEigenCache[] := ($eigenCache = <||>);

(* ============================================================================
   Vibrational Energy: Evib = Σ_i λ_i^2
   
   Physical interpretation: Total "vibrational" energy of the graph modes.
   Higher values indicate more energetic configurations.
   ============================================================================ *)

Evib[gs_?GraphStateQ] := Module[
  {eigenvalues},
  eigenvalues = getEigenvaluesCached[gs];
  If[eigenvalues === $Failed, Return[$Failed]];
  
  Total[Re[eigenvalues]^2]
];

Evib[_] := $Failed;

(* ============================================================================
   Holographic Entropy: Sholo = -Σ_i p_i log(p_i)
   
   Physical interpretation: Information-theoretic entropy of the eigenspectrum.
   Measures how "spread out" the eigenvalue distribution is.
   
   NOTE: This is a proxy implementation. The exact holographic entropy formula
   from IRH theory may involve area-law scaling or other corrections.
   ============================================================================ *)

Sholo[gs_?GraphStateQ] := Module[
  {eigenvalues, absVals, total, probs, entropy},
  
  eigenvalues = getEigenvaluesCached[gs];
  If[eigenvalues === $Failed, Return[$Failed]];
  
  (* Use absolute values for probability distribution *)
  absVals = Abs[eigenvalues];
  total = Total[absVals];
  
  If[total < 10^-15,
    (* Degenerate case: all zero eigenvalues *)
    Return[0]
  ];
  
  (* Normalize to probabilities *)
  probs = absVals / total;
  
  (* Compute Shannon entropy, avoiding log(0) *)
  entropy = -Total[
    Map[
      If[# < 10^-15, 0, # * Log[#]] &,
      probs
    ]
  ];
  
  entropy
];

Sholo[_] := $Failed;

(* ============================================================================
   Algebraic Complexity: CAlg = f(graph structure)
   
   Physical interpretation: Measure of structural complexity of the graph.
   Penalizes overly regular or overly random structures.
   
   Implementation: Uses graph metrics including:
   - Edge density
   - Degree variance
   - Clustering coefficient proxy
   
   NOTE: This is a proxy implementation. The exact complexity measure from
   IRH theory may involve different graph-theoretic quantities.
   ============================================================================ *)

CAlg[gs_?GraphStateQ] := Module[
  {n, edgeCount, maxEdges, density, adjMat, degrees, degreeVar,
   clusterProxy, complexity},
  
  n = gs["NodeCount"];
  edgeCount = gs["EdgeCount"];
  adjMat = gs["AdjacencyMatrix"];
  
  (* Edge density: ratio of edges to maximum possible *)
  maxEdges = n (n - 1) / 2;
  density = If[maxEdges > 0, edgeCount / maxEdges, 0];
  
  (* Degree variance: spread of node degrees *)
  degrees = Total[adjMat, {2}];  (* Row sums = degrees *)
  degreeVar = If[n > 1, Variance[degrees], 0];
  
  (* Clustering coefficient proxy: local triangles *)
  (* Full clustering computation is expensive, use simplified proxy *)
  clusterProxy = computeClusteringProxy[adjMat, n];
  
  (* Combine into complexity measure *)
  (* Penalize both extremes: very sparse and very dense graphs *)
  complexity = Abs[density - 0.5] + 
               Sqrt[degreeVar] / (n + 1) + 
               (1 - clusterProxy);
  
  complexity
];

CAlg[_] := $Failed;

(* Simplified clustering coefficient proxy *)
computeClusteringProxy[adjMat_, n_] := Module[
  {sample, triangles, possibleTriangles},
  
  If[n < 3, Return[0]];
  
  (* Sample some nodes for efficiency *)
  sample = RandomSample[Range[n], Min[n, 20]];
  
  triangles = 0;
  possibleTriangles = 0;
  
  Do[
    Module[{neighbors, k, neighborPairs},
      neighbors = Flatten[Position[adjMat[[i]], x_ /; x > 0]];
      k = Length[neighbors];
      If[k >= 2,
        possibleTriangles += k (k - 1) / 2;
        (* Count actual edges between neighbors *)
        neighborPairs = Subsets[neighbors, {2}];
        triangles += Count[neighborPairs, {u_, v_} /; adjMat[[u, v]] > 0]
      ]
    ],
    {i, sample}
  ];
  
  If[possibleTriangles > 0,
    triangles / possibleTriangles,
    0
  ]
];

(* ============================================================================
   Lorentzian Signature Term: DLor = g(signature)
   
   Physical interpretation: Measures how close the eigenvalue signature is
   to the desired (3,1) Lorentzian signature of physical spacetime.
   
   Target: Exactly 1 negative eigenvalue (time dimension)
   
   Implementation: Returns a term that is maximized when there is exactly
   one negative eigenvalue.
   
   NOTE: This is a proxy implementation. The exact Lorentzian criterion
   may involve more sophisticated spectral analysis.
   ============================================================================ *)

DLor[gs_?GraphStateQ] := Module[
  {eigenvalues, negCount, targetNeg = 1, penalty},
  
  eigenvalues = getEigenvaluesCached[gs];
  If[eigenvalues === $Failed, Return[$Failed]];
  
  (* Count genuinely negative eigenvalues *)
  negCount = Count[Re[eigenvalues], x_ /; x < -10^-10];
  
  (* Score: maximum at target, decreases quadratically away *)
  (* Uses a peaked function around the target *)
  penalty = (negCount - targetNeg)^2;
  
  (* Return positive contribution when signature is correct *)
  Exp[-penalty]
];

DLor[_] := $Failed;

(* ============================================================================
   Full Harmony Functional
   ============================================================================ *)

Options[Gamma] = {
  "EigenTolerance" -> 10^-10
};

Gamma[gs_?GraphStateQ, params_Association, opts:OptionsPattern[]] := Module[
  {betaH, mu, alpha, evib, sholo, calg, dlor, gamma},
  
  (* Extract parameters with defaults *)
  betaH = Lookup[params, "betaH", 1.0];
  mu = Lookup[params, "mu", 0.1];
  alpha = Lookup[params, "alpha", 0.01];
  
  (* Compute components *)
  evib = Evib[gs];
  sholo = Sholo[gs];
  calg = CAlg[gs];
  dlor = DLor[gs];
  
  If[MemberQ[{evib, sholo, calg, dlor}, $Failed],
    Return[$Failed]
  ];
  
  (* Assemble harmony functional *)
  (* Γ = βH·Evib + μ·Sholo - α·CAlg + DLor *)
  gamma = betaH * evib + mu * sholo - alpha * calg + dlor;
  
  gamma
];

Gamma[_, _, ___] := $Failed;

(* Return all components *)
GammaComponents[gs_?GraphStateQ, params_Association, opts:OptionsPattern[]] := Module[
  {betaH, mu, alpha, evib, sholo, calg, dlor, gamma},
  
  betaH = Lookup[params, "betaH", 1.0];
  mu = Lookup[params, "mu", 0.1];
  alpha = Lookup[params, "alpha", 0.01];
  
  evib = Evib[gs];
  sholo = Sholo[gs];
  calg = CAlg[gs];
  dlor = DLor[gs];
  
  If[MemberQ[{evib, sholo, calg, dlor}, $Failed],
    Return[$Failed]
  ];
  
  gamma = betaH * evib + mu * sholo - alpha * calg + dlor;
  
  <|
    "Gamma" -> gamma,
    "Evib" -> evib,
    "Sholo" -> sholo,
    "CAlg" -> calg,
    "DLor" -> dlor,
    "Parameters" -> <|"betaH" -> betaH, "mu" -> mu, "alpha" -> alpha|>
  |>
];

GammaComponents[_, _, ___] := $Failed;

End[];

EndPackage[];
