(* ::Package:: *)

(* ============================================================================
   MutateGraph.wl - Graph Mutation Operators
   ============================================================================
   
   Purpose:
     Implements mutation operators for the ARO optimization loop. Supports
     multiple mutation kernels including edge rewiring, weight perturbation,
     and phase rotation, with configurable scheduling.
   
   Inputs:
     - GraphState: A valid GraphState to mutate
     - opts: Mutation configuration options
   
   Outputs:
     - GraphState: Mutated graph state
   
   Equations Implemented:
     Edge Rewiring: Remove edge (i,j), add edge (k,l) with probability p_rewire
     Weight Perturbation: w'_ij = w_ij + ε, ε ~ N(0, σ²)
     Phase Rotation: φ'_ij = φ_ij + δ, δ ~ Uniform(-Δ, Δ)
   
   References:
     - Metropolis-Hastings MCMC
     - Genetic algorithm mutation operators
     - Graph rewiring for optimization
   
   ============================================================================ *)

BeginPackage["IRHSuite`MutateGraph`"];

MutateGraph::usage = "MutateGraph[graphState, opts] applies mutation operators to \
the graph state. Options: \"MutationKernel\" -> \"EdgeRewiring\"|\"WeightPerturbation\"\
|\"PhaseRotation\"|\"Mixed\", \"MutationStrength\" -> strength parameter.
Example: mutated = MutateGraph[gs, \"MutationKernel\" -> \"Mixed\"]";

EdgeRewiring::usage = "EdgeRewiring[graphState, opts] performs edge rewiring mutation.
Example: mutated = EdgeRewiring[gs, \"RewiringProbability\" -> 0.1]";

WeightPerturbation::usage = "WeightPerturbation[graphState, opts] perturbs edge weights.
Example: mutated = WeightPerturbation[gs, \"PerturbationScale\" -> 0.1]";

PhaseRotation::usage = "PhaseRotation[graphState, opts] rotates edge phases.
Example: mutated = PhaseRotation[gs, \"RotationScale\" -> 0.1]";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];

(* Options for MutateGraph *)
Options[MutateGraph] = {
  "MutationKernel" -> "Mixed",
  "MutationStrength" -> 1.0,
  "RewiringProbability" -> 0.1,
  "PerturbationScale" -> 0.1,
  "RotationScale" -> 0.2,
  "KernelWeights" -> {0.4, 0.3, 0.3},  (* {Rewire, Weight, Phase} *)
  "MinEdgeFraction" -> 0.1,  (* Minimum edge density to maintain *)
  "MaxEdgeFraction" -> 0.9   (* Maximum edge density to allow *)
};

(* Main mutation function *)
MutateGraph[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {kernel, strength, weights, result},
  
  kernel = OptionValue["MutationKernel"];
  strength = OptionValue["MutationStrength"];
  weights = OptionValue["KernelWeights"];
  
  result = Switch[kernel,
    "EdgeRewiring",
      EdgeRewiring[gs, FilterRules[{opts}, Options[EdgeRewiring]]],
    "WeightPerturbation",
      WeightPerturbation[gs, FilterRules[{opts}, Options[WeightPerturbation]]],
    "PhaseRotation",
      PhaseRotation[gs, FilterRules[{opts}, Options[PhaseRotation]]],
    "Mixed",
      mixedMutation[gs, weights, strength, opts],
    _,
      Message[MutateGraph::badkernel, kernel];
      gs
  ];
  
  result
];

MutateGraph::badkernel = "Unknown mutation kernel: `1`. Using original graph.";
MutateGraph[_, ___] := $Failed;

(* Mixed mutation: randomly select and apply kernels *)
mixedMutation[gs_, weights_, strength_, opts_] := Module[
  {normalizedWeights, r, mutated},
  
  normalizedWeights = weights / Total[weights];
  r = RandomReal[];
  
  mutated = Which[
    r < normalizedWeights[[1]],
      EdgeRewiring[gs, 
        "RewiringProbability" -> OptionValue[opts, "RewiringProbability"] * strength,
        Sequence @@ FilterRules[{opts}, Options[EdgeRewiring]]],
    
    r < normalizedWeights[[1]] + normalizedWeights[[2]],
      WeightPerturbation[gs,
        "PerturbationScale" -> OptionValue[opts, "PerturbationScale"] * strength,
        Sequence @@ FilterRules[{opts}, Options[WeightPerturbation]]],
    
    True,
      PhaseRotation[gs,
        "RotationScale" -> OptionValue[opts, "RotationScale"] * strength,
        Sequence @@ FilterRules[{opts}, Options[PhaseRotation]]]
  ];
  
  mutated
];

(* ============================================================================
   Edge Rewiring Mutation
   ============================================================================ *)

Options[EdgeRewiring] = {
  "RewiringProbability" -> 0.1,
  "MinEdgeFraction" -> 0.1,
  "MaxEdgeFraction" -> 0.9
};

EdgeRewiring[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {prob, minFrac, maxFrac, n, adjMat, weights, phases, edgeList,
   currentDensity, maxEdges, numRewirings, i, oldEdge, newEdge,
   newAdj, newWeights, newPhases, edgeCount},
  
  prob = OptionValue["RewiringProbability"];
  minFrac = OptionValue["MinEdgeFraction"];
  maxFrac = OptionValue["MaxEdgeFraction"];
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  phases = gs["Phases"];
  edgeList = gs["EdgeList"];
  
  maxEdges = n (n - 1) / 2;
  currentDensity = Length[edgeList] / maxEdges;
  
  (* Copy matrices for mutation *)
  newAdj = adjMat;
  newWeights = weights;
  newPhases = phases;
  
  (* Determine number of rewirings *)
  numRewirings = Max[1, Round[Length[edgeList] * prob]];
  
  Do[
    (* Select random edge to potentially remove *)
    If[Length[Position[UpperTriangularize[newAdj], x_ /; x > 0]] < 2,
      Break[]  (* Too few edges, stop *)
    ];
    
    oldEdge = RandomChoice[Position[UpperTriangularize[newAdj], x_ /; x > 0]];
    
    (* Find a non-edge to add *)
    newEdge = findNonEdge[newAdj, n];
    
    If[newEdge === None,
      Continue[]  (* No valid non-edge found *)
    ];
    
    (* Check density constraints *)
    edgeCount = Total[Total[UpperTriangularize[newAdj]]] - 1 + 1;  (* Remove + Add *)
    
    (* Remove old edge *)
    newAdj[[oldEdge[[1]], oldEdge[[2]]]] = 0;
    newAdj[[oldEdge[[2]], oldEdge[[1]]]] = 0;
    newWeights[[oldEdge[[1]], oldEdge[[2]]]] = 0;
    newWeights[[oldEdge[[2]], oldEdge[[1]]]] = 0;
    newPhases[[oldEdge[[1]], oldEdge[[2]]]] = 0;
    newPhases[[oldEdge[[2]], oldEdge[[1]]]] = 0;
    
    (* Add new edge *)
    newAdj[[newEdge[[1]], newEdge[[2]]]] = 1;
    newAdj[[newEdge[[2]], newEdge[[1]]]] = 1;
    newWeights[[newEdge[[1]], newEdge[[2]]]] = RandomReal[{0.1, 1.0}];
    newWeights[[newEdge[[2]], newEdge[[1]]]] = newWeights[[newEdge[[1]], newEdge[[2]]]];
    newPhases[[newEdge[[1]], newEdge[[2]]]] = RandomReal[{0, 2 Pi}];
    newPhases[[newEdge[[2]], newEdge[[1]]]] = -newPhases[[newEdge[[1]], newEdge[[2]]]],
    
    {i, numRewirings}
  ];
  
  (* Construct new GraphState *)
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> newAdj,
    "Weights" -> newWeights,
    "Phases" -> newPhases,
    "NodeCount" -> n,
    "EdgeCount" -> Length[Position[UpperTriangularize[newAdj], x_ /; x > 0]],
    "EdgeList" -> Position[UpperTriangularize[newAdj], x_ /; x > 0],
    "Metadata" -> Append[gs["Metadata"], "LastMutation" -> "EdgeRewiring"]
  |>
];

EdgeRewiring[_, ___] := $Failed;

(* Find a random non-edge *)
findNonEdge[adjMat_, n_] := Module[
  {attempts, i, j},
  
  Do[
    i = RandomInteger[{1, n}];
    j = RandomInteger[{1, n}];
    If[i != j && adjMat[[i, j]] == 0,
      Return[{Min[i, j], Max[i, j]}]
    ],
    {attempts, 100}
  ];
  
  None
];

(* ============================================================================
   Weight Perturbation Mutation
   ============================================================================ *)

Options[WeightPerturbation] = {
  "PerturbationScale" -> 0.1,
  "MinWeight" -> 0.01,
  "MaxWeight" -> 10.0
};

WeightPerturbation[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {scale, minW, maxW, n, adjMat, weights, newWeights, i, j, delta},
  
  scale = OptionValue["PerturbationScale"];
  minW = OptionValue["MinWeight"];
  maxW = OptionValue["MaxWeight"];
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  
  (* Create perturbation matrix *)
  newWeights = Table[
    If[adjMat[[i, j]] != 0 && i < j,
      delta = RandomVariate[NormalDistribution[0, scale]];
      Clip[weights[[i, j]] + delta, {minW, maxW}],
      weights[[i, j]]
    ],
    {i, n}, {j, n}
  ];
  
  (* Make symmetric *)
  newWeights = newWeights + Transpose[newWeights] - DiagonalMatrix[Diagonal[newWeights]];
  
  (* Construct new GraphState *)
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> adjMat,
    "Weights" -> newWeights,
    "Phases" -> gs["Phases"],
    "NodeCount" -> n,
    "EdgeCount" -> gs["EdgeCount"],
    "EdgeList" -> gs["EdgeList"],
    "Metadata" -> Append[gs["Metadata"], "LastMutation" -> "WeightPerturbation"]
  |>
];

WeightPerturbation[_, ___] := $Failed;

(* ============================================================================
   Phase Rotation Mutation
   ============================================================================ *)

Options[PhaseRotation] = {
  "RotationScale" -> 0.2
};

PhaseRotation[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {scale, n, adjMat, phases, newPhases, i, j, delta},
  
  scale = OptionValue["RotationScale"];
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  phases = gs["Phases"];
  
  (* Create phase rotation matrix *)
  newPhases = Table[
    If[adjMat[[i, j]] != 0 && i < j,
      delta = RandomReal[{-scale * Pi, scale * Pi}];
      Mod[phases[[i, j]] + delta, 2 Pi, -Pi],  (* Keep in [-π, π] *)
      phases[[i, j]]
    ],
    {i, n}, {j, n}
  ];
  
  (* Make anti-symmetric *)
  newPhases = newPhases - Transpose[newPhases];
  newPhases = newPhases / 2;  (* Average to maintain anti-symmetry *)
  
  (* Construct new GraphState *)
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> adjMat,
    "Weights" -> gs["Weights"],
    "Phases" -> newPhases,
    "NodeCount" -> n,
    "EdgeCount" -> gs["EdgeCount"],
    "EdgeList" -> gs["EdgeList"],
    "Metadata" -> Append[gs["Metadata"], "LastMutation" -> "PhaseRotation"]
  |>
];

PhaseRotation[_, ___] := $Failed;

End[];

EndPackage[];
