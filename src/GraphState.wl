(* ::Package:: *)

(* ============================================================================
   GraphState.wl - Graph State Creation and Validation
   ============================================================================
   
   Purpose:
     Creates and validates graph states for Intrinsic Resonance Holography.
     A GraphState encapsulates the combinatorial and geometric structure
     representing a discrete quantum spacetime.
   
   Inputs:
     - N (Integer): Number of nodes in the graph
     - opts: Options including Seed, Precision, InitialTopology
   
   Outputs:
     - GraphState Association containing:
       * "AdjacencyMatrix": Weighted adjacency matrix
       * "Weights": Edge weights
       * "Phases": Phase factors on edges  
       * "NodeCount": Number of nodes
       * "EdgeCount": Number of edges
       * "Metadata": Creation metadata
   
   Equations Implemented:
     - Random graph generation: P(edge) = p (Erdős–Rényi model)
     - Weight initialization: w_ij ~ Uniform(0,1) or Gaussian
     - Phase initialization: φ_ij ~ Uniform(0, 2π)
   
   References:
     - IRH Theory: Discrete graph as quantum spacetime substrate
     - Erdős–Rényi random graphs
   
   ============================================================================ *)

BeginPackage["IRHSuite`GraphState`"];

CreateGraphState::usage = "CreateGraphState[n, opts] creates a graph state with n nodes. \
Options: \"Seed\" -> integer for reproducibility, \"Precision\" -> working precision, \
\"InitialTopology\" -> \"Random\"|\"Complete\"|\"Cycle\"|\"Lattice\", \
\"EdgeProbability\" -> probability for random graphs.
Example: g = CreateGraphState[50, \"Seed\" -> 42]";

ValidateGraphState::usage = "ValidateGraphState[graphState] validates a GraphState association \
and returns True if valid, or a list of error messages.
Example: ValidateGraphState[g]";

GraphStateQ::usage = "GraphStateQ[expr] returns True if expr is a valid GraphState.
Example: GraphStateQ[g]";

GetAdjacencyMatrix::usage = "GetAdjacencyMatrix[graphState] extracts the adjacency matrix.
Example: mat = GetAdjacencyMatrix[g]";

GetNodeCount::usage = "GetNodeCount[graphState] returns the number of nodes.
Example: n = GetNodeCount[g]";

GetEdgeCount::usage = "GetEdgeCount[graphState] returns the number of edges.
Example: e = GetEdgeCount[g]";

Begin["`Private`"];

(* Options for CreateGraphState *)
Options[CreateGraphState] = {
  "Seed" -> Automatic,
  "Precision" -> MachinePrecision,
  "InitialTopology" -> "Random",
  "EdgeProbability" -> 0.3,
  "WeightDistribution" -> "Uniform",
  "PhaseDistribution" -> "Uniform"
};

(* Main creation function *)
CreateGraphState[n_Integer, opts:OptionsPattern[]] := Module[
  {seed, precision, topology, edgeProb, weightDist, phaseDist,
   adjMatrix, weights, phases, edgeList, nodeCount, edgeCount,
   metadata, timestamp},
  
  (* Validate inputs *)
  If[n < 2,
    Message[CreateGraphState::toosmall, n];
    Return[$Failed]
  ];
  
  If[n > 10000,
    Message[CreateGraphState::toolarge, n];
    Return[$Failed]
  ];
  
  (* Extract options *)
  seed = OptionValue["Seed"];
  precision = OptionValue["Precision"];
  topology = OptionValue["InitialTopology"];
  edgeProb = OptionValue["EdgeProbability"];
  weightDist = OptionValue["WeightDistribution"];
  phaseDist = OptionValue["PhaseDistribution"];
  
  (* Set seed for reproducibility *)
  If[seed =!= Automatic,
    SeedRandom[seed]
  ];
  
  (* Generate adjacency structure based on topology *)
  adjMatrix = generateTopology[n, topology, edgeProb, precision];
  If[adjMatrix === $Failed,
    Return[$Failed]
  ];
  
  (* Generate weights *)
  weights = generateWeights[adjMatrix, weightDist, precision];
  
  (* Generate phases *)
  phases = generatePhases[adjMatrix, phaseDist, precision];
  
  (* Compute edge list and counts *)
  edgeList = Position[UpperTriangularize[adjMatrix], x_ /; x != 0];
  nodeCount = n;
  edgeCount = Length[edgeList];
  
  (* Create metadata *)
  timestamp = DateString["ISODateTime"];
  metadata = <|
    "CreatedAt" -> timestamp,
    "Seed" -> seed,
    "Precision" -> precision,
    "Topology" -> topology,
    "EdgeProbability" -> edgeProb,
    "WeightDistribution" -> weightDist,
    "PhaseDistribution" -> phaseDist,
    "Version" -> "3.0"
  |>;
  
  (* Assemble and return GraphState *)
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> adjMatrix,
    "Weights" -> weights,
    "Phases" -> phases,
    "NodeCount" -> nodeCount,
    "EdgeCount" -> edgeCount,
    "EdgeList" -> edgeList,
    "Metadata" -> metadata
  |>
];

(* Messages *)
CreateGraphState::toosmall = "Node count `1` is too small. Minimum is 2.";
CreateGraphState::toolarge = "Node count `1` is too large. Maximum is 10000.";
CreateGraphState::badtopology = "Unknown topology type: `1`.";

(* Generate topology based on type *)
generateTopology[n_, "Random", p_, prec_] := Module[
  {mat, i, j},
  mat = Table[
    If[i < j,
      If[RandomReal[] < p, 1, 0],
      0
    ],
    {i, n}, {j, n}
  ];
  (* Make symmetric *)
  mat = mat + Transpose[mat];
  N[mat, prec]
];

generateTopology[n_, "Complete", _, prec_] := Module[
  {mat},
  mat = ConstantArray[1, {n, n}] - IdentityMatrix[n];
  N[mat, prec]
];

generateTopology[n_, "Cycle", _, prec_] := Module[
  {mat, i},
  mat = SparseArray[
    Join[
      Table[{i, Mod[i, n] + 1} -> 1, {i, n}],
      Table[{Mod[i, n] + 1, i} -> 1, {i, n}]
    ],
    {n, n}
  ];
  N[Normal[mat], prec]
];

generateTopology[n_, "Lattice", _, prec_] := Module[
  {side, mat, connections, i, j, idx1, idx2},
  (* Create 2D lattice if possible *)
  side = Floor[Sqrt[n]];
  mat = ConstantArray[0, {n, n}];
  Do[
    Do[
      idx1 = (i - 1) * side + j;
      If[idx1 <= n,
        (* Right neighbor *)
        idx2 = (i - 1) * side + j + 1;
        If[j < side && idx2 <= n,
          mat[[idx1, idx2]] = 1;
          mat[[idx2, idx1]] = 1
        ];
        (* Bottom neighbor *)
        idx2 = i * side + j;
        If[i < side && idx2 <= n,
          mat[[idx1, idx2]] = 1;
          mat[[idx2, idx1]] = 1
        ]
      ],
      {j, side}
    ],
    {i, side}
  ];
  N[mat, prec]
];

generateTopology[n_, topology_, _, _] := (
  Message[CreateGraphState::badtopology, topology];
  $Failed
);

(* Generate edge weights *)
generateWeights[adjMatrix_, "Uniform", prec_] := Module[
  {n, weights},
  n = Length[adjMatrix];
  weights = Table[
    If[adjMatrix[[i, j]] != 0 && i < j,
      RandomReal[{0.1, 1.0}],
      If[adjMatrix[[i, j]] != 0 && i > j,
        0,  (* Will be filled by symmetry *)
        0
      ]
    ],
    {i, n}, {j, n}
  ];
  weights = weights + Transpose[weights];
  N[weights, prec]
];

generateWeights[adjMatrix_, "Gaussian", prec_] := Module[
  {n, weights},
  n = Length[adjMatrix];
  weights = Table[
    If[adjMatrix[[i, j]] != 0 && i < j,
      Max[0.01, RandomVariate[NormalDistribution[0.5, 0.2]]],
      0
    ],
    {i, n}, {j, n}
  ];
  weights = weights + Transpose[weights];
  N[weights, prec]
];

generateWeights[adjMatrix_, _, prec_] := 
  generateWeights[adjMatrix, "Uniform", prec];

(* Generate edge phases *)
generatePhases[adjMatrix_, "Uniform", prec_] := Module[
  {n, phases},
  n = Length[adjMatrix];
  phases = Table[
    If[adjMatrix[[i, j]] != 0 && i < j,
      RandomReal[{0, 2 Pi}],
      If[adjMatrix[[i, j]] != 0 && i > j,
        0,  (* Will be filled by anti-symmetry *)
        0
      ]
    ],
    {i, n}, {j, n}
  ];
  (* Phases are anti-symmetric: φ_ji = -φ_ij *)
  phases = phases - Transpose[phases];
  N[phases, prec]
];

generatePhases[adjMatrix_, _, prec_] := 
  generatePhases[adjMatrix, "Uniform", prec];

(* Validation function *)
ValidateGraphState[gs_] := Module[
  {errors = {}, n, adjMat, weights, phases},
  
  (* Check type *)
  If[!AssociationQ[gs],
    AppendTo[errors, "GraphState must be an Association"];
    Return[errors]
  ];
  
  If[!KeyExistsQ[gs, "Type"] || gs["Type"] != "GraphState",
    AppendTo[errors, "Invalid Type field"]
  ];
  
  (* Check required keys *)
  If[!KeyExistsQ[gs, "AdjacencyMatrix"],
    AppendTo[errors, "Missing AdjacencyMatrix"]
  ];
  If[!KeyExistsQ[gs, "Weights"],
    AppendTo[errors, "Missing Weights"]
  ];
  If[!KeyExistsQ[gs, "Phases"],
    AppendTo[errors, "Missing Phases"]
  ];
  If[!KeyExistsQ[gs, "NodeCount"],
    AppendTo[errors, "Missing NodeCount"]
  ];
  
  If[Length[errors] > 0, Return[errors]];
  
  (* Check matrix properties *)
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  phases = gs["Phases"];
  n = gs["NodeCount"];
  
  If[Dimensions[adjMat] != {n, n},
    AppendTo[errors, "AdjacencyMatrix dimension mismatch"]
  ];
  If[Dimensions[weights] != {n, n},
    AppendTo[errors, "Weights dimension mismatch"]
  ];
  If[Dimensions[phases] != {n, n},
    AppendTo[errors, "Phases dimension mismatch"]
  ];
  
  (* Check symmetry of adjacency and weights *)
  If[Max[Abs[adjMat - Transpose[adjMat]]] > 10^-10,
    AppendTo[errors, "AdjacencyMatrix not symmetric"]
  ];
  If[Max[Abs[weights - Transpose[weights]]] > 10^-10,
    AppendTo[errors, "Weights not symmetric"]
  ];
  
  (* Check anti-symmetry of phases *)
  If[Max[Abs[phases + Transpose[phases]]] > 10^-10,
    AppendTo[errors, "Phases not anti-symmetric"]
  ];
  
  If[Length[errors] == 0, True, errors]
];

(* Predicate *)
GraphStateQ[gs_] := ValidateGraphState[gs] === True;

(* Accessor functions *)
GetAdjacencyMatrix[gs_?GraphStateQ] := gs["AdjacencyMatrix"];
GetAdjacencyMatrix[_] := $Failed;

GetNodeCount[gs_?GraphStateQ] := gs["NodeCount"];
GetNodeCount[_] := $Failed;

GetEdgeCount[gs_?GraphStateQ] := gs["EdgeCount"];
GetEdgeCount[_] := $Failed;

End[];

EndPackage[];
