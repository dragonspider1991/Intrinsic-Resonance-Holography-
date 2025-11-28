(* ::Package:: *)

(* ============================================================================
   ScalingFlows.wl - Coarse-Graining and Expansion Operations
   ============================================================================
   
   Purpose:
     Implements renormalization group-inspired scaling flows for graph states.
     CoarseGrain reduces complexity by merging nodes, while Expand adds
     structure. These operations enable multi-scale optimization.
   
   Inputs:
     - GraphState: A valid GraphState
     - opts: Scaling parameters
   
   Outputs:
     - GraphState: Scaled graph state
   
   Equations Implemented:
     CoarseGrain: Merge nodes based on spectral clustering
       - New node weights = sum of merged weights
       - New edges = aggregated inter-cluster connections
     
     Expand: Add structure via subdivision or refinement
       - Add intermediate nodes on edges
       - Preserve topological properties
   
   References:
     - Renormalization Group (Wilson)
     - Spectral Graph Coarsening
     - Graph Wavelets and Multi-resolution Analysis
   
   ============================================================================ *)

BeginPackage["IRHSuite`ScalingFlows`"];

CoarseGrain::usage = "CoarseGrain[graphState, opts] reduces graph complexity by \
merging similar nodes. Options: \"TargetSize\" -> target node count, \
\"Method\" -> \"Spectral\"|\"Random\"|\"Degree\".
Example: coarse = CoarseGrain[gs, \"TargetSize\" -> 50]";

Expand::usage = "Expand[graphState, opts] increases graph complexity by adding nodes. \
Options: \"ExpansionFactor\" -> scale factor, \"Method\" -> \"Subdivision\"|\"Duplication\".
Example: expanded = Expand[gs, \"ExpansionFactor\" -> 2]";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];

(* ============================================================================
   CoarseGrain: Reduce graph complexity
   ============================================================================ *)

Options[CoarseGrain] = {
  "TargetSize" -> Automatic,
  "Method" -> "Spectral",
  "PreserveConnectivity" -> True
};

CoarseGrain[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {targetSize, method, preserveConn, n, newN, clusters, result},
  
  n = gs["NodeCount"];
  targetSize = OptionValue["TargetSize"];
  method = OptionValue["Method"];
  preserveConn = OptionValue["PreserveConnectivity"];
  
  (* Determine target size *)
  newN = If[targetSize === Automatic,
    Max[2, Floor[n / 2]],
    Clip[targetSize, {2, n - 1}]
  ];
  
  If[newN >= n,
    (* No coarsening needed *)
    Return[gs]
  ];
  
  (* Compute clusters based on method *)
  clusters = Switch[method,
    "Spectral",
      spectralClustering[gs, newN],
    "Random",
      randomClustering[gs, newN],
    "Degree",
      degreeClustering[gs, newN],
    _,
      Message[CoarseGrain::badmethod, method];
      randomClustering[gs, newN]
  ];
  
  (* Build coarsened graph *)
  result = buildCoarsenedGraph[gs, clusters, newN, preserveConn];
  
  result
];

CoarseGrain::badmethod = "Unknown coarsening method: `1`. Using Random.";
CoarseGrain[_, ___] := $Failed;

(* Spectral clustering based on eigenvectors *)
spectralClustering[gs_, k_] := Module[
  {adjMat, n, laplacian, spectrum, eigenvecs, coords, clusters, i},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  
  (* Build normalized Laplacian *)
  laplacian = buildLaplacian[adjMat];
  
  (* Get first k eigenvectors *)
  spectrum = Quiet[
    Eigensystem[N[laplacian], Min[k, n - 1]],
    {Eigensystem::eival}
  ];
  
  If[!MatchQ[spectrum, {_List, _List}],
    (* Fall back to random *)
    Return[randomClustering[gs, k]]
  ];
  
  eigenvecs = spectrum[[2]];
  
  (* Use eigenvector coordinates for clustering *)
  If[Length[eigenvecs] > 0 && Length[eigenvecs[[1]]] == n,
    coords = Transpose[eigenvecs[[1 ;; Min[k-1, Length[eigenvecs]]]]];
    clusters = kMeansCluster[coords, k],
    clusters = randomClustering[gs, k]
  ];
  
  clusters
];

(* Simple k-means implementation *)
kMeansCluster[coords_List, k_Integer] := Module[
  {n, dim, centroids, assignments, newCentroids, iter, maxIter = 50},
  
  n = Length[coords];
  If[n == 0 || k <= 0, Return[Range[Min[n, k]]]];
  
  dim = If[ListQ[First[coords]], Length[First[coords]], 1];
  
  (* Handle edge cases *)
  If[n <= k, Return[Range[n]]];
  
  (* Initialize centroids randomly *)
  centroids = coords[[RandomSample[Range[n], Min[k, n]]]];
  
  (* Iterate *)
  Do[
    (* Assign points to nearest centroid *)
    assignments = Table[
      First[Ordering[
        Table[EuclideanDistance[
          If[ListQ[coords[[i]]], coords[[i]], {coords[[i]]}],
          If[ListQ[centroids[[j]]], centroids[[j]], {centroids[[j]]}]
        ], {j, Length[centroids]}]
      ]],
      {i, n}
    ];
    
    (* Update centroids *)
    newCentroids = Table[
      Mean[Select[
        MapIndexed[If[assignments[[First[#2]]] == j, #1, Nothing] &, coords],
        ListQ[#] || NumericQ[#] &
      ] /. {} -> {centroids[[j]]}],
      {j, Length[centroids]}
    ];
    
    If[newCentroids == centroids, Break[]];
    centroids = newCentroids,
    
    {iter, maxIter}
  ];
  
  assignments
];

(* Random clustering *)
randomClustering[gs_, k_] := Module[
  {n, assignments},
  n = gs["NodeCount"];
  assignments = RandomInteger[{1, k}, n];
  (* Ensure all clusters are used *)
  Do[assignments[[i]] = i, {i, Min[k, n]}];
  assignments
];

(* Degree-based clustering *)
degreeClustering[gs_, k_] := Module[
  {n, adjMat, degrees, sorted, clusterSize, assignments, i},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  degrees = Total[adjMat, {2}];
  sorted = Ordering[degrees, All, Greater];
  
  clusterSize = Ceiling[n / k];
  assignments = ConstantArray[1, n];
  
  Do[
    assignments[[sorted[[i]]]] = Ceiling[i / clusterSize],
    {i, n}
  ];
  
  assignments
];

(* Build Laplacian matrix *)
buildLaplacian[adjMat_] := Module[
  {n, degrees, D, L},
  n = Length[adjMat];
  degrees = Total[adjMat, {2}];
  D = DiagonalMatrix[degrees];
  L = D - adjMat;
  L
];

(* Build coarsened graph from clusters *)
buildCoarsenedGraph[gs_, clusters_, k_, preserveConn_] := Module[
  {n, adjMat, weights, phases, newAdj, newWeights, newPhases,
   i, j, nodesI, nodesJ, edgeWeight, edgePhase},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  phases = gs["Phases"];
  
  (* Initialize new matrices *)
  newAdj = ConstantArray[0, {k, k}];
  newWeights = ConstantArray[0.0, {k, k}];
  newPhases = ConstantArray[0.0, {k, k}];
  
  (* Aggregate edges between clusters *)
  Do[
    If[i != j,
      nodesI = Flatten[Position[clusters, i]];
      nodesJ = Flatten[Position[clusters, j]];
      
      (* Sum weights of inter-cluster edges *)
      edgeWeight = Total[Flatten[
        Table[weights[[ni, nj]], {ni, nodesI}, {nj, nodesJ}]
      ]];
      
      (* Average phases (with wrapping) *)
      edgePhase = Mean[Flatten[
        Table[phases[[ni, nj]], {ni, nodesI}, {nj, nodesJ}]
      ] /. {} -> {0}];
      
      If[edgeWeight > 0,
        newAdj[[i, j]] = 1;
        newAdj[[j, i]] = 1;
        newWeights[[i, j]] = edgeWeight / (Length[nodesI] * Length[nodesJ]);
        newWeights[[j, i]] = newWeights[[i, j]];
        newPhases[[i, j]] = edgePhase;
        newPhases[[j, i]] = -edgePhase
      ]
    ],
    {i, k}, {j, i + 1, k}
  ];
  
  (* Ensure connectivity if requested *)
  If[preserveConn,
    newAdj = ensureConnectivity[newAdj][[1]];
    (* Update weights/phases for new edges *)
    Do[
      If[newAdj[[i, j]] > 0 && newWeights[[i, j]] == 0,
        newWeights[[i, j]] = 0.5;
        newWeights[[j, i]] = 0.5;
        newPhases[[i, j]] = RandomReal[{-Pi, Pi}];
        newPhases[[j, i]] = -newPhases[[i, j]]
      ],
      {i, k}, {j, i + 1, k}
    ]
  ];
  
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> N[newAdj],
    "Weights" -> N[newWeights],
    "Phases" -> N[newPhases],
    "NodeCount" -> k,
    "EdgeCount" -> Length[Position[UpperTriangularize[newAdj], x_ /; x > 0]],
    "EdgeList" -> Position[UpperTriangularize[newAdj], x_ /; x > 0],
    "Metadata" -> Append[gs["Metadata"], 
      "CoarsenedFrom" -> n,
      "CoarseningMethod" -> "Spectral"
    ]
  |>
];

(* Ensure graph connectivity *)
ensureConnectivity[adjMat_] := Module[
  {n, components, numComp, repNodes, i, j, newAdj},
  
  n = Length[adjMat];
  newAdj = adjMat;
  
  (* Find connected components (simplified) *)
  components = findComponents[newAdj];
  numComp = Max[components];
  
  If[numComp > 1,
    (* Connect components *)
    repNodes = Table[First[Flatten[Position[components, c]]], {c, numComp}];
    Do[
      newAdj[[repNodes[[i]], repNodes[[i + 1]]]] = 1;
      newAdj[[repNodes[[i + 1]], repNodes[[i]]]] = 1,
      {i, numComp - 1}
    ]
  ];
  
  {newAdj, components}
];

(* Simple component finding *)
findComponents[adjMat_] := Module[
  {n, visited, components, compNum, queue, current, neighbors},
  
  n = Length[adjMat];
  visited = ConstantArray[False, n];
  components = ConstantArray[0, n];
  compNum = 0;
  
  Do[
    If[!visited[[start]],
      compNum++;
      queue = {start};
      While[Length[queue] > 0,
        current = First[queue];
        queue = Rest[queue];
        If[!visited[[current]],
          visited[[current]] = True;
          components[[current]] = compNum;
          neighbors = Flatten[Position[adjMat[[current]], x_ /; x > 0]];
          queue = Join[queue, Select[neighbors, !visited[[#]] &]]
        ]
      ]
    ],
    {start, n}
  ];
  
  components
];

(* ============================================================================
   Expand: Increase graph complexity
   ============================================================================ *)

Options[Expand] = {
  "ExpansionFactor" -> 2,
  "Method" -> "Subdivision",
  "MaxSize" -> 10000
};

Expand[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {factor, method, maxSize, n, newN, result},
  
  factor = OptionValue["ExpansionFactor"];
  method = OptionValue["Method"];
  maxSize = OptionValue["MaxSize"];
  
  n = gs["NodeCount"];
  newN = Min[Round[n * factor], maxSize];
  
  If[newN <= n,
    Return[gs]
  ];
  
  result = Switch[method,
    "Subdivision",
      subdivisionExpand[gs, newN],
    "Duplication",
      duplicationExpand[gs, newN],
    _,
      Message[Expand::badmethod, method];
      subdivisionExpand[gs, newN]
  ];
  
  result
];

Expand::badmethod = "Unknown expansion method: `1`. Using Subdivision.";
Expand[_, ___] := $Failed;

(* Subdivision expansion: add nodes on edges *)
subdivisionExpand[gs_, targetN_] := Module[
  {n, adjMat, weights, phases, edgeList, numNewNodes, edgesToSplit,
   newAdj, newWeights, newPhases, currentN, edge, u, v, newNode},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  phases = gs["Phases"];
  edgeList = gs["EdgeList"];
  
  numNewNodes = targetN - n;
  
  If[numNewNodes <= 0 || Length[edgeList] == 0,
    Return[gs]
  ];
  
  (* Select edges to subdivide *)
  edgesToSplit = RandomChoice[edgeList, Min[numNewNodes, Length[edgeList]]];
  
  (* Initialize expanded matrices *)
  currentN = n + Length[edgesToSplit];
  newAdj = ConstantArray[0, {currentN, currentN}];
  newWeights = ConstantArray[0.0, {currentN, currentN}];
  newPhases = ConstantArray[0.0, {currentN, currentN}];
  
  (* Copy original graph *)
  newAdj[[1 ;; n, 1 ;; n]] = adjMat;
  newWeights[[1 ;; n, 1 ;; n]] = weights;
  newPhases[[1 ;; n, 1 ;; n]] = phases;
  
  (* Add new nodes *)
  Do[
    edge = edgesToSplit[[i]];
    u = edge[[1]];
    v = edge[[2]];
    newNode = n + i;
    
    (* Remove original edge *)
    newAdj[[u, v]] = 0;
    newAdj[[v, u]] = 0;
    
    (* Add edges to new node *)
    newAdj[[u, newNode]] = 1;
    newAdj[[newNode, u]] = 1;
    newAdj[[v, newNode]] = 1;
    newAdj[[newNode, v]] = 1;
    
    (* Split weights *)
    newWeights[[u, newNode]] = weights[[u, v]] / 2;
    newWeights[[newNode, u]] = weights[[u, v]] / 2;
    newWeights[[v, newNode]] = weights[[u, v]] / 2;
    newWeights[[newNode, v]] = weights[[u, v]] / 2;
    
    (* Split phases *)
    newPhases[[u, newNode]] = phases[[u, v]] / 2;
    newPhases[[newNode, u]] = -phases[[u, v]] / 2;
    newPhases[[v, newNode]] = phases[[u, v]] / 2;
    newPhases[[newNode, v]] = -phases[[u, v]] / 2,
    
    {i, Length[edgesToSplit]}
  ];
  
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> N[newAdj],
    "Weights" -> N[newWeights],
    "Phases" -> N[newPhases],
    "NodeCount" -> currentN,
    "EdgeCount" -> Length[Position[UpperTriangularize[newAdj], x_ /; x > 0]],
    "EdgeList" -> Position[UpperTriangularize[newAdj], x_ /; x > 0],
    "Metadata" -> Append[gs["Metadata"],
      "ExpandedFrom" -> n,
      "ExpansionMethod" -> "Subdivision"
    ]
  |>
];

(* Duplication expansion: duplicate nodes with connections *)
duplicationExpand[gs_, targetN_] := Module[
  {n, adjMat, weights, phases, numNewNodes, nodesToDup,
   newAdj, newWeights, newPhases, currentN, origNode, newNode, neighbors, j},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  phases = gs["Phases"];
  
  numNewNodes = targetN - n;
  
  If[numNewNodes <= 0,
    Return[gs]
  ];
  
  (* Select nodes to duplicate *)
  nodesToDup = RandomChoice[Range[n], numNewNodes];
  
  (* Initialize expanded matrices *)
  currentN = n + numNewNodes;
  newAdj = ConstantArray[0, {currentN, currentN}];
  newWeights = ConstantArray[0.0, {currentN, currentN}];
  newPhases = ConstantArray[0.0, {currentN, currentN}];
  
  (* Copy original graph *)
  newAdj[[1 ;; n, 1 ;; n]] = adjMat;
  newWeights[[1 ;; n, 1 ;; n]] = weights;
  newPhases[[1 ;; n, 1 ;; n]] = phases;
  
  (* Duplicate nodes *)
  Do[
    origNode = nodesToDup[[i]];
    newNode = n + i;
    neighbors = Flatten[Position[adjMat[[origNode]], x_ /; x > 0]];
    
    (* Connect to original node *)
    newAdj[[origNode, newNode]] = 1;
    newAdj[[newNode, origNode]] = 1;
    newWeights[[origNode, newNode]] = 0.5;
    newWeights[[newNode, origNode]] = 0.5;
    newPhases[[origNode, newNode]] = 0;
    newPhases[[newNode, origNode]] = 0;
    
    (* Connect to subset of original neighbors *)
    Do[
      If[RandomReal[] < 0.5,
        newAdj[[j, newNode]] = 1;
        newAdj[[newNode, j]] = 1;
        newWeights[[j, newNode]] = weights[[origNode, j]] * 0.5;
        newWeights[[newNode, j]] = weights[[origNode, j]] * 0.5;
        newPhases[[j, newNode]] = phases[[origNode, j]];
        newPhases[[newNode, j]] = -phases[[origNode, j]]
      ],
      {j, neighbors}
    ],
    {i, numNewNodes}
  ];
  
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> N[newAdj],
    "Weights" -> N[newWeights],
    "Phases" -> N[newPhases],
    "NodeCount" -> currentN,
    "EdgeCount" -> Length[Position[UpperTriangularize[newAdj], x_ /; x > 0]],
    "EdgeList" -> Position[UpperTriangularize[newAdj], x_ /; x > 0],
    "Metadata" -> Append[gs["Metadata"],
      "ExpandedFrom" -> n,
      "ExpansionMethod" -> "Duplication"
    ]
  |>
];

End[];

EndPackage[];
