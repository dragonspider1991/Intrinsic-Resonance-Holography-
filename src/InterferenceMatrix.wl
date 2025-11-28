(* ::Package:: *)

(* ============================================================================
   InterferenceMatrix.wl - Signed Weighted Laplacian Construction
   ============================================================================
   
   Purpose:
     Constructs the interference matrix (signed weighted Laplacian) from a
     GraphState. This matrix encodes the quantum interference structure of
     the discrete spacetime and is central to the eigenspectrum computation.
   
   Inputs:
     - GraphState: A valid GraphState association
   
   Outputs:
     - Matrix (n×n): The signed weighted Laplacian matrix L
   
   Equations Implemented:
     The interference matrix L is defined as:
       L_ij = -w_ij * exp(i*φ_ij)  for i ≠ j (where edge exists)
       L_ii = Σ_j w_ij             (degree term)
     
     This is a complex Hermitian matrix when phases are anti-symmetric.
   
   References:
     - Graph Laplacian: Standard combinatorial Laplacian L = D - A
     - Magnetic Laplacian: L(θ)_ij = -exp(iθ_ij) for connected pairs
     - IRH Theory: Phases encode holonomy/gauge connection
   
   ============================================================================ *)

BeginPackage["IRHSuite`InterferenceMatrix`"];

BuildInterferenceMatrix::usage = "BuildInterferenceMatrix[graphState] constructs the \
signed weighted Laplacian (interference matrix) from a GraphState. \
Returns a complex Hermitian matrix encoding quantum interference patterns.
Example: L = BuildInterferenceMatrix[g]";

GetInterferenceMatrixReal::usage = "GetInterferenceMatrixReal[graphState] returns the \
real part of the interference matrix (standard weighted Laplacian).
Example: Lreal = GetInterferenceMatrixReal[g]";

GetInterferenceMatrixImag::usage = "GetInterferenceMatrixImag[graphState] returns the \
imaginary part of the interference matrix.
Example: Limag = GetInterferenceMatrixImag[g]";

Begin["`Private`"];

(* Needs GraphState package for validation *)
Needs["IRHSuite`GraphState`"];

(* Main construction function *)
BuildInterferenceMatrix[gs_?GraphStateQ] := Module[
  {n, adjMat, weights, phases, L, i, j, offDiag, degree},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  phases = gs["Phases"];
  
  (* Build off-diagonal terms: L_ij = -w_ij * exp(i*φ_ij) *)
  offDiag = Table[
    If[adjMat[[i, j]] != 0 && i != j,
      -weights[[i, j]] * Exp[I * phases[[i, j]]],
      0
    ],
    {i, n}, {j, n}
  ];
  
  (* Build diagonal terms: L_ii = Σ_j w_ij (weighted degree) *)
  degree = Table[
    Total[weights[[i]]],
    {i, n}
  ];
  
  (* Assemble full matrix *)
  L = offDiag + DiagonalMatrix[degree];
  
  (* Verify Hermiticity (should hold due to anti-symmetric phases) *)
  (* L†_ij = L*_ji = -w_ji * exp(-i*φ_ji) = -w_ij * exp(i*φ_ij) = L_ij *)
  
  L
];

BuildInterferenceMatrix[_] := (
  Message[BuildInterferenceMatrix::invalidgs];
  $Failed
);

BuildInterferenceMatrix::invalidgs = "Input is not a valid GraphState.";

(* Real part (standard weighted Laplacian) *)
GetInterferenceMatrixReal[gs_?GraphStateQ] := Module[
  {L},
  L = BuildInterferenceMatrix[gs];
  Re[L]
];

GetInterferenceMatrixReal[_] := $Failed;

(* Imaginary part *)
GetInterferenceMatrixImag[gs_?GraphStateQ] := Module[
  {L},
  L = BuildInterferenceMatrix[gs];
  Im[L]
];

GetInterferenceMatrixImag[_] := $Failed;

End[];

EndPackage[];
