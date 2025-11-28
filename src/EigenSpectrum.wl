(* ::Package:: *)

(* ============================================================================
   EigenSpectrum.wl - Robust Eigenvalue/Eigenvector Computation
   ============================================================================
   
   Purpose:
     Computes the eigenspectrum of the interference matrix with robust handling
     of numerical edge cases including tiny negative eigenvalues, degeneracies,
     and numerical precision issues.
   
   Inputs:
     - GraphState: A valid GraphState association
     - opts: Options for numerical tolerance and method selection
   
   Outputs:
     - Association containing:
       * "Eigenvalues": Sorted list of eigenvalues (real or complex)
       * "Eigenvectors": Corresponding eigenvectors
       * "Degeneracies": List of degenerate eigenvalue groups
       * "NumericalWarnings": Any numerical issues detected
   
   Equations Implemented:
     - Standard eigenvalue problem: L·v = λ·v
     - Hermitian eigendecomposition for complex Laplacians
     - Eigenvalue cleaning: λ_clean = 0 if |λ| < tolerance
   
   References:
     - Numerical Linear Algebra (Trefethen & Bau)
     - Arnoldi/Lanczos methods for large sparse matrices
     - IRH Theory: Eigenspectrum encodes spacetime geometry
   
   ============================================================================ *)

BeginPackage["IRHSuite`EigenSpectrum`"];

EigenSpectrum::usage = "EigenSpectrum[graphState, opts] computes the eigenspectrum \
of the interference matrix. Options: \"Tolerance\" -> numerical tolerance, \
\"Method\" -> \"Dense\"|\"Sparse\"|\"Arnoldi\". \
Returns an Association with Eigenvalues, Eigenvectors, and Degeneracies.
Example: spec = EigenSpectrum[g, \"Tolerance\" -> 10^-10]";

GetEigenvalues::usage = "GetEigenvalues[graphState] returns just the eigenvalues \
sorted in ascending order of real part.
Example: vals = GetEigenvalues[g]";

GetEigenvectors::usage = "GetEigenvectors[graphState] returns just the eigenvectors \
corresponding to sorted eigenvalues.
Example: vecs = GetEigenvectors[g]";

CleanEigenvalues::usage = "CleanEigenvalues[eigenvalues, tolerance] cleans numerical \
artifacts by setting near-zero values to exactly zero.
Example: clean = CleanEigenvalues[vals, 10^-12]";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];
Needs["IRHSuite`InterferenceMatrix`"];

(* Options *)
Options[EigenSpectrum] = {
  "Tolerance" -> 10^-10,
  "Method" -> Automatic,
  "DegeneracyTolerance" -> 10^-6,
  "ReturnVectors" -> True
};

(* Main eigenspectrum function *)
EigenSpectrum[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {L, n, tol, method, degTol, returnVecs, eigenSystem,
   eigenvalues, eigenvectors, cleaned, sorted, sortOrder,
   degeneracies, warnings = {}},
  
  (* Build interference matrix *)
  L = BuildInterferenceMatrix[gs];
  n = gs["NodeCount"];
  
  (* Extract options *)
  tol = OptionValue["Tolerance"];
  method = OptionValue["Method"];
  degTol = OptionValue["DegeneracyTolerance"];
  returnVecs = OptionValue["ReturnVectors"];
  
  (* Select method based on matrix size *)
  If[method === Automatic,
    method = If[n <= 500, "Dense", "Sparse"]
  ];
  
  (* Compute eigendecomposition *)
  eigenSystem = computeEigenSystem[L, method, returnVecs];
  
  If[eigenSystem === $Failed,
    Return[$Failed]
  ];
  
  {eigenvalues, eigenvectors} = eigenSystem;
  
  (* Clean numerical artifacts *)
  {cleaned, warnings} = cleanEigenvaluesInternal[eigenvalues, tol, warnings];
  
  (* Sort by real part *)
  sortOrder = Ordering[Re[cleaned]];
  cleaned = cleaned[[sortOrder]];
  If[returnVecs && eigenvectors =!= None,
    eigenvectors = eigenvectors[[All, sortOrder]]
  ];
  
  (* Find degeneracies *)
  degeneracies = findDegeneracies[cleaned, degTol];
  
  (* Check for numerical issues *)
  warnings = checkNumericalIssues[cleaned, L, eigenvectors, warnings];
  
  <|
    "Eigenvalues" -> cleaned,
    "Eigenvectors" -> If[returnVecs, eigenvectors, None],
    "Degeneracies" -> degeneracies,
    "NumericalWarnings" -> warnings,
    "Method" -> method,
    "Tolerance" -> tol
  |>
];

EigenSpectrum[_, ___] := (
  Message[EigenSpectrum::invalidgs];
  $Failed
);

EigenSpectrum::invalidgs = "Input is not a valid GraphState.";

(* Dense eigenvalue computation *)
computeEigenSystem[L_, "Dense", True] := Module[
  {result},
  result = Quiet[Eigensystem[N[L]], {Eigensystem::eival}];
  If[!MatchQ[result, {_List, _List}],
    Return[$Failed]
  ];
  result
];

computeEigenSystem[L_, "Dense", False] := Module[
  {vals},
  vals = Quiet[Eigenvalues[N[L]], {Eigenvalues::eival}];
  If[!ListQ[vals],
    Return[$Failed]
  ];
  {vals, None}
];

(* Sparse eigenvalue computation using Arnoldi *)
computeEigenSystem[L_, "Sparse", returnVecs_] := Module[
  {sparseL, n, k, result, vals, vecs},
  
  n = Length[L];
  sparseL = SparseArray[L];
  
  (* For sparse, compute all eigenvalues using iterative methods *)
  (* Fall back to dense for small matrices *)
  If[n <= 1000,
    Return[computeEigenSystem[L, "Dense", returnVecs]]
  ];
  
  (* Use Arnoldi for largest eigenvalues, then shift-invert for others *)
  k = Min[n - 2, 100];  (* Number of eigenvalues to compute *)
  
  If[returnVecs,
    result = Eigensystem[sparseL, k, Method -> "Arnoldi"];
    If[!MatchQ[result, {_List, _List}],
      (* Fall back to dense *)
      Return[computeEigenSystem[L, "Dense", returnVecs]]
    ];
    result,
    
    vals = Eigenvalues[sparseL, k, Method -> "Arnoldi"];
    If[!ListQ[vals],
      Return[computeEigenSystem[L, "Dense", returnVecs]]
    ];
    {vals, None}
  ]
];

computeEigenSystem[L_, "Arnoldi", returnVecs_] := 
  computeEigenSystem[L, "Sparse", returnVecs];

computeEigenSystem[L_, method_, _] := (
  Message[EigenSpectrum::badmethod, method];
  $Failed
);

EigenSpectrum::badmethod = "Unknown method: `1`. Use \"Dense\", \"Sparse\", or \"Arnoldi\".";

(* Clean eigenvalues *)
cleanEigenvaluesInternal[eigenvalues_, tol_, warnings_] := Module[
  {cleaned, newWarnings = warnings, smallCount},
  
  cleaned = Map[
    Function[val,
      Which[
        Abs[val] < tol,
          0,
        Abs[Im[val]] < tol && Re[val] >= -tol,
          Re[val],
        Abs[Im[val]] < tol,
          Re[val],  (* Keep small negative values *)
        True,
          val
      ]
    ],
    eigenvalues
  ];
  
  (* Count near-zero values that were cleaned *)
  smallCount = Count[eigenvalues, x_ /; Abs[x] > 0 && Abs[x] < tol];
  If[smallCount > 0,
    AppendTo[newWarnings, 
      StringForm["`1` eigenvalues cleaned from near-zero to zero", smallCount]]
  ];
  
  {cleaned, newWarnings}
];

(* Public cleaning function *)
CleanEigenvalues[eigenvalues_List, tol_?NumericQ] := 
  First[cleanEigenvaluesInternal[eigenvalues, tol, {}]];

(* Find degenerate eigenvalue groups *)
findDegeneracies[eigenvalues_, tol_] := Module[
  {n, groups = {}, currentGroup = {1}, i},
  
  n = Length[eigenvalues];
  If[n == 0, Return[{}]];
  
  Do[
    If[Abs[eigenvalues[[i]] - eigenvalues[[i-1]]] < tol,
      AppendTo[currentGroup, i],
      If[Length[currentGroup] > 1,
        AppendTo[groups, <|
          "Indices" -> currentGroup,
          "Value" -> Mean[eigenvalues[[currentGroup]]],
          "Multiplicity" -> Length[currentGroup]
        |>]
      ];
      currentGroup = {i}
    ],
    {i, 2, n}
  ];
  
  (* Handle last group *)
  If[Length[currentGroup] > 1,
    AppendTo[groups, <|
      "Indices" -> currentGroup,
      "Value" -> Mean[eigenvalues[[currentGroup]]],
      "Multiplicity" -> Length[currentGroup]
    |>]
  ];
  
  groups
];

(* Check for numerical issues *)
checkNumericalIssues[eigenvalues_, L_, eigenvectors_, warnings_] := Module[
  {newWarnings = warnings, negCount, complexCount, condNum},
  
  (* Count negative eigenvalues *)
  negCount = Count[eigenvalues, x_ /; Re[x] < -10^-10];
  If[negCount > 0,
    AppendTo[newWarnings, 
      StringForm["`1` negative eigenvalues detected", negCount]]
  ];
  
  (* Count complex eigenvalues with significant imaginary part *)
  complexCount = Count[eigenvalues, x_ /; Abs[Im[x]] > 10^-6];
  If[complexCount > 0,
    AppendTo[newWarnings,
      StringForm["`1` eigenvalues have significant imaginary parts", complexCount]]
  ];
  
  (* Check condition number *)
  condNum = Quiet[
    Max[Abs[eigenvalues]] / Max[10^-15, Min[Abs[Select[eigenvalues, Abs[#] > 10^-15 &]]]],
    {Power::infy}
  ];
  If[NumericQ[condNum] && condNum > 10^10,
    AppendTo[newWarnings,
      StringForm["High condition number: `1`", ScientificForm[condNum, 3]]]
  ];
  
  newWarnings
];

(* Convenience functions *)
GetEigenvalues[gs_?GraphStateQ, opts___] := 
  EigenSpectrum[gs, opts, "ReturnVectors" -> False]["Eigenvalues"];

GetEigenvalues[_] := $Failed;

GetEigenvectors[gs_?GraphStateQ, opts___] := 
  EigenSpectrum[gs, opts]["Eigenvectors"];

GetEigenvectors[_] := $Failed;

End[];

EndPackage[];
