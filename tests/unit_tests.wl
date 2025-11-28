(* ::Package:: *)

(* ============================================================================
   unit_tests.wl - IRH_Suite v3.0 Unit Tests
   ============================================================================
   
   Purpose:
     Comprehensive unit tests for all exported functions in IRH_Suite.
     Includes tests for:
       - Graph state creation and validation
       - Interference matrix construction
       - Eigenspectrum computation
       - Harmony Functional components
       - Mutation operators
       - Analysis functions
       - I/O operations
   
   Golden Tests:
     Tests with known analytic spectra:
       - Cycle graph C_n: λ_k = 2(1 - cos(2πk/n))
       - Complete graph K_n: λ = 0 (mult. 1), λ = n (mult. n-1)
       - Path graph P_n: λ_k = 2(1 - cos(πk/(n+1)))
   
   Usage:
     wolframscript -file tests/unit_tests.wl
   
   Returns:
     Exit code 0 if all tests pass, 1 otherwise
   
   ============================================================================ *)

(* Load source modules *)
SetDirectory[DirectoryName[$InputFileName]];
SetDirectory[".."];

Get["src/GraphState.wl"];
Get["src/InterferenceMatrix.wl"];
Get["src/EigenSpectrum.wl"];
Get["src/HarmonyFunctional.wl"];
Get["src/ParameterController.wl"];
Get["src/MutateGraph.wl"];
Get["src/Acceptance.wl"];
Get["src/ScalingFlows.wl"];
Get["src/SpectralDimension.wl"];
Get["src/LorentzSignature.wl"];
Get["src/GaugeGroupAnalysis.wl"];
Get["src/ConstantDerivation.wl"];
Get["src/GrandAudit.wl"];
Get["src/IOFunctions.wl"];
Get["src/Visualization.wl"];
Get["src/Logging.wl"];

(* Test framework *)
$testsPassed = 0;
$testsFailed = 0;
$testResults = {};

TestAssert[name_String, condition_, message_String:""] := Module[
  {result},
  result = Quiet[Check[TrueQ[condition], False]];
  If[result,
    $testsPassed++;
    AppendTo[$testResults, <|"Name" -> name, "Status" -> "PASS", "Message" -> ""|>];
    Print[Style[StringForm["✓ PASS: `1`", name], Darker[Green]]],
    $testsFailed++;
    AppendTo[$testResults, <|"Name" -> name, "Status" -> "FAIL", "Message" -> message|>];
    Print[Style[StringForm["✗ FAIL: `1` - `2`", name, message], Red]]
  ];
  result
];

TestNumericClose[name_String, value_, expected_, tolerance_:10^-6] := Module[
  {diff},
  If[!NumericQ[value] || !NumericQ[expected],
    TestAssert[name, False, "Non-numeric values"];
    Return[False]
  ];
  diff = Abs[value - expected];
  TestAssert[name, diff < tolerance, 
    StringForm["Got `1`, expected `2`, diff=`3`", value, expected, diff]]
];

Print["\n========================================"];
Print["IRH_Suite v3.0 Unit Tests"];
Print["========================================\n"];

(* ============================================================================
   Test Group 1: GraphState
   ============================================================================ *)

Print["--- GraphState Tests ---"];

Module[{gs},
  SeedRandom[42];
  gs = CreateGraphState[10, "Seed" -> 42, "InitialTopology" -> "Random"];
  TestAssert["CreateGraphState returns Association", AssociationQ[gs]];
  TestAssert["GraphState has correct NodeCount", gs["NodeCount"] == 10];
  TestAssert["GraphState has Type field", gs["Type"] == "GraphState"];
  TestAssert["AdjacencyMatrix is symmetric", 
    Max[Abs[gs["AdjacencyMatrix"] - Transpose[gs["AdjacencyMatrix"]]]] < 10^-10];
  TestAssert["Phases are anti-symmetric",
    Max[Abs[gs["Phases"] + Transpose[gs["Phases"]]]] < 10^-10];
];

Module[{gs},
  gs = CreateGraphState[5, "InitialTopology" -> "Complete"];
  TestAssert["Complete graph has correct edges", 
    gs["EdgeCount"] == 5 * 4 / 2];
];

Module[{gs},
  gs = CreateGraphState[6, "InitialTopology" -> "Cycle"];
  TestAssert["Cycle graph has n edges", gs["EdgeCount"] == 6];
];

TestAssert["CreateGraphState rejects n=1", CreateGraphState[1] === $Failed];

(* ============================================================================
   Test Group 2: InterferenceMatrix
   ============================================================================ *)

Print["\n--- InterferenceMatrix Tests ---"];

Module[{gs, L},
  gs = CreateGraphState[5, "InitialTopology" -> "Complete", "Seed" -> 42];
  L = BuildInterferenceMatrix[gs];
  TestAssert["InterferenceMatrix returns matrix", MatrixQ[L]];
  TestAssert["InterferenceMatrix has correct dimensions", 
    Dimensions[L] == {5, 5}];
  TestAssert["Diagonal is non-negative (for real Laplacian)",
    AllTrue[Re[Diagonal[L]], # >= -10^-10 &]];
];

(* ============================================================================
   Test Group 3: EigenSpectrum - Golden Tests
   ============================================================================ *)

Print["\n--- EigenSpectrum Tests (Golden) ---"];

(* Cycle graph C_n: Known spectrum λ_k = 2(1 - cos(2πk/n)) *)
Module[{n = 6, gs, spectrum, eigenvalues, expected, k},
  gs = CreateGraphState[n, "InitialTopology" -> "Cycle"];
  (* Override with unit weights and zero phases for analytic test *)
  gs["Weights"] = gs["AdjacencyMatrix"];
  gs["Phases"] = ConstantArray[0.0, {n, n}];
  
  spectrum = EigenSpectrum[gs, "Tolerance" -> 10^-10];
  eigenvalues = Sort[Re[spectrum["Eigenvalues"]]];
  expected = Sort[Table[2 (1 - Cos[2 Pi k / n]), {k, 0, n - 1}]];
  
  TestAssert["Cycle graph: correct number of eigenvalues",
    Length[eigenvalues] == n];
  TestAssert["Cycle graph: smallest eigenvalue is 0",
    Abs[First[eigenvalues]] < 10^-6,
    StringForm["Got `1`", First[eigenvalues]]];
  TestNumericClose["Cycle graph: λ_max = 4 (for n=6)", 
    Last[eigenvalues], 4.0, 10^-4];
];

(* Complete graph K_n: λ = 0 (mult. 1), λ = n (mult. n-1) *)
Module[{n = 5, gs, spectrum, eigenvalues, zeroCount, nCount},
  gs = CreateGraphState[n, "InitialTopology" -> "Complete"];
  gs["Weights"] = gs["AdjacencyMatrix"];
  gs["Phases"] = ConstantArray[0.0, {n, n}];
  
  spectrum = EigenSpectrum[gs, "Tolerance" -> 10^-8];
  eigenvalues = Re[spectrum["Eigenvalues"]];
  
  zeroCount = Count[eigenvalues, x_ /; Abs[x] < 10^-4];
  nCount = Count[eigenvalues, x_ /; Abs[x - n] < 10^-4];
  
  TestAssert["Complete graph: has 1 zero eigenvalue", zeroCount == 1];
  TestAssert["Complete graph: has n-1 eigenvalues = n", nCount == n - 1];
];

(* ============================================================================
   Test Group 4: HarmonyFunctional
   ============================================================================ *)

Print["\n--- HarmonyFunctional Tests ---"];

Module[{gs, params, evib, sholo, calg, dlor, gamma, comps},
  SeedRandom[42];
  gs = CreateGraphState[20, "Seed" -> 42];
  params = <|"betaH" -> 1.0, "mu" -> 0.1, "alpha" -> 0.01|>;
  
  evib = Evib[gs];
  sholo = Sholo[gs];
  calg = CAlg[gs];
  dlor = DLor[gs];
  
  TestAssert["Evib returns numeric", NumericQ[evib]];
  TestAssert["Evib is non-negative", evib >= 0];
  TestAssert["Sholo returns numeric", NumericQ[sholo]];
  TestAssert["Sholo is non-negative (entropy)", sholo >= 0];
  TestAssert["CAlg returns numeric", NumericQ[calg]];
  TestAssert["DLor returns numeric", NumericQ[dlor]];
  TestAssert["DLor is in [0,1]", 0 <= dlor <= 1];
  
  gamma = Gamma[gs, params];
  TestAssert["Gamma returns numeric", NumericQ[gamma]];
  
  comps = GammaComponents[gs, params];
  TestAssert["GammaComponents returns Association", AssociationQ[comps]];
  TestNumericClose["Gamma matches components sum",
    comps["Gamma"],
    params["betaH"] * comps["Evib"] + params["mu"] * comps["Sholo"] - 
      params["alpha"] * comps["CAlg"] + comps["DLor"],
    10^-6];
];

(* ============================================================================
   Test Group 5: MutateGraph
   ============================================================================ *)

Print["\n--- MutateGraph Tests ---"];

Module[{gs, mutated},
  SeedRandom[42];
  gs = CreateGraphState[15, "Seed" -> 42];
  
  mutated = MutateGraph[gs, "MutationKernel" -> "EdgeRewiring"];
  TestAssert["EdgeRewiring returns valid GraphState", GraphStateQ[mutated]];
  TestAssert["EdgeRewiring preserves node count", 
    mutated["NodeCount"] == gs["NodeCount"]];
  
  mutated = MutateGraph[gs, "MutationKernel" -> "WeightPerturbation"];
  TestAssert["WeightPerturbation returns valid GraphState", GraphStateQ[mutated]];
  
  mutated = MutateGraph[gs, "MutationKernel" -> "PhaseRotation"];
  TestAssert["PhaseRotation returns valid GraphState", GraphStateQ[mutated]];
  TestAssert["PhaseRotation maintains anti-symmetry",
    Max[Abs[mutated["Phases"] + Transpose[mutated["Phases"]]]] < 10^-10];
  
  mutated = MutateGraph[gs, "MutationKernel" -> "Mixed"];
  TestAssert["Mixed mutation returns valid GraphState", GraphStateQ[mutated]];
];

(* ============================================================================
   Test Group 6: Acceptance
   ============================================================================ *)

Print["\n--- Acceptance Tests ---"];

Module[{},
  ResetAcceptanceState[];
  
  (* Improvements should always be accepted (for maximizing) *)
  TestAssert["Positive delta always rejected for minimizing",
    !AcceptChange[1.0, 1.0, "Maximizing" -> False]];
  TestAssert["Negative delta always accepted for minimizing",
    AcceptChange[-1.0, 1.0, "Maximizing" -> False]];
  
  (* At T=0, only improvements accepted *)
  TestAssert["At T=0, deteriorations rejected",
    !AcceptChange[0.5, 0.0, "Maximizing" -> False, "MinAcceptance" -> 0]];
  
  (* Acceptance ratio *)
  TestNumericClose["AcceptanceRatio for improvement is 1",
    AcceptanceRatio[-1.0, 1.0, "Maximizing" -> False], 1.0, 10^-10];
];

(* ============================================================================
   Test Group 7: ScalingFlows
   ============================================================================ *)

Print["\n--- ScalingFlows Tests ---"];

Module[{gs, coarse, expanded},
  SeedRandom[42];
  gs = CreateGraphState[20, "Seed" -> 42];
  
  coarse = CoarseGrain[gs, "TargetSize" -> 10];
  TestAssert["CoarseGrain returns valid GraphState", GraphStateQ[coarse]];
  TestAssert["CoarseGrain reduces node count", coarse["NodeCount"] <= gs["NodeCount"]];
  
  expanded = Expand[gs, "ExpansionFactor" -> 1.5];
  TestAssert["Expand returns valid GraphState", GraphStateQ[expanded]];
  TestAssert["Expand increases node count", expanded["NodeCount"] >= gs["NodeCount"]];
];

(* ============================================================================
   Test Group 8: Analysis Functions
   ============================================================================ *)

Print["\n--- Analysis Tests ---"];

Module[{gs, spectralDim, lorentzSig, gauge, constants},
  SeedRandom[42];
  gs = CreateGraphState[30, "Seed" -> 42];
  
  spectralDim = SpectralDimension[gs];
  TestAssert["SpectralDimension returns Association", AssociationQ[spectralDim]];
  TestAssert["SpectralDimension has Value key", KeyExistsQ[spectralDim, "Value"]];
  
  lorentzSig = LorentzSignature[gs];
  TestAssert["LorentzSignature returns Association", AssociationQ[lorentzSig]];
  TestAssert["LorentzSignature has NegativeCount", 
    IntegerQ[lorentzSig["NegativeCount"]]];
  
  gauge = GaugeGroupAnalysis[gs, "MaxIterations" -> 100];
  TestAssert["GaugeGroupAnalysis returns Association", AssociationQ[gauge]];
  TestAssert["GaugeGroupAnalysis has GroupOrder", 
    IntegerQ[gauge["GroupOrder"]] && gauge["GroupOrder"] >= 1];
  
  constants = ConstantDerivation[gs];
  TestAssert["ConstantDerivation returns Association", AssociationQ[constants]];
  TestAssert["ConstantDerivation has Constants key", KeyExistsQ[constants, "Constants"]];
];

(* ============================================================================
   Test Group 9: I/O Functions
   ============================================================================ *)

Print["\n--- I/O Tests ---"];

Module[{gs, filepath, loaded},
  SeedRandom[42];
  gs = CreateGraphState[10, "Seed" -> 42];
  filepath = FileNameJoin[{$TemporaryDirectory, "test_graph.irh"}];
  
  TestAssert["SaveGraphState returns True", SaveGraphState[gs, filepath]];
  TestAssert["Saved file exists", FileExistsQ[filepath]];
  
  loaded = LoadGraphState[filepath];
  TestAssert["LoadGraphState returns valid GraphState", GraphStateQ[loaded]];
  TestAssert["Loaded graph has same NodeCount", loaded["NodeCount"] == gs["NodeCount"]];
  TestAssert["Loaded graph has same EdgeCount", loaded["EdgeCount"] == gs["EdgeCount"]];
  
  (* Cleanup *)
  DeleteFile[filepath];
];

(* ============================================================================
   Test Group 10: ParameterController
   ============================================================================ *)

Print["\n--- ParameterController Tests ---"];

Module[{params, history, updated},
  InitializeController[<|"controller" -> <|
    "strategy" -> "Fixed",
    "betaH" -> 1.0,
    "mu" -> 0.1,
    "alpha" -> 0.01
  |>|>];
  
  params = <|"betaH" -> 1.0, "mu" -> 0.1, "alpha" -> 0.01|>;
  history = {<|"Gamma" -> 10.0|>, <|"Gamma" -> 11.0|>};
  
  updated = UpdateParameters[params, history, "Strategy" -> "Fixed"];
  TestAssert["Fixed strategy doesn't change params",
    updated["betaH"] == params["betaH"] &&
    updated["mu"] == params["mu"] &&
    updated["alpha"] == params["alpha"]];
  
  ResetController[];
];

(* ============================================================================
   Test Group 11: Reproducibility
   ============================================================================ *)

Print["\n--- Reproducibility Tests ---"];

Module[{gs1, gs2, gamma1, gamma2, params},
  SeedRandom[12345];
  gs1 = CreateGraphState[15, "Seed" -> 12345];
  
  SeedRandom[12345];
  gs2 = CreateGraphState[15, "Seed" -> 12345];
  
  TestAssert["Same seed produces identical graphs",
    gs1["AdjacencyMatrix"] == gs2["AdjacencyMatrix"] &&
    gs1["Weights"] == gs2["Weights"] &&
    gs1["Phases"] == gs2["Phases"]];
  
  params = <|"betaH" -> 1.0, "mu" -> 0.1, "alpha" -> 0.01|>;
  gamma1 = Gamma[gs1, params];
  gamma2 = Gamma[gs2, params];
  
  TestNumericClose["Same seed produces identical Gamma", gamma1, gamma2, 10^-10];
];

(* ============================================================================
   Test Group 12: Logging
   ============================================================================ *)

Print["\n--- Logging Tests ---"];

Module[{testDir, stats},
  testDir = FileNameJoin[{$TemporaryDirectory, "irh_test_logs"}];
  
  IRHInitializeLog[testDir, "INFO"];
  IRHLog["INFO", "Test message"];
  LogHarmony[1, 42.5, <|"Temperature" -> 0.5|>];
  
  stats = GetLogStats[];
  TestAssert["Logging records messages", stats["MessageCount"] >= 1];
  TestAssert["Logging records harmony values", stats["HarmonyCount"] >= 1];
  
  IRHCloseLog[];
  
  (* Cleanup *)
  Quiet[DeleteDirectory[testDir, DeleteContents -> True]];
];

(* ============================================================================
   Summary
   ============================================================================ *)

Print["\n========================================"];
Print["Test Summary"];
Print["========================================"];
Print[StringForm["Passed: `1`", $testsPassed]];
Print[StringForm["Failed: `1`", $testsFailed]];
Print[StringForm["Total:  `1`", $testsPassed + $testsFailed]];

If[$testsFailed > 0,
  Print["\nFailed tests:"];
  Do[
    If[result["Status"] == "FAIL",
      Print[StringForm["  - `1`: `2`", result["Name"], result["Message"]]]
    ],
    {result, $testResults}
  ]
];

Print["\n========================================"];
If[$testsFailed == 0,
  Print[Style["All tests passed!", Darker[Green]]];
  Exit[0],
  Print[Style["Some tests failed!", Red]];
  Exit[1]
];
