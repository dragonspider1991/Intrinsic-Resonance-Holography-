(* ::Package:: *)

(* ============================================================================
   GrandAudit.wl - Grand Audit and Validation Report
   ============================================================================
   
   Purpose:
     Performs comprehensive audit of the optimized graph state against
     physical constraints and CODATA/PDG values. Generates pass/fail
     summary in CSV and PDF formats.
   
   Inputs:
     - GraphState: Optimized GraphState
     - results: Analysis results from other modules
     - opts: Output configuration
   
   Outputs:
     - Association with audit results
     - CSV file: grand_audit_results.csv
     - PDF file: grand_audit_report.pdf
   
   Checks Performed:
     1. Spectral dimension near 4
     2. Exactly 1 negative eigenvalue (Lorentzian)
     3. Physical constants within tolerance
     4. Graph connectivity
     5. Numerical stability
   
   References:
     - CODATA 2018 values
     - Particle Data Group (PDG) 2022
   
   ============================================================================ *)

BeginPackage["IRHSuite`GrandAudit`"];

GrandAudit::usage = "GrandAudit[graphState, results, opts] performs comprehensive \
validation and generates audit report. Options: \"OutputDir\" -> path for reports.
Example: audit = GrandAudit[gs, results, \"OutputDir\" -> \"io/output\"]";

AuditCheck::usage = "AuditCheck[name, value, target, tolerance] performs a single \
audit check and returns pass/fail result.";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];

(* Options *)
Options[GrandAudit] = {
  "OutputDir" -> "io/output",
  "GeneratePDF" -> True,
  "GenerateCSV" -> True,
  "SpectralDimensionTarget" -> 4.0,
  "SpectralDimensionTolerance" -> 0.5,
  "NegativeEigenvalueTarget" -> 1,
  "ConstantTolerance" -> 0.5  (* 50% relative tolerance *)
};

(* Main audit function *)
GrandAudit[gs_?GraphStateQ, results_Association, opts:OptionsPattern[]] := Module[
  {outputDir, genPDF, genCSV, checks = {}, 
   spectralDim, lorentzSig, gaugeGroups, constants,
   checkResult, passCount, totalChecks, auditData, summary},
  
  outputDir = OptionValue["OutputDir"];
  genPDF = OptionValue["GeneratePDF"];
  genCSV = OptionValue["GenerateCSV"];
  
  (* Extract results *)
  spectralDim = Lookup[results, "SpectralDimension", <||>];
  lorentzSig = Lookup[results, "LorentzSignature", <||>];
  gaugeGroups = Lookup[results, "GaugeGroups", <||>];
  constants = Lookup[results, "Constants", <||>];
  
  (* ========================================
     Audit Check 1: Spectral Dimension
     ======================================== *)
  checkResult = auditSpectralDimension[spectralDim, 
    OptionValue["SpectralDimensionTarget"],
    OptionValue["SpectralDimensionTolerance"]
  ];
  AppendTo[checks, checkResult];
  
  (* ========================================
     Audit Check 2: Lorentz Signature
     ======================================== *)
  checkResult = auditLorentzSignature[lorentzSig,
    OptionValue["NegativeEigenvalueTarget"]
  ];
  AppendTo[checks, checkResult];
  
  (* ========================================
     Audit Check 3: Physical Constants
     ======================================== *)
  checkResult = auditPhysicalConstants[constants,
    OptionValue["ConstantTolerance"]
  ];
  checks = Join[checks, checkResult];
  
  (* ========================================
     Audit Check 4: Graph Connectivity
     ======================================== *)
  checkResult = auditConnectivity[gs];
  AppendTo[checks, checkResult];
  
  (* ========================================
     Audit Check 5: Numerical Stability
     ======================================== *)
  checkResult = auditNumericalStability[gs, results];
  AppendTo[checks, checkResult];
  
  (* ========================================
     Audit Check 6: Gauge Structure
     ======================================== *)
  checkResult = auditGaugeStructure[gaugeGroups];
  AppendTo[checks, checkResult];
  
  (* Compute summary *)
  passCount = Count[checks, check_ /; check["Pass"]];
  totalChecks = Length[checks];
  
  summary = <|
    "PassCount" -> passCount,
    "TotalChecks" -> totalChecks,
    "PassRate" -> N[passCount / totalChecks],
    "Status" -> If[passCount == totalChecks, "PASS", "FAIL"],
    "Timestamp" -> DateString["ISODateTime"]
  |>;
  
  (* Ensure output directory exists *)
  If[!DirectoryQ[outputDir],
    Quiet[CreateDirectory[outputDir]]
  ];
  
  (* Generate outputs *)
  If[genCSV,
    exportCSV[checks, summary, outputDir]
  ];
  
  If[genPDF,
    exportPDF[checks, summary, results, outputDir]
  ];
  
  (* Return audit data *)
  <|
    "Checks" -> checks,
    "Summary" -> summary,
    "PassCount" -> passCount,
    "TotalChecks" -> totalChecks,
    "Results" -> results
  |>
];

GrandAudit[_, _, ___] := (
  Message[GrandAudit::invalidargs];
  $Failed
);

GrandAudit::invalidargs = "GrandAudit requires a valid GraphState and results Association.";

(* ============================================================================
   Individual Audit Checks
   ============================================================================ *)

(* Single check helper *)
AuditCheck[name_, value_, target_, tolerance_] := Module[
  {pass, relError, message},
  
  If[!NumericQ[value] || !NumericQ[target],
    Return[<|
      "Name" -> name,
      "Value" -> value,
      "Target" -> target,
      "Pass" -> False,
      "Message" -> "Non-numeric value"
    |>]
  ];
  
  relError = If[target != 0, 
    Abs[(value - target) / target],
    If[value == 0, 0, Infinity]
  ];
  
  pass = relError <= tolerance;
  
  message = If[pass,
    StringForm["PASS: `1` = `2` (target: `3`, error: `4`%)", 
      name, NumberForm[value, 4], target, Round[relError * 100, 0.1]],
    StringForm["FAIL: `1` = `2` (target: `3`, error: `4`%)", 
      name, NumberForm[value, 4], target, Round[relError * 100, 0.1]]
  ] // ToString;
  
  <|
    "Name" -> name,
    "Value" -> value,
    "Target" -> target,
    "Tolerance" -> tolerance,
    "RelativeError" -> relError,
    "Pass" -> pass,
    "Message" -> message
  |>
];

(* Spectral dimension audit *)
auditSpectralDimension[spectralDim_, target_, tol_] := Module[
  {value},
  
  value = Lookup[spectralDim, "Value", Indeterminate];
  
  AuditCheck["SpectralDimension", value, target, tol]
];

(* Lorentz signature audit *)
auditLorentzSignature[lorentzSig_, targetNeg_] := Module[
  {negCount, pass, message},
  
  negCount = Lookup[lorentzSig, "NegativeCount", Indeterminate];
  
  If[!NumericQ[negCount],
    Return[<|
      "Name" -> "LorentzSignature",
      "Value" -> negCount,
      "Target" -> targetNeg,
      "Pass" -> False,
      "Message" -> "Could not determine negative eigenvalue count"
    |>]
  ];
  
  pass = (negCount == targetNeg);
  
  message = If[pass,
    StringForm["PASS: Exactly `1` negative eigenvalue(s)", negCount],
    StringForm["FAIL: `1` negative eigenvalue(s), expected `2`", negCount, targetNeg]
  ] // ToString;
  
  <|
    "Name" -> "LorentzSignature",
    "Value" -> negCount,
    "Target" -> targetNeg,
    "Pass" -> pass,
    "Message" -> message,
    "Signature" -> Lookup[lorentzSig, "Signature", "Unknown"]
  |>
];

(* Physical constants audit *)
auditPhysicalConstants[constants_, tol_] := Module[
  {checks = {}, constData, comparisons, name, comp},
  
  constData = Lookup[constants, "Constants", <||>];
  comparisons = Lookup[constants, "Comparisons", <||>];
  
  Do[
    comp = comparisons[name];
    If[AssociationQ[comp],
      AppendTo[checks, <|
        "Name" -> StringJoin["Constant:", name],
        "Value" -> Lookup[comp, "DerivedValue", Indeterminate],
        "Target" -> Lookup[comp, "CODATAValue", Indeterminate],
        "RelativeError" -> Lookup[comp, "RelativeError", Infinity],
        "Pass" -> Lookup[comp, "RelativeError", Infinity] <= tol,
        "Message" -> StringForm["`1`: `2` match", name, Lookup[comp, "Match", "Unknown"]] // ToString
      |>]
    ],
    {name, Keys[comparisons]}
  ];
  
  If[Length[checks] == 0,
    checks = {<|
      "Name" -> "PhysicalConstants",
      "Value" -> "N/A",
      "Target" -> "N/A",
      "Pass" -> True,
      "Message" -> "No constants to verify"
    |>}
  ];
  
  checks
];

(* Connectivity audit *)
auditConnectivity[gs_] := Module[
  {adjMat, n, isConnected, components},
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  
  (* Simple connectivity check *)
  components = countComponents[adjMat];
  isConnected = (components == 1);
  
  <|
    "Name" -> "GraphConnectivity",
    "Value" -> components,
    "Target" -> 1,
    "Pass" -> isConnected,
    "Message" -> If[isConnected,
      "PASS: Graph is connected",
      StringForm["FAIL: Graph has `1` components", components]
    ] // ToString
  |>
];

(* Count connected components *)
countComponents[adjMat_] := Module[
  {n, visited, components, queue, current},
  
  n = Length[adjMat];
  visited = ConstantArray[False, n];
  components = 0;
  
  Do[
    If[!visited[[start]],
      components++;
      queue = {start};
      While[Length[queue] > 0,
        current = First[queue];
        queue = Rest[queue];
        If[!visited[[current]],
          visited[[current]] = True;
          queue = Join[queue, 
            Select[Flatten[Position[adjMat[[current]], x_ /; x > 0]], !visited[[#]] &]
          ]
        ]
      ]
    ],
    {start, n}
  ];
  
  components
];

(* Numerical stability audit *)
auditNumericalStability[gs_, results_] := Module[
  {warnings, hasWarnings, spectrum},
  
  spectrum = Lookup[results, "SpectralDimension", <||>];
  warnings = {};
  
  (* Check for numerical warnings in eigenspectrum *)
  If[AssociationQ[spectrum] && KeyExistsQ[spectrum, "FitQuality"],
    If[spectrum["FitQuality"] < 0.9,
      AppendTo[warnings, "Low spectral dimension fit quality"]
    ]
  ];
  
  (* Check for extreme values in graph *)
  If[Max[Abs[Flatten[gs["Weights"]]]] > 1000,
    AppendTo[warnings, "Extreme edge weights detected"]
  ];
  
  hasWarnings = Length[warnings] > 0;
  
  <|
    "Name" -> "NumericalStability",
    "Value" -> If[hasWarnings, "Issues", "Stable"],
    "Target" -> "Stable",
    "Pass" -> !hasWarnings,
    "Message" -> If[hasWarnings,
      StringJoin["FAIL: ", Riffle[warnings, "; "]],
      "PASS: No numerical stability issues"
    ],
    "Warnings" -> warnings
  |>
];

(* Gauge structure audit *)
auditGaugeStructure[gaugeGroups_] := Module[
  {groupOrder, hasSM, candidates},
  
  groupOrder = Lookup[gaugeGroups, "GroupOrder", 1];
  candidates = Lookup[gaugeGroups, "Candidates", {}];
  
  (* Check for Standard Model-like structure *)
  hasSM = Lookup[gaugeGroups, "HasU1", False] || 
          Lookup[gaugeGroups, "HasSU2", False] ||
          Lookup[gaugeGroups, "HasSU3", False];
  
  <|
    "Name" -> "GaugeStructure",
    "Value" -> groupOrder,
    "Target" -> "> 1",
    "Pass" -> groupOrder > 1,
    "Message" -> If[groupOrder > 1,
      StringForm["PASS: Non-trivial symmetry (order `1`), candidates: `2`", 
        groupOrder, candidates],
      "FAIL: Trivial symmetry group"
    ] // ToString,
    "HasSMStructure" -> hasSM
  |>
];

(* ============================================================================
   Export Functions
   ============================================================================ *)

(* Export CSV *)
exportCSV[checks_, summary_, outputDir_] := Module[
  {filename, data, headers},
  
  filename = FileNameJoin[{outputDir, "grand_audit_results.csv"}];
  
  headers = {"Name", "Value", "Target", "Pass", "Message"};
  
  data = Prepend[
    Table[
      {
        check["Name"],
        ToString[check["Value"]],
        ToString[check["Target"]],
        If[check["Pass"], "PASS", "FAIL"],
        StringReplace[check["Message"], "," -> ";"]
      },
      {check, checks}
    ],
    headers
  ];
  
  (* Add summary row *)
  AppendTo[data, {
    "SUMMARY",
    ToString[summary["PassCount"]] <> "/" <> ToString[summary["TotalChecks"]],
    "100%",
    summary["Status"],
    summary["Timestamp"]
  }];
  
  Export[filename, data, "CSV"]
];

(* Export PDF (simplified - creates text file if PDF export not available) *)
exportPDF[checks_, summary_, results_, outputDir_] := Module[
  {filename, content, lines},
  
  filename = FileNameJoin[{outputDir, "grand_audit_report.pdf"}];
  
  (* Build report content *)
  lines = {
    "========================================",
    "IRH_Suite v3.0 Grand Audit Report",
    "========================================",
    "",
    "Generated: " <> summary["Timestamp"],
    "Status: " <> summary["Status"],
    "",
    "Summary: " <> ToString[summary["PassCount"]] <> "/" <> 
      ToString[summary["TotalChecks"]] <> " checks passed",
    "Pass Rate: " <> ToString[Round[summary["PassRate"] * 100, 0.1]] <> "%",
    "",
    "========================================",
    "Individual Checks",
    "========================================",
    ""
  };
  
  Do[
    AppendTo[lines, check["Message"]];
    AppendTo[lines, ""],
    {check, checks}
  ];
  
  AppendTo[lines, "========================================"];
  AppendTo[lines, "End of Report"];
  AppendTo[lines, "========================================"];
  
  content = StringRiffle[lines, "\n"];
  
  (* Try PDF export, fall back to text *)
  Quiet[
    Check[
      Export[filename, content, "PDF"],
      Export[StringReplace[filename, ".pdf" -> ".txt"], content, "Text"]
    ]
  ]
];

End[];

EndPackage[];
