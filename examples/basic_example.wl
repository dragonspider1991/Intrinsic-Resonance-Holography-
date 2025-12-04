(* ::Package:: *)

(* ============================================================================
   IRH_Suite v3.0 - Basic Example
   ============================================================================
   
   This example demonstrates the basic usage of IRH_Suite:
   1. Creating a graph state
   2. Computing the Harmony Functional
   3. Running a short optimization
   4. Analyzing results
   
   Usage:
     wolframscript -file examples/basic_example.wl
   
   ============================================================================ *)

SetDirectory[DirectoryName[$InputFileName]];
SetDirectory[".."];

(* Load all modules *)
Get["src/GraphState.wl"];
Get["src/InterferenceMatrix.wl"];
Get["src/EigenSpectrum.wl"];
Get["src/HarmonyFunctional.wl"];
Get["src/MutateGraph.wl"];
Get["src/Acceptance.wl"];
Get["src/ParameterController.wl"];
Get["src/ScalingFlows.wl"];
Get["src/AROEngine.wl"];
Get["src/SpectralDimension.wl"];
Get["src/LorentzSignature.wl"];
Get["src/IOFunctions.wl"];
Get["src/Logging.wl"];

Print["========================================"];
Print["IRH_Suite v3.0 - Basic Example"];
Print["========================================\n"];

(* Set seed for reproducibility *)
SeedRandom[42];

(* Create a small graph state *)
Print["1. Creating graph state with 30 nodes..."];
gs = CreateGraphState[30, 
  "Seed" -> 42, 
  "InitialTopology" -> "Random",
  "EdgeProbability" -> 0.3
];
Print["   Created graph with ", gs["NodeCount"], " nodes and ", gs["EdgeCount"], " edges.\n"];

(* Compute initial Harmony Functional *)
Print["2. Computing Harmony Functional..."];
params = <|"betaH" -> 1.0, "mu" -> 0.1, "alpha" -> 0.01|>;
comps = GammaComponents[gs, params];

Print["   Components:"];
Print["     Evib  = ", comps["Evib"]];
Print["     Sholo = ", comps["Sholo"]];
Print["     CAlg  = ", comps["CAlg"]];
Print["     DLor  = ", comps["DLor"]];
Print["     Γ     = ", comps["Gamma"], "\n"];

(* Run a short optimization *)
Print["3. Running ARO optimization (100 iterations)..."];
IRHInitializeLog[$TemporaryDirectory, "WARNING"];  (* Suppress most output *)

result = HAGOEngine[gs,
  "MaxIterations" -> 100,
  "CheckpointInterval" -> 50,
  "Temperature" -> <|"initial" -> 1.0, "final" -> 0.1, "schedule" -> "exponential"|>,
  "Seed" -> 42
];

Print["   Optimization complete!"];
Print["   Initial Γ: ", First[result["History"]]["Gamma"]];
Print["   Final Γ:   ", result["FinalGamma"]];
Print["   Iterations: ", result["TotalIterations"], "\n"];

IRHCloseLog[];

(* Analyze the optimized graph *)
Print["4. Analyzing optimized graph..."];
optGraph = result["OptimizedGraph"];

spectralDim = SpectralDimension[optGraph];
Print["   Spectral dimension: ", spectralDim["Value"], " ± ", spectralDim["Error"]];

lorentzSig = LorentzSignature[optGraph];
Print["   Lorentz signature: ", lorentzSig["Signature"]];
Print["   Negative eigenvalues: ", lorentzSig["NegativeCount"]];

(* Save the result *)
Print["\n5. Saving optimized graph..."];
outputPath = FileNameJoin[{$TemporaryDirectory, "example_opt.irh"}];
SaveGraphState[optGraph, outputPath];
Print["   Saved to: ", outputPath];

Print["\n========================================"];
Print["Example complete!"];
Print["========================================"];
