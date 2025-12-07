(* ::Package:: *)

(* ============================================================================
   IRH_Suite v3.0 - Main Entry Point
   ============================================================================
   
   Purpose:
     Main orchestration script for Intrinsic Resonance Holography Suite.
     Initializes configuration, runs the ARO optimization loop, and
     generates all output artifacts.
   
   Inputs:
     - project_config.json: Configuration file with all parameters
     - CLI arguments (optional): Override config values
   
   Outputs:
     - G_opt.irh: Optimized graph state
     - spectral_dimension_report.json: Analysis results
     - grand_audit_report.pdf: CODATA/PDG comparison
     - log_harmony.csv: Evolution log
   
   Usage:
     wolframscript -file main.wl
     wolframscript -file main.wl -seed 123 -maxIterations 500
   
   References:
     - IRH Theory Papers (see /docs/)
     - Harmony Functional: Γ = βH·Evib + μ·Sholo - α·CAlg + DLor
   
   ============================================================================ *)

(* Load all source modules *)
Get["src/GraphState.wl"];
Get["src/InterferenceMatrix.wl"];
Get["src/EigenSpectrum.wl"];
Get["src/HarmonyFunctional.wl"];
Get["src/ParameterController.wl"];
Get["src/MutateGraph.wl"];
Get["src/Acceptance.wl"];
Get["src/ScalingFlows.wl"];
Get["src/AROEngine.wl"];
Get["src/SpectralDimension.wl"];
Get["src/LorentzSignature.wl"];
Get["src/GaugeGroupAnalysis.wl"];
Get["src/ConstantDerivation.wl"];
Get["src/GrandAudit.wl"];
Get["src/IOFunctions.wl"];
Get["src/Visualization.wl"];
Get["src/Logging.wl"];

(* ============================================================================
   Configuration Loading
   ============================================================================ *)

IRHLoadConfig::usage = "IRHLoadConfig[configPath] loads configuration from JSON file.";
IRHLoadConfig[configPath_String] := Module[
  {config},
  If[!FileExistsQ[configPath],
    Print["Error: Configuration file not found: ", configPath];
    Return[$Failed]
  ];
  config = Import[configPath, "RawJSON"];
  If[config === $Failed,
    Print["Error: Failed to parse configuration file"];
    Return[$Failed]
  ];
  config
];

(* Parse command line arguments *)
IRHParseCLI::usage = "IRHParseCLI[] parses command line arguments and returns overrides.";
IRHParseCLI[] := Module[
  {args, overrides = <||>, i = 1},
  args = $ScriptCommandLine;
  While[i <= Length[args],
    Switch[args[[i]],
      "-seed", 
        overrides["seed"] = ToExpression[args[[++i]]],
      "-maxIterations",
        overrides["maxIterations"] = ToExpression[args[[++i]]],
      "-precision",
        overrides["precision"] = ToExpression[args[[++i]]],
      "-outputDir",
        overrides["outputDir"] = args[[++i]],
      "-logLevel",
        overrides["logLevel"] = args[[++i]],
      "-graphSize",
        overrides["graphSize"] = ToExpression[args[[++i]]]
    ];
    i++
  ];
  overrides
];

(* Merge CLI overrides with config *)
IRHMergeConfig[config_Association, overrides_Association] := 
  Merge[{config, overrides}, Last];

(* ============================================================================
   Main Execution
   ============================================================================ *)

IRHMain::usage = "IRHMain[] runs the complete IRH Suite optimization and analysis pipeline.";
IRHMain[] := Module[
  {config, cliOverrides, startTime, initGraph, result, optGraph,
   spectralDim, lorentzSig, gaugeAnalysis, constants, auditResult,
   manifest},
  
  Print["========================================"];
  Print["IRH_Suite v3.0 - Starting Execution"];
  Print["========================================\n"];
  
  (* Load and merge configuration *)
  Print["Loading configuration..."];
  config = IRHLoadConfig["project_config.json"];
  If[config === $Failed, Return[$Failed]];
  
  cliOverrides = IRHParseCLI[];
  config = IRHMergeConfig[config, cliOverrides];
  
  Print["Configuration loaded successfully."];
  Print["  Version: ", config["version"]];
  Print["  Seed: ", config["seed"]];
  Print["  Graph Size: ", config["graphSize"]];
  Print["  Max Iterations: ", config["maxIterations"]];
  Print["\n"];
  
  (* Initialize RNG with seed for reproducibility *)
  SeedRandom[config["seed"]];
  
  (* Set numerical precision *)
  $MinPrecision = config["precision"];
  
  (* Create output directory if needed *)
  If[!DirectoryQ[config["outputDir"]],
    CreateDirectory[config["outputDir"]]
  ];
  
  (* Initialize logging *)
  IRHInitializeLog[config["outputDir"], config["logLevel"]];
  
  startTime = AbsoluteTime[];
  
  (* ========================================
     Phase 1: Initialize Graph State
     ======================================== *)
  Print["Phase 1: Creating initial graph state..."];
  initGraph = CreateGraphState[config["graphSize"], 
    "Seed" -> config["seed"],
    "Precision" -> config["precision"]
  ];
  If[initGraph === $Failed,
    Print["Error: Failed to create initial graph state"];
    Return[$Failed]
  ];
  Print["  Initial graph created with ", config["graphSize"], " nodes.\n"];
  
  (* ========================================
     Phase 2: Run ARO Optimization
     ======================================== *)
  Print["Phase 2: Running ARO optimization..."];
  result = HAGOEngine[initGraph,
    "MaxIterations" -> config["maxIterations"],
    "CheckpointInterval" -> config["checkpointInterval"],
    "Temperature" -> config["temperature"],
    "Optimizer" -> config["optimizer"],
    "Controller" -> config["controller"],
    "OutputDir" -> config["outputDir"],
    "LogLevel" -> config["logLevel"]
  ];
  
  If[result === $Failed,
    Print["Error: ARO optimization failed"];
    Return[$Failed]
  ];
  
  optGraph = result["OptimizedGraph"];
  Print["  Optimization complete."];
  Print["  Final Γ: ", result["FinalGamma"]];
  Print["  Iterations: ", result["TotalIterations"], "\n"];
  
  (* ========================================
     Phase 3: Analysis
     ======================================== *)
  Print["Phase 3: Running analysis..."];
  
  (* Spectral Dimension *)
  Print["  Computing spectral dimension..."];
  spectralDim = SpectralDimension[optGraph, 
    "FitRange" -> config["analysis"]["spectralDimensionFitRange"]
  ];
  Print["    d_s = ", spectralDim["Value"], " ± ", spectralDim["Error"]];
  
  (* Lorentz Signature *)
  Print["  Computing Lorentz signature..."];
  lorentzSig = LorentzSignature[optGraph,
    "Tolerance" -> config["analysis"]["eigenvalueTolerance"]
  ];
  Print["    Negative eigenvalues: ", lorentzSig["NegativeCount"]];
  Print["    Signature: ", lorentzSig["Signature"]];
  
  (* Gauge Group Analysis *)
  Print["  Analyzing gauge group structure..."];
  gaugeAnalysis = GaugeGroupAnalysis[optGraph,
    "MaxIterations" -> config["analysis"]["automorphismMaxIterations"]
  ];
  Print["    Automorphism group order: ", gaugeAnalysis["GroupOrder"]];
  Print["    Candidate Lie groups: ", gaugeAnalysis["Candidates"]];
  
  (* Constant Derivation *)
  Print["  Deriving physical constants..."];
  constants = ConstantDerivation[optGraph];
  Print["    Derived ", Length[constants["Constants"]], " physical constants.\n"];
  
  (* ========================================
     Phase 4: Grand Audit
     ======================================== *)
  Print["Phase 4: Running Grand Audit..."];
  auditResult = GrandAudit[optGraph, 
    <|
      "SpectralDimension" -> spectralDim,
      "LorentzSignature" -> lorentzSig,
      "GaugeGroups" -> gaugeAnalysis,
      "Constants" -> constants
    |>,
    "OutputDir" -> config["outputDir"]
  ];
  Print["  Audit complete: ", auditResult["PassCount"], "/", 
        auditResult["TotalChecks"], " checks passed.\n"];
  
  (* ========================================
     Phase 5: Save Outputs
     ======================================== *)
  Print["Phase 5: Saving outputs..."];
  
  (* Save optimized graph *)
  SaveGraphState[optGraph, 
    FileNameJoin[{config["outputDir"], "G_opt.irh"}]
  ];
  Print["  Saved: G_opt.irh"];
  
  (* Export spectral dimension report *)
  Export[FileNameJoin[{config["outputDir"], "spectral_dimension_report.json"}],
    spectralDim, "RawJSON"
  ];
  Print["  Saved: spectral_dimension_report.json"];
  
  (* Close log *)
  IRHCloseLog[];
  Print["  Saved: log_harmony.csv"];
  
  (* Generate manifest *)
  manifest = <|
    "version" -> config["version"],
    "seed" -> config["seed"],
    "timestamp" -> DateString["ISODateTime"],
    "executionTime" -> AbsoluteTime[] - startTime,
    "artifacts" -> {
      <|"file" -> "G_opt.irh", 
        "hash" -> Hash[Export["String", optGraph, "RawJSON"], "SHA256"]|>,
      <|"file" -> "spectral_dimension_report.json", 
        "hash" -> Hash[Export["String", spectralDim, "RawJSON"], "SHA256"]|>,
      <|"file" -> "grand_audit_report.pdf", 
        "hash" -> "generated"|>,
      <|"file" -> "log_harmony.csv", 
        "hash" -> "streaming"|>
    },
    "results" -> <|
      "finalGamma" -> result["FinalGamma"],
      "spectralDimension" -> spectralDim["Value"],
      "negativeEigenvalues" -> lorentzSig["NegativeCount"],
      "auditPassRate" -> N[auditResult["PassCount"]/auditResult["TotalChecks"]]
    |>
  |>;
  
  Export[FileNameJoin[{config["outputDir"], "run_manifest.json"}],
    manifest, "RawJSON"
  ];
  Print["  Saved: run_manifest.json\n"];
  
  (* ========================================
     Complete
     ======================================== *)
  Print["========================================"];
  Print["IRH_Suite v3.0 construction complete."];
  Print["The computational universe is operational."];
  Print["========================================"];
  Print["Total execution time: ", AbsoluteTime[] - startTime, " seconds"];
  
  manifest
];

(* Run main if executed as script *)
If[$ScriptCommandLine =!= {},
  IRHMain[]
];
