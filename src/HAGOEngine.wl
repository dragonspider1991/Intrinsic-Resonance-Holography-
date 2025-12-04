(* ::Package:: *)

(* ============================================================================
   HAGOEngine.wl - Harmony-Guided Adaptive Graph Optimization Engine
   ============================================================================
   
   Purpose:
     Orchestrates the main ARO optimization loop, coordinating mutation,
     acceptance, parameter updates, checkpointing, and logging. This is
     the central engine that drives IRH spacetime evolution.
   
   Inputs:
     - initGraph: Initial GraphState
     - opts: Configuration options including MaxIterations, Temperature, etc.
   
   Outputs:
     - Association containing:
       * "OptimizedGraph": Final optimized GraphState
       * "FinalGamma": Final Harmony Functional value
       * "History": Complete optimization history
       * "TotalIterations": Number of iterations completed
   
   Algorithm:
     1. Initialize: Set parameters, temperature, logging
     2. Loop for maxIterations:
        a. Mutate current graph
        b. Compute new Gamma
        c. Accept/reject via Metropolis
        d. Update parameters (adaptive)
        e. Anneal temperature
        f. Log and checkpoint
     3. Return best configuration
   
   References:
     - Simulated Annealing
     - Metropolis-Hastings MCMC
     - IRH Theory: Harmony Functional optimization
   
   ============================================================================ *)

BeginPackage["IRHSuite`HAGOEngine`"];

HAGOEngine::usage = "HAGOEngine[initGraph, opts] runs the ARO optimization loop. \
Options: \"MaxIterations\", \"Temperature\", \"CheckpointInterval\", etc. \
Returns an Association with the optimized graph and optimization history.
Example: result = HAGOEngine[g, \"MaxIterations\" -> 1000]";

HAGOStep::usage = "HAGOStep[state] performs a single ARO iteration. \
Used internally by HAGOEngine but exposed for debugging.";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];
Needs["IRHSuite`HarmonyFunctional`"];
Needs["IRHSuite`MutateGraph`"];
Needs["IRHSuite`Acceptance`"];
Needs["IRHSuite`ParameterController`"];
Needs["IRHSuite`Logging`"];

(* Options *)
Options[HAGOEngine] = {
  "MaxIterations" -> 1000,
  "CheckpointInterval" -> 100,
  "Temperature" -> <|
    "initial" -> 1.0,
    "final" -> 0.01,
    "schedule" -> "exponential"
  |>,
  "Optimizer" -> <|
    "mutationProbability" -> 0.3,
    "edgeRewiringWeight" -> 0.4,
    "weightPerturbationWeight" -> 0.3,
    "phaseRotationWeight" -> 0.3
  |>,
  "Controller" -> <|
    "strategy" -> "Fixed",
    "betaH" -> 1.0,
    "mu" -> 0.1,
    "alpha" -> 0.01,
    "learningRate" -> 0.001
  |>,
  "OutputDir" -> "io/output",
  "LogLevel" -> "INFO",
  "Seed" -> Automatic,
  "ConvergenceTolerance" -> 10^-6,
  "ConvergenceWindow" -> 50,
  "Maximizing" -> True  (* Maximize Gamma by default *)
};

(* Main engine function *)
HAGOEngine[initGraph_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {maxIter, checkpointInt, tempConfig, optConfig, ctrlConfig,
   outputDir, logLevel, seed, convTol, convWin, maximizing,
   currentGraph, bestGraph, currentGamma, bestGamma,
   temperature, params, history, iteration, startTime,
   mutated, newGamma, deltaGamma, accepted, converged,
   result},
  
  (* Extract options *)
  maxIter = OptionValue["MaxIterations"];
  checkpointInt = OptionValue["CheckpointInterval"];
  tempConfig = OptionValue["Temperature"];
  optConfig = OptionValue["Optimizer"];
  ctrlConfig = OptionValue["Controller"];
  outputDir = OptionValue["OutputDir"];
  logLevel = OptionValue["LogLevel"];
  seed = OptionValue["Seed"];
  convTol = OptionValue["ConvergenceTolerance"];
  convWin = OptionValue["ConvergenceWindow"];
  maximizing = OptionValue["Maximizing"];
  
  (* Set random seed if specified *)
  If[seed =!= Automatic,
    SeedRandom[seed]
  ];
  
  (* Initialize controller *)
  InitializeController[<|"controller" -> ctrlConfig|>];
  
  (* Initialize acceptance state *)
  ResetAcceptanceState[];
  
  (* Initialize parameters *)
  params = <|
    "betaH" -> Lookup[ctrlConfig, "betaH", 1.0],
    "mu" -> Lookup[ctrlConfig, "mu", 0.1],
    "alpha" -> Lookup[ctrlConfig, "alpha", 0.01]
  |>;
  
  (* Initialize state *)
  currentGraph = initGraph;
  bestGraph = initGraph;
  currentGamma = Gamma[currentGraph, params];
  bestGamma = currentGamma;
  
  If[currentGamma === $Failed,
    Print["Error: Failed to compute initial Gamma"];
    Return[$Failed]
  ];
  
  (* Initialize temperature *)
  temperature = Lookup[tempConfig, "initial", 1.0];
  
  (* Initialize history *)
  history = {
    <|
      "Iteration" -> 0,
      "Gamma" -> currentGamma,
      "Temperature" -> temperature,
      "Accepted" -> True,
      "Parameters" -> params
    |>
  };
  
  startTime = AbsoluteTime[];
  converged = False;
  
  (* Log start *)
  IRHLog["INFO", StringForm["ARO Engine started. MaxIterations=`1`", maxIter]];
  IRHLog["INFO", StringForm["Initial Gamma=`1`", currentGamma]];
  
  (* Main optimization loop *)
  Do[
    (* Generate mutation *)
    mutated = MutateGraph[currentGraph,
      "MutationKernel" -> "Mixed",
      "KernelWeights" -> {
        Lookup[optConfig, "edgeRewiringWeight", 0.4],
        Lookup[optConfig, "weightPerturbationWeight", 0.3],
        Lookup[optConfig, "phaseRotationWeight", 0.3]
      },
      "MutationStrength" -> 1.0 - (iteration - 1) / maxIter  (* Decreasing strength *)
    ];
    
    If[mutated === $Failed,
      IRHLog["WARNING", StringForm["Mutation failed at iteration `1`", iteration]];
      Continue[]
    ];
    
    (* Compute new Gamma *)
    newGamma = Gamma[mutated, params];
    
    If[newGamma === $Failed,
      IRHLog["WARNING", StringForm["Gamma computation failed at iteration `1`", iteration]];
      Continue[]
    ];
    
    (* Compute change *)
    deltaGamma = newGamma - currentGamma;
    
    (* Accept/reject *)
    accepted = AcceptChange[deltaGamma, temperature, "Maximizing" -> maximizing];
    
    If[accepted,
      currentGraph = mutated;
      currentGamma = newGamma;
      
      (* Update best if improved *)
      If[(maximizing && currentGamma > bestGamma) || (!maximizing && currentGamma < bestGamma),
        bestGraph = currentGraph;
        bestGamma = currentGamma;
        IRHLog["DEBUG", StringForm["New best Gamma=`1` at iteration `2`", bestGamma, iteration]]
      ]
    ];
    
    (* Update parameters *)
    params = UpdateParameters[params, history, 
      "Strategy" -> Lookup[ctrlConfig, "strategy", "Fixed"]
    ];
    
    (* Update temperature (annealing) *)
    temperature = annealTemperature[tempConfig, iteration, maxIter];
    
    (* Record history *)
    AppendTo[history, <|
      "Iteration" -> iteration,
      "Gamma" -> currentGamma,
      "BestGamma" -> bestGamma,
      "Temperature" -> temperature,
      "Accepted" -> accepted,
      "DeltaGamma" -> deltaGamma,
      "Parameters" -> params
    |>];
    
    (* Log harmony *)
    LogHarmony[iteration, currentGamma, <|
      "Temperature" -> temperature,
      "Accepted" -> accepted,
      "BestGamma" -> bestGamma
    |>];
    
    (* Check convergence *)
    If[checkConvergence[history, convTol, convWin],
      IRHLog["INFO", StringForm["Converged at iteration `1`", iteration]];
      converged = True;
      Break[]
    ];
    
    (* Checkpoint *)
    If[Mod[iteration, checkpointInt] == 0,
      checkpoint[bestGraph, history, outputDir, iteration];
      IRHLog["INFO", StringForm["Checkpoint at iteration `1`, Gamma=`2`", iteration, bestGamma]]
    ];
    
    (* Progress logging *)
    If[Mod[iteration, Max[1, Floor[maxIter/10]]] == 0,
      Print[StringForm["  Iteration `1`/`2`: Gamma=`3`, Best=`4`, T=`5`",
        iteration, maxIter, 
        NumberForm[currentGamma, 4],
        NumberForm[bestGamma, 4],
        NumberForm[temperature, 3]
      ]]
    ],
    
    {iteration, 1, maxIter}
  ];
  
  (* Final checkpoint *)
  checkpoint[bestGraph, history, outputDir, "final"];
  
  (* Log completion *)
  IRHLog["INFO", StringForm["ARO Engine completed. Final Gamma=`1`, Best=`2`", 
    currentGamma, bestGamma]];
  IRHLog["INFO", StringForm["Total time: `1` seconds", AbsoluteTime[] - startTime]];
  
  (* Return result *)
  result = <|
    "OptimizedGraph" -> bestGraph,
    "FinalGamma" -> bestGamma,
    "CurrentGamma" -> currentGamma,
    "History" -> history,
    "TotalIterations" -> Length[history] - 1,
    "Converged" -> converged,
    "ExecutionTime" -> AbsoluteTime[] - startTime,
    "FinalParameters" -> params,
    "AcceptanceStats" -> GetAcceptanceState[]
  |>;
  
  result
];

HAGOEngine[_, ___] := (
  Message[HAGOEngine::invalidgs];
  $Failed
);

HAGOEngine::invalidgs = "HAGOEngine requires a valid GraphState as input.";

(* Temperature annealing schedules *)
annealTemperature[config_, iteration_, maxIter_] := Module[
  {initial, final, schedule, progress, newTemp},
  
  initial = Lookup[config, "initial", 1.0];
  final = Lookup[config, "final", 0.01];
  schedule = Lookup[config, "schedule", "exponential"];
  
  progress = iteration / maxIter;
  
  newTemp = Switch[schedule,
    "exponential",
      initial * (final / initial)^progress,
    "linear",
      initial + (final - initial) * progress,
    "quadratic",
      initial + (final - initial) * progress^2,
    "logarithmic",
      initial / (1 + Log[1 + iteration]),
    "cosine",
      final + (initial - final) * (1 + Cos[Pi * progress]) / 2,
    _,
      initial * (final / initial)^progress
  ];
  
  Max[final, newTemp]
];

(* Check convergence *)
checkConvergence[history_, tolerance_, window_] := Module[
  {recent, gammas, mean, variance},
  
  If[Length[history] < window,
    Return[False]
  ];
  
  recent = Take[history, -window];
  gammas = #["Gamma"] & /@ recent;
  
  mean = Mean[gammas];
  variance = Variance[gammas];
  
  (* Converged if variance is very small *)
  If[NumericQ[variance],
    Sqrt[variance] / Max[1, Abs[mean]] < tolerance,
    False
  ]
];

(* Checkpoint function *)
checkpoint[graph_, history_, outputDir_, iteration_] := Module[
  {filename, historyFile},
  
  (* Ensure output directory exists *)
  If[!DirectoryQ[outputDir],
    Quiet[CreateDirectory[outputDir]]
  ];
  
  (* Save graph state *)
  filename = FileNameJoin[{outputDir, 
    StringForm["checkpoint_`1`.irh", iteration] // ToString
  }];
  
  Export[filename, 
    <|"Graph" -> graph, "Iteration" -> iteration|>,
    "JSON"
  ];
  
  (* Save history *)
  historyFile = FileNameJoin[{outputDir, "optimization_history.json"}];
  Export[historyFile, history, "JSON"]
];

(* Single step for debugging *)
HAGOStep[state_Association] := Module[
  {graph, params, temperature, mutated, newGamma, deltaGamma, accepted},
  
  graph = state["Graph"];
  params = state["Parameters"];
  temperature = state["Temperature"];
  
  mutated = MutateGraph[graph, "MutationKernel" -> "Mixed"];
  newGamma = Gamma[mutated, params];
  deltaGamma = newGamma - state["Gamma"];
  accepted = AcceptChange[deltaGamma, temperature, "Maximizing" -> True];
  
  <|
    "Graph" -> If[accepted, mutated, graph],
    "Gamma" -> If[accepted, newGamma, state["Gamma"]],
    "Parameters" -> params,
    "Temperature" -> temperature,
    "Accepted" -> accepted,
    "DeltaGamma" -> deltaGamma
  |>
];

End[];

EndPackage[];
