(* ::Package:: *)

(* ============================================================================
   Visualization.wl - Graph and Spectral Visualization
   ============================================================================
   
   Purpose:
     Provides visualization functions for GraphStates including 3D graph
     layout, spectral density plots, and Gamma evolution tracking.
   
   Inputs:
     - GraphState or analysis results
     - opts: Visualization options
   
   Outputs:
     - Graphics objects suitable for display or export
   
   Visualizations:
     - Plot3DGraph: 3D spring-embedded layout
     - PlotSpectralDensity: Eigenvalue distribution
     - PlotGammaEvolution: Optimization trajectory
     - PlotSignature: Eigenvalue signature visualization
   
   References:
     - Graph drawing algorithms
     - Spectral visualization techniques
   
   ============================================================================ *)

BeginPackage["IRHSuite`Visualization`"];

Plot3DGraph::usage = "Plot3DGraph[graphState, opts] creates a 3D visualization of \
the graph structure with weighted edges.
Example: Plot3DGraph[gs, \"ColorScheme\" -> \"Rainbow\"]";

PlotSpectralDensity::usage = "PlotSpectralDensity[graphState, opts] plots the \
eigenvalue density distribution.
Example: PlotSpectralDensity[gs]";

PlotGammaEvolution::usage = "PlotGammaEvolution[history, opts] plots the evolution \
of the Harmony Functional during optimization.
Example: PlotGammaEvolution[result[\"History\"]]";

PlotSignature::usage = "PlotSignature[graphState, opts] visualizes the eigenvalue \
signature pattern.
Example: PlotSignature[gs]";

ExportVisualization::usage = "ExportVisualization[graphic, filepath] exports a \
visualization to PNG or PDF.";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];
Needs["IRHSuite`EigenSpectrum`"];

(* Options *)
Options[Plot3DGraph] = {
  "ColorScheme" -> "Rainbow",
  "EdgeThickness" -> Automatic,
  "NodeSize" -> Automatic,
  "ShowWeights" -> False,
  "Layout" -> "SpringElectricalEmbedding"
};

Options[PlotSpectralDensity] = {
  "Bins" -> Automatic,
  "ShowNegative" -> True,
  "Smoothing" -> False
};

Options[PlotGammaEvolution] = {
  "ShowTemperature" -> True,
  "ShowAcceptance" -> False,
  "Smooth" -> False
};

Options[PlotSignature] = {
  "ColorPositive" -> Blue,
  "ColorNegative" -> Red,
  "ColorZero" -> Gray
};

(* 3D Graph visualization *)
Plot3DGraph[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {colorScheme, layout, n, adjMat, weights, g, coords},
  
  colorScheme = OptionValue["ColorScheme"];
  layout = OptionValue["Layout"];
  
  n = gs["NodeCount"];
  adjMat = gs["AdjacencyMatrix"];
  weights = gs["Weights"];
  
  (* Create graph object *)
  g = AdjacencyGraph[adjMat];
  
  (* Use 3D layout *)
  coords = GraphEmbedding[g, {layout, 3}];
  
  If[coords === $Failed || Length[coords] != n,
    (* Fall back to random coordinates *)
    coords = RandomReal[{-1, 1}, {n, 3}]
  ];
  
  (* Build graphics *)
  Graphics3D[{
    (* Edges *)
    Table[
      If[adjMat[[i, j]] > 0 && i < j,
        {
          Opacity[Clip[weights[[i, j]], {0.2, 1}]],
          Thickness[0.002],
          Line[{coords[[i]], coords[[j]]}]
        },
        Nothing
      ],
      {i, n}, {j, n}
    ] // Flatten,
    
    (* Nodes *)
    Table[
      {
        ColorData[colorScheme][i/n],
        Sphere[coords[[i]], 0.05]
      },
      {i, n}
    ]
  },
    Boxed -> False,
    Lighting -> "Neutral",
    PlotLabel -> StringForm["GraphState (N=`1`, E=`2`)", n, gs["EdgeCount"]],
    ImageSize -> Large
  ]
];

Plot3DGraph[_, ___] := (
  Message[Plot3DGraph::invalidgs];
  Graphics3D[{}]
);

Plot3DGraph::invalidgs = "Plot3DGraph requires a valid GraphState.";

(* Spectral density plot *)
PlotSpectralDensity[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {bins, showNeg, smoothing, spectrum, eigenvalues, realParts,
   negVals, posVals, zeroCount},
  
  bins = OptionValue["Bins"];
  showNeg = OptionValue["ShowNegative"];
  smoothing = OptionValue["Smoothing"];
  
  (* Get eigenvalues *)
  spectrum = EigenSpectrum[gs, "ReturnVectors" -> False];
  If[spectrum === $Failed,
    Return[Graphics[{}, PlotLabel -> "Failed to compute spectrum"]]
  ];
  
  eigenvalues = spectrum["Eigenvalues"];
  realParts = Re[eigenvalues];
  
  (* Separate positive and negative *)
  negVals = Select[realParts, # < -10^-10 &];
  posVals = Select[realParts, # > 10^-10 &];
  zeroCount = Count[realParts, x_ /; Abs[x] <= 10^-10];
  
  (* Create histogram *)
  If[bins === Automatic,
    bins = Max[10, Floor[Sqrt[Length[eigenvalues]]]]
  ];
  
  Show[
    Histogram[realParts, bins, "Probability",
      PlotLabel -> "Eigenvalue Density",
      FrameLabel -> {"Eigenvalue λ", "Probability"},
      ChartStyle -> If[showNeg, 
        {Directive[Blue, Opacity[0.7]]},
        {Blue}
      ],
      PlotRange -> All,
      Frame -> True
    ],
    Graphics[{
      Text[
        StringForm["N=`1`, Neg=`2`, Zero=`3`, Pos=`4`", 
          Length[eigenvalues], Length[negVals], zeroCount, Length[posVals]],
        Scaled[{0.95, 0.95}],
        {1, 1}
      ]
    }],
    ImageSize -> Large
  ]
];

PlotSpectralDensity[_, ___] := Graphics[{}, PlotLabel -> "Invalid input"];

(* Gamma evolution plot *)
PlotGammaEvolution[history_List, opts:OptionsPattern[]] := Module[
  {showTemp, showAccept, smooth, iterations, gammas, temps, 
   gammaPlot, tempPlot, plots},
  
  showTemp = OptionValue["ShowTemperature"];
  showAccept = OptionValue["ShowAcceptance"];
  smooth = OptionValue["Smooth"];
  
  (* Extract data from history *)
  iterations = #["Iteration"] & /@ history;
  gammas = #["Gamma"] & /@ history;
  
  If[!AllTrue[gammas, NumericQ],
    Return[Graphics[{}, PlotLabel -> "Invalid history data"]]
  ];
  
  (* Main Gamma plot *)
  gammaPlot = ListLinePlot[
    Transpose[{iterations, gammas}],
    PlotStyle -> Blue,
    PlotLabel -> "Harmony Functional Evolution",
    FrameLabel -> {"Iteration", "Γ"},
    Frame -> True,
    PlotRange -> All,
    ImageSize -> Large
  ];
  
  (* Temperature overlay if requested *)
  If[showTemp && AllTrue[history, KeyExistsQ[#, "Temperature"] &],
    temps = #["Temperature"] & /@ history;
    tempPlot = ListLinePlot[
      Transpose[{iterations, temps / Max[temps] * Max[gammas]}],
      PlotStyle -> {Red, Dashed},
      PlotLegends -> {"Temperature (scaled)"}
    ];
    gammaPlot = Show[gammaPlot, tempPlot, PlotRange -> All]
  ];
  
  gammaPlot
];

PlotGammaEvolution[_, ___] := Graphics[{}, PlotLabel -> "Invalid history"];

(* Signature visualization *)
PlotSignature[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {colPos, colNeg, colZero, spectrum, eigenvalues, realParts,
   negIdx, posIdx, zeroIdx, points},
  
  colPos = OptionValue["ColorPositive"];
  colNeg = OptionValue["ColorNegative"];
  colZero = OptionValue["ColorZero"];
  
  spectrum = EigenSpectrum[gs, "ReturnVectors" -> False];
  If[spectrum === $Failed,
    Return[Graphics[{}, PlotLabel -> "Failed to compute spectrum"]]
  ];
  
  eigenvalues = Sort[Re[spectrum["Eigenvalues"]]];
  
  (* Classify *)
  negIdx = Flatten[Position[eigenvalues, x_ /; x < -10^-10]];
  posIdx = Flatten[Position[eigenvalues, x_ /; x > 10^-10]];
  zeroIdx = Complement[Range[Length[eigenvalues]], negIdx, posIdx];
  
  (* Create visualization *)
  points = Join[
    Table[{colNeg, PointSize[Large], Point[{i, eigenvalues[[i]]}]}, {i, negIdx}],
    Table[{colZero, PointSize[Medium], Point[{i, eigenvalues[[i]]}]}, {i, zeroIdx}],
    Table[{colPos, PointSize[Medium], Point[{i, eigenvalues[[i]]}]}, {i, posIdx}]
  ];
  
  Graphics[
    Join[
      {Line[{{0, 0}, {Length[eigenvalues] + 1, 0}}]},  (* Zero line *)
      points
    ],
    Axes -> True,
    AxesLabel -> {"Index", "Eigenvalue λ"},
    PlotLabel -> StringForm["Eigenvalue Signature: (`1`, `2`)", 
      Length[posIdx], Length[negIdx]],
    PlotRange -> All,
    ImageSize -> Large
  ]
];

PlotSignature[_, ___] := Graphics[{}, PlotLabel -> "Invalid input"];

(* Export visualization *)
ExportVisualization[graphic_, filepath_String] := Module[
  {ext, result},
  
  ext = ToLowerCase[FileExtension[filepath]];
  
  result = Switch[ext,
    "png", Export[filepath, graphic, "PNG", ImageResolution -> 300],
    "pdf", Export[filepath, graphic, "PDF"],
    "svg", Export[filepath, graphic, "SVG"],
    "eps", Export[filepath, graphic, "EPS"],
    _, Export[filepath <> ".png", graphic, "PNG", ImageResolution -> 300]
  ];
  
  result
];

End[];

EndPackage[];
