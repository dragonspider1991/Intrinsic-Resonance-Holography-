(* ::Package:: *)

(* ============================================================================
   SpectralDimension.wl - Spectral Dimension Computation
   ============================================================================
   
   Purpose:
     Computes the spectral dimension of a graph state, which characterizes
     the effective dimensionality of the discrete spacetime as probed by
     diffusion processes. Target: d_s ≈ 4 for physical spacetime.
   
   Inputs:
     - GraphState: A valid GraphState
     - opts: Fitting parameters
   
   Outputs:
     - Association containing:
       * "Value": Estimated spectral dimension
       * "Error": Uncertainty estimate
       * "FitQuality": R² or residuals
       * "FitData": Raw data used for fitting
   
   Equations Implemented:
     The spectral dimension is defined via the heat kernel:
       P(t) = Tr[exp(-t·L)] ≈ t^(-d_s/2) for small t
     
     We compute d_s from the log-log slope:
       d_s = -2 · d(log P)/d(log t)
   
   References:
     - Spectral Geometry
     - Causal Dynamical Triangulations
     - Random walk on graphs
   
   ============================================================================ *)

BeginPackage["IRHSuite`SpectralDimension`"];

SpectralDimension::usage = "SpectralDimension[graphState, opts] computes the spectral \
dimension via heat kernel analysis. Options: \"FitRange\" -> {tMin, tMax}, \
\"NumPoints\" -> number of sample points.
Example: ds = SpectralDimension[gs]";

HeatKernelTrace::usage = "HeatKernelTrace[graphState, t] computes Tr[exp(-t·L)] \
for the interference matrix L at diffusion time t.
Example: p = HeatKernelTrace[gs, 0.1]";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];
Needs["IRHSuite`EigenSpectrum`"];

(* Options *)
Options[SpectralDimension] = {
  "FitRange" -> {0.01, 10.0},
  "NumPoints" -> 50,
  "FitMethod" -> "LinearRegression",
  "UseEigenvalues" -> True
};

(* Main spectral dimension function *)
SpectralDimension[gs_?GraphStateQ, opts:OptionsPattern[]] := Module[
  {fitRange, numPoints, fitMethod, useEigen,
   tValues, pValues, logT, logP, fitData, fit, slope, intercept,
   ds, dsError, residuals, rSquared},
  
  fitRange = OptionValue["FitRange"];
  numPoints = OptionValue["NumPoints"];
  fitMethod = OptionValue["FitMethod"];
  useEigen = OptionValue["UseEigenvalues"];
  
  (* Generate time points (log-spaced) *)
  tValues = Exp[Subdivide[Log[fitRange[[1]]], Log[fitRange[[2]]], numPoints - 1]];
  
  (* Compute heat kernel trace at each time *)
  If[useEigen,
    pValues = heatKernelFromEigen[gs, tValues],
    pValues = Table[HeatKernelTrace[gs, t], {t, tValues}]
  ];
  
  (* Filter out invalid values *)
  fitData = Select[
    Transpose[{tValues, pValues}],
    NumericQ[#[[2]]] && #[[2]] > 0 &
  ];
  
  If[Length[fitData] < 5,
    Return[<|
      "Value" -> Indeterminate,
      "Error" -> Infinity,
      "FitQuality" -> 0,
      "Message" -> "Insufficient valid data points"
    |>]
  ];
  
  (* Log transform *)
  logT = Log[#[[1]]] & /@ fitData;
  logP = Log[#[[2]]] & /@ fitData;
  
  (* Linear regression: log P = -ds/2 * log t + const *)
  {slope, intercept, dsError, rSquared, residuals} = 
    linearFit[logT, logP];
  
  (* Extract spectral dimension: slope = -ds/2 *)
  ds = -2 * slope;
  dsError = 2 * dsError;  (* Propagate uncertainty *)
  
  <|
    "Value" -> ds,
    "Error" -> dsError,
    "FitQuality" -> rSquared,
    "Slope" -> slope,
    "Intercept" -> intercept,
    "Residuals" -> residuals,
    "FitData" -> <|
      "LogT" -> logT,
      "LogP" -> logP
    |>,
    "FitRange" -> fitRange,
    "NumPoints" -> Length[fitData]
  |>
];

SpectralDimension[_, ___] := (
  Message[SpectralDimension::invalidgs];
  $Failed
);

SpectralDimension::invalidgs = "SpectralDimension requires a valid GraphState.";

(* Heat kernel trace from eigenvalues *)
heatKernelFromEigen[gs_, tValues_] := Module[
  {spectrum, eigenvalues},
  
  spectrum = EigenSpectrum[gs, "ReturnVectors" -> False];
  
  If[spectrum === $Failed,
    Return[ConstantArray[$Failed, Length[tValues]]]
  ];
  
  eigenvalues = spectrum["Eigenvalues"];
  
  (* P(t) = Σ exp(-λ_i * t) *)
  Table[
    Total[Exp[-Re[eigenvalues] * t]],
    {t, tValues}
  ]
];

(* Direct heat kernel trace computation *)
HeatKernelTrace[gs_?GraphStateQ, t_?NumericQ] := Module[
  {L, expL, trace},
  
  L = BuildInterferenceMatrix[gs];
  
  If[L === $Failed,
    Return[$Failed]
  ];
  
  (* Compute matrix exponential trace *)
  (* For efficiency, use eigenvalue method for trace *)
  expL = MatrixExp[-t * Re[L]];
  trace = Tr[expL];
  
  trace
];

HeatKernelTrace[_, _] := $Failed;

(* Linear regression with error estimates *)
linearFit[xData_, yData_] := Module[
  {n, xMean, yMean, sxx, sxy, syy, slope, intercept, 
   residuals, sse, mse, seSlope, rSquared},
  
  n = Length[xData];
  
  If[n < 3,
    Return[{0, 0, Infinity, 0, {}}]
  ];
  
  xMean = Mean[xData];
  yMean = Mean[yData];
  
  sxx = Total[(xData - xMean)^2];
  sxy = Total[(xData - xMean) * (yData - yMean)];
  syy = Total[(yData - yMean)^2];
  
  If[sxx == 0,
    Return[{0, yMean, Infinity, 0, yData - yMean}]
  ];
  
  slope = sxy / sxx;
  intercept = yMean - slope * xMean;
  
  (* Residuals *)
  residuals = yData - (slope * xData + intercept);
  sse = Total[residuals^2];
  
  (* Mean squared error and standard error of slope *)
  mse = If[n > 2, sse / (n - 2), sse];
  seSlope = If[sxx > 0, Sqrt[mse / sxx], Infinity];
  
  (* R-squared *)
  rSquared = If[syy > 0, 1 - sse / syy, 0];
  
  {slope, intercept, seSlope, rSquared, residuals}
];

End[];

EndPackage[];
