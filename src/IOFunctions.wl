(* ::Package:: *)

(* ============================================================================
   IOFunctions.wl - Save and Load Graph States
   ============================================================================
   
   Purpose:
     Provides serialization and deserialization of GraphState objects to/from
     .irh files (JSON format with optional binary eigen-cache).
   
   Inputs:
     - GraphState: A valid GraphState to save
     - filepath: Path to .irh file
   
   Outputs:
     - Saved files: .irh (JSON), .irh.cache (binary eigenspectrum)
     - Loaded GraphState association
   
   File Format:
     .irh files are JSON with the following structure:
     {
       "format": "IRH_Suite_v3.0",
       "type": "GraphState",
       "nodeCount": N,
       "edgeCount": E,
       "adjacencyMatrix": [[...]],
       "weights": [[...]],
       "phases": [[...]],
       "metadata": {...}
     }
   
   References:
     - JSON specification (RFC 8259)
     - Mathematica serialization formats
   
   ============================================================================ *)

BeginPackage["IRHSuite`IOFunctions`"];

SaveGraphState::usage = "SaveGraphState[graphState, filepath] saves a GraphState to \
an .irh file (JSON format). Returns True on success.
Example: SaveGraphState[gs, \"io/output/G_opt.irh\"]";

LoadGraphState::usage = "LoadGraphState[filepath] loads a GraphState from an .irh file. \
Returns a GraphState Association or $Failed.
Example: gs = LoadGraphState[\"io/output/G_opt.irh\"]";

SaveEigenCache::usage = "SaveEigenCache[eigenspectrum, filepath] saves eigenspectrum \
as binary cache for faster loading.";

LoadEigenCache::usage = "LoadEigenCache[filepath] loads cached eigenspectrum.";

ExportToGraph::usage = "ExportToGraph[graphState, filepath, format] exports GraphState \
to standard graph formats (GraphML, DOT, etc.).";

Begin["`Private`"];

Needs["IRHSuite`GraphState`"];

(* Save GraphState to .irh file *)
SaveGraphState[gs_?GraphStateQ, filepath_String] := Module[
  {jsonData, dir, result},
  
  (* Ensure directory exists *)
  dir = DirectoryName[filepath];
  If[dir != "" && !DirectoryQ[dir],
    Quiet[CreateDirectory[dir]]
  ];
  
  (* Build JSON structure *)
  jsonData = <|
    "format" -> "IRH_Suite_v3.0",
    "type" -> "GraphState",
    "nodeCount" -> gs["NodeCount"],
    "edgeCount" -> gs["EdgeCount"],
    "adjacencyMatrix" -> Normal[gs["AdjacencyMatrix"]],
    "weights" -> Normal[gs["Weights"]],
    "phases" -> Normal[gs["Phases"]],
    "edgeList" -> gs["EdgeList"],
    "metadata" -> gs["Metadata"]
  |>;
  
  (* Export *)
  result = Export[filepath, jsonData, "RawJSON"];
  
  If[result === filepath,
    True,
    Message[SaveGraphState::exportfail, filepath];
    False
  ]
];

SaveGraphState::exportfail = "Failed to export GraphState to `1`.";
SaveGraphState[_, _] := (Message[SaveGraphState::invalidgs]; False);
SaveGraphState::invalidgs = "First argument must be a valid GraphState.";

(* Load GraphState from .irh file *)
LoadGraphState[filepath_String] := Module[
  {jsonData, adjMat, weights, phases, n, edgeList, metadata},
  
  If[!FileExistsQ[filepath],
    Message[LoadGraphState::notfound, filepath];
    Return[$Failed]
  ];
  
  (* Import JSON *)
  jsonData = Quiet[Import[filepath, "RawJSON"]];
  
  If[!AssociationQ[jsonData],
    Message[LoadGraphState::parsefail, filepath];
    Return[$Failed]
  ];
  
  (* Validate format *)
  If[Lookup[jsonData, "format", ""] != "IRH_Suite_v3.0",
    Message[LoadGraphState::wrongformat, filepath]
    (* Continue anyway - may be compatible *)
  ];
  
  (* Extract data *)
  n = jsonData["nodeCount"];
  adjMat = N[jsonData["adjacencyMatrix"]];
  weights = N[jsonData["weights"]];
  phases = N[jsonData["phases"]];
  edgeList = jsonData["edgeList"];
  metadata = Lookup[jsonData, "metadata", <||>];
  
  (* Validate dimensions *)
  If[Dimensions[adjMat] != {n, n},
    Message[LoadGraphState::baddim, filepath];
    Return[$Failed]
  ];
  
  (* Construct GraphState *)
  <|
    "Type" -> "GraphState",
    "AdjacencyMatrix" -> adjMat,
    "Weights" -> weights,
    "Phases" -> phases,
    "NodeCount" -> n,
    "EdgeCount" -> Length[edgeList],
    "EdgeList" -> edgeList,
    "Metadata" -> Append[metadata, "LoadedFrom" -> filepath]
  |>
];

LoadGraphState::notfound = "File not found: `1`";
LoadGraphState::parsefail = "Failed to parse JSON from: `1`";
LoadGraphState::wrongformat = "File `1` may not be IRH_Suite v3.0 format.";
LoadGraphState::baddim = "Dimension mismatch in file: `1`";
LoadGraphState[_] := $Failed;

(* Save eigenspectrum cache *)
SaveEigenCache[spectrum_Association, filepath_String] := Module[
  {cacheFile, data},
  
  cacheFile = filepath <> ".cache";
  
  data = <|
    "eigenvalues" -> spectrum["Eigenvalues"],
    "degeneracies" -> spectrum["Degeneracies"],
    "method" -> spectrum["Method"],
    "tolerance" -> spectrum["Tolerance"],
    "timestamp" -> DateString["ISODateTime"]
  |>;
  
  Export[cacheFile, data, "WXF"]
];

(* Load eigenspectrum cache *)
LoadEigenCache[filepath_String] := Module[
  {cacheFile, data},
  
  cacheFile = filepath <> ".cache";
  
  If[!FileExistsQ[cacheFile],
    Return[$Failed]
  ];
  
  data = Quiet[Import[cacheFile, "WXF"]];
  
  If[!AssociationQ[data],
    Return[$Failed]
  ];
  
  <|
    "Eigenvalues" -> data["eigenvalues"],
    "Eigenvectors" -> None,
    "Degeneracies" -> data["degeneracies"],
    "NumericalWarnings" -> {},
    "Method" -> data["method"],
    "Tolerance" -> data["tolerance"],
    "CachedAt" -> data["timestamp"]
  |>
];

(* Export to standard graph formats *)
Options[ExportToGraph] = {
  "Format" -> "GraphML"
};

ExportToGraph[gs_?GraphStateQ, filepath_String, opts:OptionsPattern[]] := Module[
  {format, g, result},
  
  format = OptionValue["Format"];
  
  (* Build Mathematica Graph object *)
  g = AdjacencyGraph[gs["AdjacencyMatrix"], 
    EdgeWeight -> Flatten[gs["Weights"]],
    VertexLabels -> "Name"
  ];
  
  (* Export *)
  result = Switch[format,
    "GraphML",
      Export[filepath, g, "GraphML"],
    "DOT",
      Export[filepath, g, "DOT"],
    "GXL",
      Export[filepath, g, "GXL"],
    "Pajek",
      Export[filepath, g, "Pajek"],
    _,
      Message[ExportToGraph::unknownformat, format];
      $Failed
  ];
  
  result
];

ExportToGraph::unknownformat = "Unknown format: `1`. Use GraphML, DOT, GXL, or Pajek.";
ExportToGraph[_, _, ___] := $Failed;

End[];

EndPackage[];
