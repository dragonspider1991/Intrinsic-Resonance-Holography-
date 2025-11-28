(* ::Package:: *)

(* ============================================================================
   Logging.wl - Exhaustive Logging System
   ============================================================================
   
   Purpose:
     Provides comprehensive logging facilities including timestamped CSV
     logging of Harmony Functional evolution and human-readable console output.
   
   Inputs:
     - Log level: DEBUG, INFO, WARNING, ERROR
     - Log messages and structured data
   
   Outputs:
     - Console output (configurable verbosity)
     - CSV file: log_harmony.csv
     - Rotating log files
   
   Format:
     CSV columns: Timestamp, Iteration, Gamma, Temperature, Accepted, 
                  DeltaGamma, BestGamma, NodeCount, EdgeCount, Meta
   
   References:
     - Structured logging best practices
   
   ============================================================================ *)

BeginPackage["IRHSuite`Logging`"];

IRHInitializeLog::usage = "IRHInitializeLog[outputDir, logLevel] initializes the \
logging system. Call once at startup.
Example: IRHInitializeLog[\"io/output\", \"INFO\"]";

IRHLog::usage = "IRHLog[level, message] logs a message at the specified level.
Example: IRHLog[\"INFO\", \"Starting optimization\"]";

LogHarmony::usage = "LogHarmony[iteration, gamma, meta] logs a Harmony Functional \
value to the CSV file.
Example: LogHarmony[100, 42.5, <|\"Temperature\" -> 0.5|>]";

IRHCloseLog::usage = "IRHCloseLog[] closes all log files and flushes buffers.
Example: IRHCloseLog[]";

GetLogStats::usage = "GetLogStats[] returns statistics about the logging session.";

Begin["`Private`"];

(* Global logging state *)
$logState = <|
  "Initialized" -> False,
  "OutputDir" -> "io/output",
  "LogLevel" -> "INFO",
  "CSVStream" -> None,
  "CSVPath" -> "",
  "MessageCount" -> 0,
  "HarmonyCount" -> 0,
  "StartTime" -> None,
  "Levels" -> <|"DEBUG" -> 0, "INFO" -> 1, "WARNING" -> 2, "ERROR" -> 3|>
|>;

(* Initialize logging *)
IRHInitializeLog[outputDir_String, logLevel_String] := Module[
  {csvPath, headerRow},
  
  (* Ensure output directory exists *)
  If[!DirectoryQ[outputDir],
    Quiet[CreateDirectory[outputDir]]
  ];
  
  (* Set up CSV file *)
  csvPath = FileNameJoin[{outputDir, "log_harmony.csv"}];
  
  (* Write CSV header *)
  headerRow = "Timestamp,Iteration,Gamma,BestGamma,Temperature,Accepted,DeltaGamma,NodeCount,EdgeCount,Meta";
  
  (* Open stream for appending *)
  $logState["CSVStream"] = OpenWrite[csvPath];
  WriteLine[$logState["CSVStream"], headerRow];
  
  (* Update state *)
  $logState["Initialized"] = True;
  $logState["OutputDir"] = outputDir;
  $logState["LogLevel"] = logLevel;
  $logState["CSVPath"] = csvPath;
  $logState["MessageCount"] = 0;
  $logState["HarmonyCount"] = 0;
  $logState["StartTime"] = AbsoluteTime[];
  
  IRHLog["INFO", "Logging initialized"];
];

(* Check if level should be logged *)
shouldLog[level_String] := Module[
  {levelVal, currentVal},
  
  levelVal = Lookup[$logState["Levels"], level, 1];
  currentVal = Lookup[$logState["Levels"], $logState["LogLevel"], 1];
  
  levelVal >= currentVal
];

(* Log message *)
IRHLog[level_String, message_] := Module[
  {timestamp, formattedMsg},
  
  If[!$logState["Initialized"],
    (* Auto-initialize with defaults *)
    IRHInitializeLog["io/output", "INFO"]
  ];
  
  If[!shouldLog[level],
    Return[Null]
  ];
  
  timestamp = DateString[{"Year", "-", "Month", "-", "Day", " ", 
    "Hour", ":", "Minute", ":", "Second"}];
  
  formattedMsg = StringForm["[`1`] [`2`] `3`", timestamp, level, message];
  
  (* Console output *)
  If[level == "ERROR" || level == "WARNING",
    Print[Style[formattedMsg, If[level == "ERROR", Red, Orange]]],
    Print[formattedMsg]
  ];
  
  $logState["MessageCount"]++;
];

(* Log Harmony Functional value *)
LogHarmony[iteration_Integer, gamma_?NumericQ, meta_Association] := Module[
  {timestamp, bestGamma, temperature, accepted, deltaGamma, 
   nodeCount, edgeCount, metaStr, csvLine},
  
  If[!$logState["Initialized"] || $logState["CSVStream"] === None,
    Return[Null]
  ];
  
  (* Extract metadata *)
  timestamp = DateString[{"Year", "-", "Month", "-", "Day", "T",
    "Hour", ":", "Minute", ":", "Second"}];
  bestGamma = Lookup[meta, "BestGamma", gamma];
  temperature = Lookup[meta, "Temperature", 0];
  accepted = If[Lookup[meta, "Accepted", True], 1, 0];
  deltaGamma = Lookup[meta, "DeltaGamma", 0];
  nodeCount = Lookup[meta, "NodeCount", 0];
  edgeCount = Lookup[meta, "EdgeCount", 0];
  
  (* Additional metadata as JSON-like string *)
  metaStr = StringReplace[
    ToString[KeyDrop[meta, {"BestGamma", "Temperature", "Accepted", "DeltaGamma", 
                            "NodeCount", "EdgeCount"}]],
    {"," -> ";", "\n" -> " "}
  ];
  
  (* Format CSV line *)
  csvLine = StringJoin[Riffle[
    ToString /@ {
      timestamp,
      iteration,
      gamma,
      bestGamma,
      temperature,
      accepted,
      deltaGamma,
      nodeCount,
      edgeCount,
      metaStr
    },
    ","
  ]];
  
  (* Write to CSV *)
  WriteLine[$logState["CSVStream"], csvLine];
  
  $logState["HarmonyCount"]++;
  
  (* Periodic flush *)
  If[Mod[$logState["HarmonyCount"], 100] == 0,
    Quiet[Close[$logState["CSVStream"]]];
    $logState["CSVStream"] = OpenAppend[$logState["CSVPath"]]
  ]
];

LogHarmony[_, _, _] := Null;

(* Close logging *)
IRHCloseLog[] := Module[
  {},
  
  If[$logState["CSVStream"] =!= None,
    IRHLog["INFO", StringForm["Logging complete. `1` messages, `2` harmony values recorded.",
      $logState["MessageCount"], $logState["HarmonyCount"]]];
    Quiet[Close[$logState["CSVStream"]]];
    $logState["CSVStream"] = None
  ];
  
  $logState["Initialized"] = False;
];

(* Get statistics *)
GetLogStats[] := <|
  "MessageCount" -> $logState["MessageCount"],
  "HarmonyCount" -> $logState["HarmonyCount"],
  "Duration" -> If[$logState["StartTime"] =!= None,
    AbsoluteTime[] - $logState["StartTime"],
    0
  ],
  "CSVPath" -> $logState["CSVPath"],
  "LogLevel" -> $logState["LogLevel"]
|>;

End[];

EndPackage[];
