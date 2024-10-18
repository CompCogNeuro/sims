// Code generated by "core generate -add-types"; DO NOT EDIT.

package main

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "main.Config", IDName: "config", Doc: "Config has config parameters related to running the sim", Fields: []types.Field{{Name: "NRuns", Doc: "total number of runs to do when running Train"}, {Name: "NEpochs", Doc: "total number of epochs per run"}, {Name: "NTrials", Doc: "total number of trials per epochs per run"}, {Name: "NZero", Doc: "stop run after this number of perfect, zero-error epochs."}, {Name: "TestInterval", Doc: "how often to run through all the test patterns, in terms of training epochs.\ncan use 0 or -1 for no testing."}}})

var _ = types.AddType(&types.Type{Name: "main.Sim", IDName: "sim", Doc: "Sim encapsulates the entire simulation model, and we define all the\nfunctionality as methods on this struct.  This structure keeps all relevant\nstate information organized and available without having to pass everything around\nas arguments to methods, and provides the core GUI interface (note the view tags\nfor the fields which provide hints to how things should be displayed).", Fields: []types.Field{{Name: "BurstDaGain", Doc: "BurstDaGain is the strength of dopamine bursts: 1 default -- reduce for PD OFF, increase for PD ON"}, {Name: "DipDaGain", Doc: "DipDaGain is the strength of dopamine dips: 1 default -- reduce to siulate D2 agonists"}, {Name: "Config", Doc: "Config contains misc configuration parameters for running the sim"}, {Name: "Net", Doc: "the network -- click to view / edit parameters for layers, paths, etc"}, {Name: "Params", Doc: "network parameter management"}, {Name: "Loops", Doc: "contains looper control loops for running sim"}, {Name: "Stats", Doc: "contains computed statistic values"}, {Name: "Logs", Doc: "Contains all the logs and information about the logs.'"}, {Name: "Envs", Doc: "Environments"}, {Name: "Context", Doc: "leabra timing parameters and state"}, {Name: "ViewUpdate", Doc: "netview update parameters"}, {Name: "GUI", Doc: "manages all the gui elements"}, {Name: "RandSeeds", Doc: "a list of random seeds to use for each run"}}})

var _ = types.AddType(&types.Type{Name: "main.Actions", IDName: "actions", Doc: "Actions are SIR actions"})

var _ = types.AddType(&types.Type{Name: "main.SIREnv", IDName: "sir-env", Doc: "SIREnv implements the store-ignore-recall task", Fields: []types.Field{{Name: "Name", Doc: "name of this environment"}, {Name: "NStim", Doc: "number of different stimuli that can be maintained"}, {Name: "RewVal", Doc: "value for reward, based on whether model output = target"}, {Name: "NoRewVal", Doc: "value for non-reward"}, {Name: "Act", Doc: "current action"}, {Name: "Stim", Doc: "current stimulus"}, {Name: "Maint", Doc: "current stimulus being maintained"}, {Name: "Input", Doc: "input pattern with stim"}, {Name: "CtrlInput", Doc: "input pattern with action"}, {Name: "Output", Doc: "output pattern of what to respond"}, {Name: "Reward", Doc: "reward value"}, {Name: "Trial", Doc: "trial is the step counter within epoch"}}})