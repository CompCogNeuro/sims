// Code generated by "core generate -add-types"; DO NOT EDIT.

package main

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "main.OnOff", IDName: "on-off", Doc: "OnOff represents stimulus On / Off timing", Fields: []types.Field{{Name: "Act", Doc: "is this stimulus active -- use it?"}, {Name: "On", Doc: "when stimulus turns on"}, {Name: "Off", Doc: "when stimulu turns off"}, {Name: "P", Doc: "probability of being active on any given trial"}, {Name: "OnVar", Doc: "variability in onset timing (max number of trials before/after On that it could start)"}, {Name: "OffVar", Doc: "variability in offset timing (max number of trials before/after Off that it could end)"}, {Name: "CurAct", Doc: "current active status based on P probability"}, {Name: "CurOn", Doc: "current on / off values using Var variability"}, {Name: "CurOff", Doc: "current on / off values using Var variability"}}})

var _ = types.AddType(&types.Type{Name: "main.CondEnv", IDName: "cond-env", Doc: "CondEnv simulates an n-armed bandit, where each of n inputs is associated with\na specific probability of reward.", Fields: []types.Field{{Name: "Name", Doc: "name of this environment"}, {Name: "TotTime", Doc: "total time for trial"}, {Name: "CSA", Doc: "Conditioned stimulus A (e.g., Tone)"}, {Name: "CSB", Doc: "Conditioned stimulus B (e.g., Light)"}, {Name: "CSC", Doc: "Conditioned stimulus C"}, {Name: "US", Doc: "Unconditioned stimulus -- reward"}, {Name: "RewVal", Doc: "value for reward"}, {Name: "NoRewVal", Doc: "value for non-reward"}, {Name: "Input", Doc: "one-hot input representation of current option"}, {Name: "Reward", Doc: "single reward value"}, {Name: "Trial", Doc: "one trial is a pass through all TotTime Events"}, {Name: "Event", Doc: "event is one time step within Trial -- e.g., CS turning on, etc"}}})

var _ = types.AddType(&types.Type{Name: "main.Config", IDName: "config", Doc: "Config has config parameters related to running the sim", Fields: []types.Field{{Name: "NRuns", Doc: "total number of runs to do when running Train"}, {Name: "NEpochs", Doc: "total number of epochs per run"}, {Name: "NTrials", Doc: "total number of trials per epoch"}}})

var _ = types.AddType(&types.Type{Name: "main.Sim", IDName: "sim", Doc: "Sim encapsulates the entire simulation model, and we define all the\nfunctionality as methods on this struct.  This structure keeps all relevant\nstate information organized and available without having to pass everything around\nas arguments to methods, and provides the core GUI interface (note the view tags\nfor the fields which provide hints to how things should be displayed).", Fields: []types.Field{{Name: "Discount", Doc: "discount factor for future rewards"}, {Name: "Lrate", Doc: "learning rate"}, {Name: "Config", Doc: "Config contains misc configuration parameters for running the sim"}, {Name: "Net", Doc: "the network -- click to view / edit parameters for layers, paths, etc"}, {Name: "Params", Doc: "network parameter management"}, {Name: "Loops", Doc: "contains looper control loops for running sim"}, {Name: "Stats", Doc: "contains computed statistic values"}, {Name: "Logs", Doc: "Contains all the logs and information about the logs.'"}, {Name: "Envs", Doc: "Environments"}, {Name: "Context", Doc: "leabra timing parameters and state"}, {Name: "ViewUpdate", Doc: "netview update parameters"}, {Name: "GUI", Doc: "manages all the gui elements"}, {Name: "RandSeeds", Doc: "a list of random seeds to use for each run"}}})
