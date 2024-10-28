// Code generated by "core generate -add-types"; DO NOT EDIT.

package main

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "main.EnvType", IDName: "env-type", Doc: "EnvType is the type of test environment"})

var _ = types.AddType(&types.Type{Name: "main.Config", IDName: "config", Doc: "Config has config parameters related to running the sim", Fields: []types.Field{{Name: "NRuns", Doc: "total number of runs to do when running Train"}, {Name: "NEpochs", Doc: "total number of epochs per run"}, {Name: "NTrials", Doc: "total number of trials for training"}, {Name: "NZero", Doc: "stop run after this number of perfect, zero-error epochs."}, {Name: "TestInterval", Doc: "how often to run through all the test patterns, in terms of training epochs.\ncan use 0 or -1 for no testing."}}})

var _ = types.AddType(&types.Type{Name: "main.Sim", IDName: "sim", Doc: "Sim encapsulates the entire simulation model, and we define all the\nfunctionality as methods on this struct.  This structure keeps all relevant\nstate information organized and available without having to pass everything around\nas arguments to methods, and provides the core GUI interface (note the view tags\nfor the fields which provide hints to how things should be displayed).", Fields: []types.Field{{Name: "TestingEnv", Doc: "the environment to use for testing -- only takes effect for TestAll."}, {Name: "Config", Doc: "simulation configuration parameters -- set by .toml config file and / or args"}, {Name: "Net", Doc: "the network -- click to view / edit parameters for layers, paths, etc"}, {Name: "Params", Doc: "all parameter management"}, {Name: "Train", Doc: "training patterns"}, {Name: "Probe", Doc: "probe patterns"}, {Name: "Besner", Doc: "nonword testing patterns"}, {Name: "Glushko", Doc: "nonword testing patterns"}, {Name: "Taraban", Doc: "nonword testing patterns"}, {Name: "PhonCons", Doc: "phonology consonant patterns"}, {Name: "PhonVowel", Doc: "phonology vowel patterns"}, {Name: "Loops", Doc: "contains looper control loops for running sim"}, {Name: "Stats", Doc: "contains computed statistic values"}, {Name: "Logs", Doc: "Contains all the logs and information about the logs.'"}, {Name: "Envs", Doc: "Environments"}, {Name: "Context", Doc: "leabra timing parameters and state"}, {Name: "ViewUpdate", Doc: "netview update parameters"}, {Name: "GUI", Doc: "manages all the gui elements"}, {Name: "RandSeeds", Doc: "a list of random seeds to use for each run"}}})
