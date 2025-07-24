// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objrec

import "github.com/emer/leabra/v2/leabra"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = leabra.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "needs some special inhibition and learning params",
			Set: func(ly *leabra.LayerParams) {
			}},
		{Sel: "#V1", Doc: "pool inhib (not used), initial activity",
			Set: func(ly *leabra.LayerParams) {
			}},
		{Sel: "#V4", Doc: "pool inhib, sparse activity",
			Set: func(ly *leabra.LayerParams) {
			}},
		{Sel: "#IT", Doc: "initial activity",
			Set: func(ly *leabra.LayerParams) {
			}},
		{Sel: "#Output", Doc: "high inhib for one-hot output",
			Set: func(ly *leabra.LayerParams) {
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = leabra.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "",
			Set: func(pt *leabra.PathParams) {
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
			Set: func(pt *leabra.PathParams) {
			}},
	},
}

/*

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "yes extra learning factors",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "false", // not obviously beneficial, maybe worse
				"Path.Learn.Lrate":       "0.04",  // must set initial lrate here when using schedule!
				// "Path.WtInit.Sym":        "false", // slows first couple of epochs but then no diff
			}},
		{Sel: "Layer", Desc: "needs some special inhibition and learning params",
			Params: params.Params{
				"Layer.Learn.AvgL.Gain": "2.5", // standard
				"Layer.Act.Gbar.L":      "0.1", // more distributed activity with 0.1
			}},
		{Sel: ".BackPath", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
			Params: params.Params{
				"Path.WtScale.Rel": "0.1",
			}},
		{Sel: "#V1", Desc: "pool inhib (not used), initial activity",
			Params: params.Params{
				"Layer.Inhib.Pool.On":     "true", // clamped, so not relevant, but just in case
				"Layer.Inhib.ActAvg.Init": "0.1",
			}},
		{Sel: "#V4", Desc: "pool inhib, sparse activity",
			Params: params.Params{
				"Layer.Inhib.Pool.On":     "true", // needs pool-level
				"Layer.Inhib.ActAvg.Init": "0.05", // sparse
			}},
		{Sel: "#IT", Desc: "initial activity",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.1",
			}},
		{Sel: "#Output", Desc: "high inhib for one-hot output",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "2.8",
				"Layer.Inhib.ActAvg.Init": "0.05",
			}},
	},
	"NovelLearn": {
		{Sel: "Path", Desc: "lr = 0",
			Params: params.Params{
				"Path.Learn.Lrate":     "0",
				"Path.Learn.LrateInit": "0", // make sure for sched
			}},
		{Sel: ".NovLearn", Desc: "lr = 0.02",
			Params: params.Params{
				"Path.Learn.Lrate":     "0.02",
				"Path.Learn.LrateInit": "0.02", // double sure
			}},
	},
}

*/
