// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "no extra learning factors, hebbian learning",
			Params: params.Params{
				"Path.Learn.Norm.On":      "false",
				"Path.Learn.Momentum.On":  "false",
				"Path.Learn.WtBal.On":     "false",
				"Path.Learn.XCal.MLrn":    "0", // pure hebb
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "1",
				"Path.Learn.WtSig.Gain":   "1", // key: more graded weights
			}},
		{Sel: "Layer", Desc: "needs some special inhibition and learning params",
			Params: params.Params{
				"Layer.Learn.AvgL.Gain":   "1", // this is critical! much lower
				"Layer.Learn.AvgL.Min":    "0.01",
				"Layer.Learn.AvgL.Init":   "0.2",
				"Layer.Inhib.Layer.Gi":    "2",
				"Layer.Inhib.Layer.FBTau": "3",
				"Layer.Inhib.ActAvg.Init": "0.2",
				"Layer.Act.Gbar.L":        "0.1",
				"Layer.Act.Noise.Dist":    "Gaussian",
				"Layer.Act.Noise.Var":     "0.02",
				"Layer.Act.Noise.Type":    "GeNoise",
				"Layer.Act.Noise.Fixed":   "false",
			}},
		{Sel: ".BackPath", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2",
			}},
		{Sel: ".ExciteLateral", Desc: "lateral excitatory connection",
			Params: params.Params{
				"Path.WtInit.Mean": ".5",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
				"Path.WtScale.Rel": "0.2",
			}},
		{Sel: ".InhibLateral", Desc: "lateral inhibitory connection",
			Params: params.Params{
				"Path.WtInit.Mean": "0",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
				"Path.WtScale.Abs": "0.2",
			}},
	},
}
