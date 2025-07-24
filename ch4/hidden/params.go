// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hidden

import (
	"github.com/emer/leabra/v2/leabra"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = leabra.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "needs some special inhibition and learning params",
			Set: func(ly *leabra.LayerParams) {
				ly.Learn.AvgL.Gain = 1.5
				ly.Inhib.Layer.Gi = 1.3
				ly.Inhib.ActAvg.Init = 0.5
				ly.Inhib.ActAvg.Fixed = true
				ly.Act.Gbar.L = 0.1
			}},
	},
	"Hebbian":     {},
	"ErrorDriven": {},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = leabra.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "basic path params",
			Set: func(pt *leabra.PathParams) {
				pt.Learn.Norm.On = false
				pt.Learn.Momentum.On = false
				pt.Learn.WtBal.On = false
			}},
		{Sel: ".BackPath", Doc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *leabra.PathParams) {
				pt.WtScale.Rel = 0.3
			}},
	},
	"Hebbian": {
		{Sel: "Path", Doc: "",
			Set: func(pt *leabra.PathParams) {
				pt.Learn.XCal.MLrn = 0
				pt.Learn.XCal.SetLLrn = true
				pt.Learn.XCal.LLrn = 1
			}},
	},
	"ErrorDriven": {
		{Sel: "Path", Doc: "",
			Set: func(pt *leabra.PathParams) {
				pt.Learn.XCal.MLrn = 1
				pt.Learn.XCal.SetLLrn = true
				pt.Learn.XCal.LLrn = 0
			}},
	},
}
