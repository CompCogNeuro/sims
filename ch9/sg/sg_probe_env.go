// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
)

// ProbeEnv generates sentences using a grammar that is parsed from a
// text file.  The core of the grammar is rules with various items
// chosen at random during generation -- these items can be
// more rules terminal tokens.
type ProbeEnv struct {
	Nm        string          `desc:"name of this environment"`
	Dsc       string          `desc:"description of this environment"`
	Words     []string        `desc:"list of words used for activating state units according to index"`
	WordState etensor.Float32 `desc:"current sentence activation state"`
	Run       env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch     env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial     env.Ctr         `view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"`
}

func (ev *ProbeEnv) Name() string { return ev.Nm }
func (ev *ProbeEnv) Desc() string { return ev.Dsc }

// InitTMat initializes matrix and labels to given size
func (ev *ProbeEnv) Validate() error {
	return nil
}

func (ev *ProbeEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ev *ProbeEnv) States() env.Elements {
	els := env.Elements{
		{"Input", []int{len(ev.Words)}, nil},
	}
	return els
}

func (ev *ProbeEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.WordState
	}
	return nil
}

func (ev *ProbeEnv) Actions() env.Elements {
	return nil
}

func (ev *ProbeEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Trial.Max = len(ev.Words)
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0

	ev.WordState.SetShape([]int{len(ev.Words)}, nil, []string{"Words"})
}

// String returns the current state as a string
func (ev *ProbeEnv) String() string {
	if ev.Trial.Cur < len(ev.Words) {
		return fmt.Sprintf("%s", ev.Words[ev.Trial.Cur])
	} else {
		return ""
	}
}

// RenderState renders the current state
func (ev *ProbeEnv) RenderState() {
	ev.WordState.SetZeros()
	if ev.Trial.Cur > 0 && ev.Trial.Cur < len(ev.Words) {
		ev.WordState.SetFloat1D(ev.Trial.Cur, 1)
	}
}

func (ev *ProbeEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	ev.RenderState()
	return true
}

func (ev *ProbeEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *ProbeEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*ProbeEnv)(nil)
