// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/etensor"
)

// LEDEnv generates images of old-school "LED" style "letters" composed of a set of horizontal
// and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
// Renders using SVG.
type LEDEnv struct {
	Nm    string  `desc:"name of this environment"`
	Dsc   string  `desc:"description of this environment"`
	Run   env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch env.Ctr `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial env.Ctr `view:"inline" desc:"trial is the step counter within epoch"`
}

func (fe *LEDEnv) Name() string { return fe.Nm }
func (fe *LEDEnv) Desc() string { return fe.Dsc }

// InitTMat initializes matrix and labels to given size
func (fe *LEDEnv) InitTMat(nst int) {
	fe.TMat.SetShape([]int{nst, nst}, nil, []string{"cur", "next"})
	fe.Labels.SetShape([]int{nst, nst}, nil, []string{"cur", "next"})
	fe.TMat.SetZeros()
	fe.Labels.SetZeros()
	fe.NNext.SetShape([]int{1}, nil, nil)
	fe.NextStates.SetShape([]int{nst}, nil, nil)
	fe.NextLabels.SetShape([]int{nst}, nil, nil)
}

// SetTMat sets given transition matrix probability and label
func (fe *LEDEnv) SetTMat(fm, to int, p float64, lbl string) {
	fe.TMat.Set([]int{fm, to}, p)
	fe.Labels.Set([]int{fm, to}, lbl)
}

// TMatReber sets the transition matrix to the standard Reber grammar FSA
func (fe *LEDEnv) TMatReber() {
	fe.InitTMat(8)
	fe.SetTMat(0, 1, 1, "B")   // 0 = start
	fe.SetTMat(1, 2, 0.5, "T") // 1 = state 0 in usu diagram (+1 for all states)
	fe.SetTMat(1, 3, 0.5, "P")
	fe.SetTMat(2, 2, 0.5, "S")
	fe.SetTMat(2, 4, 0.5, "X")
	fe.SetTMat(3, 3, 0.5, "T")
	fe.SetTMat(3, 5, 0.5, "V")
	fe.SetTMat(4, 6, 0.5, "S")
	fe.SetTMat(4, 3, 0.5, "X")
	fe.SetTMat(5, 6, 0.5, "V")
	fe.SetTMat(5, 4, 0.5, "P")
	fe.SetTMat(6, 7, 1, "E") // 7 = end
	fe.Init(0)
}

func (fe *LEDEnv) Validate() error {
	if fe.TMat.Len() == 0 {
		return fmt.Errorf("LEDEnv: %v has no transition matrix TMat set", fe.Nm)
	}
	return nil
}

func (fe *LEDEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (fe *LEDEnv) States() env.Elements {
	nst := fe.TMat.Dim(0)
	if nst < 2 {
		nst = 2 // at least usu
	}
	els := env.Elements{
		{"NNext", []int{1}, nil},
		{"NextStates", []int{nst}, []string{"nstates"}},
		{"NextLabels", []int{nst}, []string{"nstates"}},
	}
	return els
}

func (fe *LEDEnv) State(element string) etensor.Tensor {
	switch element {
	case "NNext":
		return &fe.NNext
	case "NextStates":
		return &fe.NextStates
	case "NextLabels":
		return &fe.NextLabels
	}
	return nil
}

func (fe *LEDEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string
func (fe *LEDEnv) String() string {
	nn := fe.NNext.Values[0]
	lbls := fe.NextLabels.Values[0:nn]
	return fmt.Sprintf("S_%d_%v", fe.CurState, lbls)
}

func (fe *LEDEnv) Init(run int) {
	fe.Run.Scale = env.Run
	fe.Epoch.Scale = env.Epoch
	fe.Trial.Scale = env.Trial
	fe.Run.Init()
	fe.Epoch.Init()
	fe.Seq.Init()
	fe.Trial.Init()
	fe.Run.Cur = run
	fe.Trial.Max = 0
	fe.Trial.Cur = -1 // init state -- key so that first Step() = 0
	fe.CurState = 0
	fe.PrvState = -1
}

// NextState sets NextStates including randomly chosen one at start
func (fe *LEDEnv) NextState() {
	nst := fe.TMat.Dim(0)
	if fe.CurState < 0 || fe.CurState >= nst-1 {
		fe.CurState = 0
	}
	ri := fe.CurState * nst
	ps := fe.TMat.Values[ri : ri+nst]
	ls := fe.Labels.Values[ri : ri+nst]
	nxt := erand.PChoose64(ps) // next state chosen at random
	fe.NextStates.Set1D(0, nxt)
	fe.NextLabels.Set1D(0, ls[nxt])
	idx := 1
	for i, p := range ps {
		if i != nxt && p > 0 {
			fe.NextStates.Set1D(idx, i)
			fe.NextLabels.Set1D(idx, ls[i])
			idx++
		}
	}
	fe.NNext.Set1D(0, idx)
	fe.PrvState = fe.CurState
	fe.CurState = nxt
}

func (fe *LEDEnv) Step() bool {
	fe.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	fe.NextState()
	fe.Trial.Incr()
	if fe.PrvState == 0 {
		if fe.Seq.Incr() {
			fe.Epoch.Incr()
		}
	}
	return true
}

func (fe *LEDEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (fe *LEDEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return fe.Run.Query()
	case env.Epoch:
		return fe.Epoch.Query()
	case env.Sequence:
		return fe.Seq.Query()
	case env.Trial:
		return fe.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*LEDEnv)(nil)
